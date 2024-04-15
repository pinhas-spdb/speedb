// Copyright (C) 2023 Speedb Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "rocksdb/write_buffer_manager.h"

#include <array>
#include <memory>

#include "cache/cache_entry_roles.h"
#include "cache/cache_reservation_manager.h"
#include "db/db_impl/db_impl.h"
#include "logging/logging.h"
#include "monitoring/instrumented_mutex.h"
#include "rocksdb/status.h"
#include "rocksdb/write_controller.h"
#include "test_util/sync_point.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {

auto WriteBufferManager::FlushInitiationOptions::Sanitize() const
    -> FlushInitiationOptions {
  size_t sanitized_max_num_parallel_flushes = max_num_parallel_flushes;
  if (sanitized_max_num_parallel_flushes == 0) {
    sanitized_max_num_parallel_flushes = kDfltMaxNumParallelFlushes;
  }

  return FlushInitiationOptions(sanitized_max_num_parallel_flushes);
}

WriteBufferManager::WriteBufferManager(
    size_t _buffer_size, std::shared_ptr<Cache> cache, bool allow_stall,
    bool initiate_flushes,
    const FlushInitiationOptions& flush_initiation_options,
    uint16_t start_delay_percent)
    : buffer_size_(_buffer_size),
      mutable_limit_(buffer_size_ * kMutableLimit),
      memory_used_(0),
      memory_inactive_(0),
      memory_being_freed_(0U),
      cache_res_mgr_(nullptr),
      allow_stall_(allow_stall),
      start_delay_percent_(start_delay_percent),
      stall_active_(false),
      initiate_flushes_(initiate_flushes),
      flush_initiation_options_(flush_initiation_options.Sanitize()),
      flushes_mu_(new InstrumentedMutex),
      flushes_initiators_mu_(new InstrumentedMutex),
      flushes_wakeup_cv_(new InstrumentedCondVar(flushes_mu_.get())) {
  if (cache) {
    // Memtable's memory usage tends to fluctuate frequently
    // therefore we set delayed_decrease = true to save some dummy entry
    // insertion on memory increase right after memory decrease
    cache_res_mgr_ = std::make_shared<
        CacheReservationManagerImpl<CacheEntryRole::kWriteBuffer>>(
        cache, true /* delayed_decrease */);
  }

  if (initiate_flushes_) {
    InitFlushInitiationVars(buffer_size());
  }
  if (start_delay_percent_ >= 100) {
    // unsuitable value, sanitizing to default value.
    start_delay_percent_ = kDfltStartDelayPercentThreshold;
  }
}

WriteBufferManager::~WriteBufferManager() {
#ifndef NDEBUG
  std::unique_lock<std::mutex> lock(mu_);
  assert(queue_.empty());
#endif
  TerminateFlushesThread();
}

std::size_t WriteBufferManager::dummy_entries_in_cache_usage() const {
  if (cache_res_mgr_ != nullptr) {
    return cache_res_mgr_->GetTotalReservedCacheSize();
  } else {
    return 0;
  }
}

void WriteBufferManager::ReserveMem(size_t mem) {
  auto is_enabled = enabled();
  size_t new_memory_used = 0U;

  if (cache_res_mgr_ != nullptr) {
    new_memory_used = ReserveMemWithCache(mem);
  } else if (is_enabled) {
    auto old_memory_used =
        memory_used_.fetch_add(mem, std::memory_order_relaxed);
    new_memory_used = old_memory_used + mem;
  }
  for (auto loggers : loggers_to_client_ids_map_) {
    ROCKS_LOG_WARN(loggers.first,
                   "WBM (%p) ReserveMem called with:  %" PRIu64
                   " , memory_used: %" PRIu64,
                   this, mem, new_memory_used);
  }
  if (is_enabled) {
    UpdateUsageState(new_memory_used, static_cast<int64_t>(mem), buffer_size());
    // Checking outside the locks is not reliable, but avoids locking
    // unnecessarily which is expensive
    if (UNLIKELY(ShouldInitiateAnotherFlushMemOnly(new_memory_used))) {
      ReevaluateNeedForMoreFlushesNoLockHeld(new_memory_used);
    }
  }
}

// Should only be called from write thread
size_t WriteBufferManager::ReserveMemWithCache(size_t mem) {
  assert(cache_res_mgr_ != nullptr);
  // Use a mutex to protect various data structures. Can be optimized to a
  // lock-free solution if it ends up with a performance bottleneck.
  std::lock_guard<std::mutex> lock(cache_res_mgr_mu_);

  size_t new_mem_used = memory_used_.load(std::memory_order_relaxed) + mem;
  memory_used_.store(new_mem_used, std::memory_order_relaxed);
  Status s = cache_res_mgr_->UpdateCacheReservation(new_mem_used);

  // We absorb the error since WriteBufferManager is not able to handle
  // this failure properly. Ideallly we should prevent this allocation
  // from happening if this cache charging fails.
  // [TODO] We'll need to improve it in the future and figure out what to do on
  // error
  s.PermitUncheckedError();

  return new_mem_used;
}

void WriteBufferManager::ScheduleFreeMem(size_t mem) {
  if (enabled()) {
    memory_inactive_.fetch_add(mem, std::memory_order_relaxed);
  }
}

void WriteBufferManager::FreeMemBegin(size_t mem) {
  if (enabled()) {
    memory_being_freed_.fetch_add(mem, std::memory_order_relaxed);
  }
}

// Freeing 'mem' bytes was aborted and that memory is no longer in the process
// of being freed
void WriteBufferManager::FreeMemAborted(size_t mem) {
  if (enabled()) {
    [[maybe_unused]] const auto curr_memory_being_freed =
        memory_being_freed_.fetch_sub(mem, std::memory_order_relaxed);
    assert(curr_memory_being_freed >= mem);
  }
}

void WriteBufferManager::FreeMem(size_t mem) {
  const auto is_enabled = enabled();
  size_t new_memory_used = 0U;

  if (cache_res_mgr_ != nullptr) {
    new_memory_used = FreeMemWithCache(mem);
  } else if (is_enabled) {
    auto old_memory_used =
        memory_used_.fetch_sub(mem, std::memory_order_relaxed);
    assert(old_memory_used >= mem);
    new_memory_used = old_memory_used - mem;
  }
  for (auto loggers : loggers_to_client_ids_map_) {
    ROCKS_LOG_WARN(loggers.first,
                   "WBM (%p) FreeMem called with: %" PRIu64
                   ", memory_used: %" PRIu64,
                   this, mem, new_memory_used);
  }
  if (is_enabled) {
    [[maybe_unused]] const auto curr_memory_inactive =
        memory_inactive_.fetch_sub(mem, std::memory_order_relaxed);
    [[maybe_unused]] const auto curr_memory_being_freed =
        memory_being_freed_.fetch_sub(mem, std::memory_order_relaxed);

    assert(curr_memory_inactive >= mem);
    assert(curr_memory_being_freed >= mem);

    UpdateUsageState(new_memory_used, static_cast<int64_t>(-mem),
                     buffer_size());
  }

  // Check if stall is active and can be ended.
  MaybeEndWriteStall();

  if (is_enabled) {
    // Checking outside the locks is not reliable, but avoids locking
    // unnecessarily which is expensive
    if (UNLIKELY(ShouldInitiateAnotherFlushMemOnly(new_memory_used))) {
      ReevaluateNeedForMoreFlushesNoLockHeld(new_memory_used);
    }
  }
}

size_t WriteBufferManager::FreeMemWithCache(size_t mem) {
  assert(cache_res_mgr_ != nullptr);
  // Use a mutex to protect various data structures. Can be optimized to a
  // lock-free solution if it ends up with a performance bottleneck.
  std::lock_guard<std::mutex> lock(cache_res_mgr_mu_);

  const auto old_mem_used = memory_used_.load(std::memory_order_relaxed);
  assert(old_mem_used >= mem);
  size_t new_mem_used = old_mem_used - mem;
  memory_used_.store(new_mem_used, std::memory_order_relaxed);
  Status s = cache_res_mgr_->UpdateCacheReservation(new_mem_used);

  // We absorb the error since WriteBufferManager is not able to handle
  // this failure properly.
  // [TODO] We'll need to improve it in the future and figure out what to do on
  // error
  s.PermitUncheckedError();

  return new_mem_used;
}

void WriteBufferManager::BeginWriteStall(StallInterface* wbm_stall) {
  assert(wbm_stall != nullptr);
  assert(allow_stall_.load(std::memory_order_relaxed));

  // Allocate outside of the lock.
  std::list<StallInterface*> new_node = {wbm_stall};

  {
    std::unique_lock<std::mutex> lock(mu_);
    // Verify if the stall conditions are stil active.
    if (ShouldStall()) {
      stall_active_.store(true, std::memory_order_relaxed);
      queue_.splice(queue_.end(), std::move(new_node));
    }
  }

  // If the node was not consumed, the stall has ended already and we can signal
  // the caller.
  if (!new_node.empty()) {
    new_node.front()->Signal();
  }
}

// Called when memory is freed in FreeMem or the buffer size has changed.
void WriteBufferManager::MaybeEndWriteStall() {
  // Stall conditions have not been resolved.
  if (allow_stall_.load(std::memory_order_relaxed) &&
      IsStallThresholdExceeded()) {
    return;
  }

  // Perform all deallocations outside of the lock.
  std::list<StallInterface*> cleanup;

  std::unique_lock<std::mutex> lock(mu_);
  if (!stall_active_.load(std::memory_order_relaxed)) {
    return;  // Nothing to do.
  }

  // Unblock new writers.
  stall_active_.store(false, std::memory_order_relaxed);

  // Unblock the writers in the queue.
  for (StallInterface* wbm_stall : queue_) {
    wbm_stall->Signal();
  }
  cleanup = std::move(queue_);
}

void WriteBufferManager::RemoveDBFromQueue(StallInterface* wbm_stall) {
  assert(wbm_stall != nullptr);

  // Deallocate the removed nodes outside of the lock.
  std::list<StallInterface*> cleanup;

  if (enabled() && allow_stall_.load(std::memory_order_relaxed)) {
    std::unique_lock<std::mutex> lock(mu_);
    for (auto it = queue_.begin(); it != queue_.end();) {
      auto next = std::next(it);
      if (*it == wbm_stall) {
        cleanup.splice(cleanup.end(), queue_, std::move(it));
      }
      it = next;
    }
  }
  wbm_stall->Signal();
}

std::string WriteBufferManager::GetPrintableOptions() const {
  std::string ret;
  const int kBufferSize = 200;
  char buffer[kBufferSize];

  // The assumed width of the callers display code
  int field_width = 85;

  snprintf(buffer, kBufferSize, "%*s: %" ROCKSDB_PRIszt "\n", field_width,
           "wbm.size", buffer_size());
  ret.append(buffer);

  const Cache* cache = nullptr;
  if (cache_res_mgr_ != nullptr) {
    cache =
        static_cast<CacheReservationManagerImpl<CacheEntryRole::kWriteBuffer>*>(
            cache_res_mgr_.get())
            ->TEST_GetCache();
  }
  snprintf(buffer, kBufferSize, "%*s: %p\n", field_width, "wbm.cache", cache);
  ret.append(buffer);

  snprintf(buffer, kBufferSize, "%*s: %d\n", field_width, "wbm.allow_stalls",
           allow_stall_.load());
  ret.append(buffer);

  snprintf(buffer, kBufferSize, "%*s: %d\n", field_width,
           "wbm.start_delay_percent", start_delay_percent_);
  ret.append(buffer);

  snprintf(buffer, kBufferSize, "%*s: %d\n", field_width,
           "wbm.initiate_flushes", IsInitiatingFlushes());
  ret.append(buffer);

  return ret;
}

WriteBufferManager::WBMClientId WriteBufferManager::RegisterWCAndLogger(
    std::shared_ptr<WriteController> wc, std::shared_ptr<Logger> logger) {
  uint64_t client_id = 0;
  {
    std::lock_guard<std::mutex> lock(controllers_map_mutex_);
    // make sure we haven`t wrapped around
    assert(next_client_id_ != std::numeric_limits<uint64_t>::max());
    client_id = next_client_id_++;
    controllers_to_client_ids_map_[wc].insert(client_id);
    controllers_to_loggers_map_[wc.get()].insert(logger.get());
  }
  {
    std::lock_guard<std::mutex> lock(loggers_map_mutex_);
    loggers_to_client_ids_map_[logger].insert(client_id);
  }
  return client_id;
}

template <typename SharedPtrType>
bool WriteBufferManager::RemoveFromMap(
    const SharedPtrType& ptr, WBMClientId wbm_client_id, std::mutex& mutex,
    std::unordered_map<SharedPtrType, WBMClientIds>& map) {
  std::lock_guard<std::mutex> lock(mutex);
  assert(map.count(ptr));
  assert(map[ptr].empty() == false);
  assert(map[ptr].count(wbm_client_id));
  map[ptr].erase(wbm_client_id);
  if (map[ptr].empty()) {
    map.erase(ptr);
    return true;
  } else {
    return false;
  }
}

void WriteBufferManager::DeregisterWCAndLogger(
    std::shared_ptr<WriteController> wc, std::shared_ptr<Logger> logger,
    WBMClientId wbm_client_id) {
  // value of 0 means the wc and logger weren`t registered.
  assert(wbm_client_id > 0);
  bool last_logger = RemoveFromMap(logger, wbm_client_id, loggers_map_mutex_,
                                   loggers_to_client_ids_map_);
  bool last_controller =
      RemoveFromMap(wc, wbm_client_id, controllers_map_mutex_,
                    controllers_to_client_ids_map_);
  std::lock_guard<std::mutex> lock(controllers_map_mutex_);
  if (last_controller) {
    // the db calling this should still have a ref to this wc
    assert(wc.unique() == false);
    if (wc->is_dynamic_delay()) {
      wc->HandleRemoveDelayReq(this);
    }
    controllers_to_loggers_map_.erase(wc.get());
  } else if (last_logger) {
    assert(controllers_to_loggers_map_.count(wc.get()));
    controllers_to_loggers_map_[wc.get()].erase(logger.get());
  }
}

namespace {

// highest delay factor is kMaxDelayedWriteFactor - 1 and the write rate is:
// max_write_rate * (kMaxDelayedWriteFactor - factor / kMaxDelayedWriteFactor)
uint64_t CalcDelayFactor(size_t quota, size_t updated_memory_used,
                         size_t usage_start_delay_threshold) {
  assert(updated_memory_used >= usage_start_delay_threshold);
  double extra_used_memory = updated_memory_used - usage_start_delay_threshold;
  double max_used_memory = quota - usage_start_delay_threshold;

  uint64_t delay_factor = (extra_used_memory / max_used_memory) *
                          WriteBufferManager::kMaxDelayedWriteFactor;
  if (delay_factor < 1U) {
    delay_factor = 1U;
  }
  return delay_factor;
}

uint64_t CalcDelayFromFactor(uint64_t max_write_rate, uint64_t delay_factor) {
  assert(delay_factor > 0U);
  auto wbm_write_rate = max_write_rate;
  if (max_write_rate >= WriteController::kMinWriteRate) {
    // If user gives rate less than kMinWriteRate, don't adjust it.
    assert(delay_factor <= WriteBufferManager::kMaxDelayedWriteFactor);
    auto write_rate_factor =
        static_cast<double>(WriteBufferManager::kMaxDelayedWriteFactor -
                            delay_factor) /
        WriteBufferManager::kMaxDelayedWriteFactor;
    wbm_write_rate = max_write_rate * write_rate_factor;
    if (wbm_write_rate < WriteController::kMinWriteRate) {
      wbm_write_rate = WriteController::kMinWriteRate;
    }
  }

  return wbm_write_rate;
}

}  // Unnamed Namespace

void WriteBufferManager::WBMSetupDelay(
    uint64_t delay_factor, WriteController* wc,
    const std::unordered_set<Logger*>& loggers) {
  // the final rate depends on the WC max rate so each WC can receive a
  // different delay requirement.
  const uint64_t wbm_write_rate =
      CalcDelayFromFactor(wc->max_delayed_write_rate(), delay_factor);

  for (auto logger : loggers) {
    ROCKS_LOG_WARN(logger,
                   "WBM (%p) sets a delay requirement of %" PRIu64
                   " using WC - %p",
                   this, wbm_write_rate, wc);
  }

  wc->HandleNewDelayReq(this, wbm_write_rate);
}

void WriteBufferManager::ResetDelay(
    UsageState usage_state, WriteController* wc,
    const std::unordered_set<Logger*>& loggers) {
  auto usage_state_str = "No Delay";
  if (usage_state == UsageState::kStop) {
    usage_state_str = "Max memory reached";
  }

  for (auto logger : loggers) {
    ROCKS_LOG_WARN(logger,
                   "WBM (%p) resets its delay requirement using WC - %p. "
                   "UsageState is: %s",
                   this, wc, usage_state_str);
  }

  wc->HandleRemoveDelayReq(this);
}

void WriteBufferManager::UpdateControllerDelayState() {
  const auto [usage_state, delay_factor] = GetUsageStateInfo();
  std::lock_guard<std::mutex> lock(controllers_map_mutex_);
  for (auto& wc_and_client_ids : controllers_to_client_ids_map_) {
    // make sure that controllers_to_client_ids_map_ does not hold the last ref
    // to the WC since holding the last ref means that the last DB that was
    // using this WC has destructed and using this WC is no longer valid.
    assert(wc_and_client_ids.first.unique() == false);
    WriteController* wc = wc_and_client_ids.first.get();
    if (wc && wc->is_dynamic_delay()) {
      const auto& loggers = controllers_to_loggers_map_[wc];
      if (usage_state == UsageState::kDelay) {
        WBMSetupDelay(delay_factor, wc, loggers);
      } else {
        ResetDelay(usage_state, wc, loggers);
      }
    }
  }
}

uint64_t WriteBufferManager::CalcNewCodedUsageState(
    size_t new_memory_used, int64_t memory_changed_size, size_t quota,
    uint64_t old_coded_usage_state) {
  auto [old_usage_state, old_delay_factor] =
      ParseCodedUsageState(old_coded_usage_state);

  auto new_usage_state = old_usage_state;
  auto new_delay_factor = old_delay_factor;
  size_t usage_start_delay_threshold = (start_delay_percent_ / 100.0) * quota;
  auto step_size =
      (quota - usage_start_delay_threshold) / kMaxDelayedWriteFactor;

  if (new_memory_used < usage_start_delay_threshold) {
    new_usage_state = WriteBufferManager::UsageState::kNone;
  } else if (new_memory_used >= quota) {
    new_usage_state = WriteBufferManager::UsageState::kStop;
  } else {
    new_usage_state = WriteBufferManager::UsageState::kDelay;
  }

  auto calc_new_delay_factor = false;

  if (new_usage_state != old_usage_state) {
    if (new_usage_state == WriteBufferManager::UsageState::kDelay) {
      calc_new_delay_factor = true;
    }
  } else if (new_usage_state == WriteBufferManager::UsageState::kDelay) {
    if (memory_changed_size == 0) {
      calc_new_delay_factor = true;
    } else {
      auto old_memory_used = new_memory_used - memory_changed_size;
      // Calculate & notify only if the memory usage changed "steps"
      if ((old_memory_used / step_size) != (new_memory_used / step_size)) {
        calc_new_delay_factor = true;
      }
    }
  }

  if (calc_new_delay_factor) {
    new_delay_factor =
        CalcDelayFactor(quota, new_memory_used, usage_start_delay_threshold);
  }

  return CalcCodedUsageState(new_usage_state, new_delay_factor);
}

uint64_t WriteBufferManager::CalcCodedUsageState(UsageState usage_state,
                                                 uint64_t delay_factor) {
  switch (usage_state) {
    case UsageState::kNone:
      return kNoneCodedUsageState;
    case UsageState::kDelay:
      assert((delay_factor > kNoneCodedUsageState) &&
             (delay_factor <= kStopCodedUsageState));

      if (delay_factor <= kNoneCodedUsageState) {
        return kNoneCodedUsageState + 1;
      } else if (delay_factor > kStopCodedUsageState) {
        delay_factor = kStopCodedUsageState;
      }
      return delay_factor;
    case UsageState::kStop:
      return kStopCodedUsageState;
    default:
      assert(0);
      // We should never get here (BUG).
      return kNoneCodedUsageState;
  }
}

auto WriteBufferManager::ParseCodedUsageState(uint64_t coded_usage_state)
    -> std::pair<UsageState, uint64_t> {
  if (coded_usage_state <= kNoneCodedUsageState) {
    return {UsageState::kNone, kNoDelayedWriteFactor};
  } else if (coded_usage_state < kStopCodedUsageState) {
    return {UsageState::kDelay, coded_usage_state};
  } else {
    return {UsageState::kStop, kStopDelayedWriteFactor};
  }
}

void WriteBufferManager::UpdateUsageState(size_t new_memory_used,
                                          int64_t memory_changed_size,
                                          size_t quota) {
  assert(enabled());
  if (allow_stall_.load(std::memory_order_relaxed) == false) {
    return;
  }

  auto done = false;
  auto old_coded_usage_state = coded_usage_state_.load();
  auto new_coded_usage_state = old_coded_usage_state;
  while (done == false) {
    new_coded_usage_state = CalcNewCodedUsageState(
        new_memory_used, memory_changed_size, quota, old_coded_usage_state);

    if (old_coded_usage_state != new_coded_usage_state) {
      // Try to update the usage state with the usage state calculated by the
      // current thread. Failure (done == false) means one or
      // more threads have updated the current state, rendering our own
      // calculation irrelevant. In case done == false,
      // old_coded_usage_state will be the value of the state that was updated
      // by the other thread(s).
      done = coded_usage_state_.compare_exchange_weak(old_coded_usage_state,
                                                      new_coded_usage_state);
      if (done == false) {
        // Retry. However,
        new_memory_used = memory_usage();
        memory_changed_size = 0;
      } else {
        // WBM state has changed. need to update the WCs.
        UpdateControllerDelayState();
      }
    } else {
      done = true;
    }
  }
}

// =============================================================================
void WriteBufferManager::RegisterFlushInitiator(
    void* initiator, InitiateFlushRequestCb request) {
  {
    InstrumentedMutexLock lock(flushes_initiators_mu_.get());
    assert(FindInitiator(initiator) == kInvalidInitiatorIdx);

    flush_initiators_.push_back({initiator, request});
    if (flush_initiators_.size() == 1) {
      assert(next_candidate_initiator_idx_ == kInvalidInitiatorIdx);
      next_candidate_initiator_idx_ = 0U;
    }

    assert(next_candidate_initiator_idx_ < flush_initiators_.size());
  }

  // flushes_initiators_mu_ is held but not flushes_mu_
  WakeupFlushInitiationThreadNoLockHeld();
}

void WriteBufferManager::DeregisterFlushInitiator(void* initiator) {
  InstrumentedMutexLock lock(flushes_initiators_mu_.get());
  auto initiator_idx = FindInitiator(initiator);
  assert(IsInitiatorIdxValid(initiator_idx));

  flush_initiators_.erase(flush_initiators_.begin() + initiator_idx);

  // If the deregistered initiator was the next candidate and also the last
  // one, update the next candidate (possibly none left)
  assert(next_candidate_initiator_idx_ != kInvalidInitiatorIdx);
  if (next_candidate_initiator_idx_ >= flush_initiators_.size()) {
    UpdateNextCandidateInitiatorIdx();
  }

  // No need to wake up the flush initiation thread
}

void WriteBufferManager::InitFlushInitiationVars(size_t quota) {
  assert(initiate_flushes_);

  {
    InstrumentedMutexLock lock(flushes_mu_.get());
    additional_flush_step_size_ =
        quota * kStartFlushPercentThreshold / 100 /
        flush_initiation_options_.max_num_parallel_flushes;
    flush_initiation_start_size_ = additional_flush_step_size_;
    min_mutable_flush_size_ = std::min<size_t>(
        quota / (2 * flush_initiation_options_.max_num_parallel_flushes),
        64 * (1 << 20));
    RecalcFlushInitiationSize();
  }

  if (flushes_thread_.joinable() == false) {
    flushes_thread_ =
        port::Thread(&WriteBufferManager::InitiateFlushesThread, this);
  }
}

void WriteBufferManager::InitiateFlushesThread() {
  while (true) {
    // Should return true when the waiting should stop (no spurious wakeups
    // guaranteed)
    auto StopWaiting = [this]() {
      return (new_flushes_wakeup_ &&
              (terminate_flushes_thread_ || (num_flushes_to_initiate_ > 0U)));
    };

    InstrumentedMutexLock lock(flushes_mu_.get());
    while (StopWaiting() == false) {
      flushes_wakeup_cv_->Wait();
    }

    new_flushes_wakeup_ = false;

    if (terminate_flushes_thread_) {
      break;
    }

    // The code below tries to initiate num_flushes_to_initiate_ flushes by
    // invoking its registered initiators, and requesting them to initiate a
    // flush of a certain minimum size. The initiation is done in iterations. An
    // iteration is an attempt to give evey initiator an opportunity to flush,
    // in a round-robin ordering. An initiator may or may not be able to
    // initiate a flush. Reasons for not initiating could be:
    // - The flush is less than the specified minimum size.
    // - The initiator is in the process of shutting down or being disposed of.
    //
    // The assumption is that in case flush initiation stopped when
    // num_flushes_to_initiate_ == 0, there will be some future event that will
    // wake up this thread and initiation attempts will be retried:
    // - Initiator will be enabled
    // - A flush in progress will end
    // - The memory_used() will increase above additional_flush_initiation_size_

    // Two iterations:
    // 1. Flushes of a min size.
    // 2. Flushes of any size
    constexpr size_t kNumIters = 2U;
    const std::array<size_t, kNumIters> kMinFlushSizes{min_mutable_flush_size_,
                                                       0U};

    auto iter = 0U;
    while ((iter < kMinFlushSizes.size()) && (num_flushes_to_initiate_ > 0U)) {
      auto num_repeated_failures_to_initiate = 0U;
      while (num_flushes_to_initiate_ > 0U) {
        bool was_flush_initiated = false;
        {
          // Below an initiator is requested to initate a flush. The initiator
          // may call another WBM method that relies on these counters. The
          // counters are updated here, while under the flushes_mu_ lock
          // (released below) to ensure num_flushes_to_initiate_ can't become
          // negative Not recalculating flush initiation size since the
          // increment & decrement cancel each other with respect to the recalc.
          ++num_running_flushes_;
          assert(num_flushes_to_initiate_ > 0U);
          --num_flushes_to_initiate_;

          // Unlocking the flushed_mu_ since flushing (via the initiator cb) may
          // call a WBM service (e.g., ReserveMem()), that, in turn, needs to
          // flushes_mu_lock the same mutex => will get stuck
          InstrumentedMutexUnlock flushes_mu_unlocker(flushes_mu_.get());

          InstrumentedMutexLock initiators_lock(flushes_initiators_mu_.get());
          // Once we are under the flushes_initiators_mu_ lock, we may check:
          // 1. Has the last initiator deregistered?
          // 2. Have all existing initiators failed to initiate a flush?
          if (flush_initiators_.empty() ||
              (num_repeated_failures_to_initiate >= flush_initiators_.size())) {
            // No flush was initiated => undo the counters update
            assert(num_running_flushes_ > 0U);
            --num_running_flushes_;
            ++num_flushes_to_initiate_;
            break;
          }
          assert(IsInitiatorIdxValid(next_candidate_initiator_idx_));
          auto& initiator = flush_initiators_[next_candidate_initiator_idx_];
          UpdateNextCandidateInitiatorIdx();

          // TODO: Use a weak-pointer for the registered initiators. That would
          // allow us to release the flushes_initiators_mu_ mutex before calling
          // the callback (which may take a long time).
          was_flush_initiated = initiator.cb(kMinFlushSizes[iter]);
        }

        if (!was_flush_initiated) {
          // No flush was initiated => undo the counters update
          assert(num_running_flushes_ > 0U);
          --num_running_flushes_;
          ++num_flushes_to_initiate_;
          ++num_repeated_failures_to_initiate;
        } else {
          num_repeated_failures_to_initiate = 0U;
        }
      }
      ++iter;
    }
    TEST_SYNC_POINT_CALLBACK(
        "WriteBufferManager::InitiateFlushesThread::DoneInitiationsAttempt",
        &num_flushes_to_initiate_);
  }
}

void WriteBufferManager::TerminateFlushesThread() {
  {
    flushes_mu_->Lock();

    terminate_flushes_thread_ = true;
    WakeupFlushInitiationThreadLockHeld();
  }

  if (flushes_thread_.joinable()) {
    flushes_thread_.join();
  }
}

void WriteBufferManager::FlushStarted(bool wbm_initiated) {
  // num_running_flushes_ is incremented in our thread when initiating flushes
  // => Already accounted for
  if (wbm_initiated || !enabled()) {
    return;
  }

  flushes_mu_->Lock();

  ++num_running_flushes_;
  // Any number of non-wbm-initiated flushes may be initiated, so, we must not
  // underflow num_flushes_to_initiate_
  if (num_flushes_to_initiate_ > 0U) {
    --num_flushes_to_initiate_;
  }

  size_t curr_memory_used = memory_usage();
  RecalcFlushInitiationSize();
  ReevaluateNeedForMoreFlushesLockHeld(curr_memory_used);
}

void WriteBufferManager::FlushEnded(bool /* wbm_initiated */) {
  if (!enabled()) {
    return;
  }

  flushes_mu_->Lock();

  // The WBM may be enabled after a flush has started. In that case
  // the WBM will not be aware of the number of running flushes at the time
  // it is enabled. The counter will become valid once all of the flushes
  // that were running when it was enabled will have completed.
  assert(num_running_flushes_ > 0U);
  --num_running_flushes_;

  size_t curr_memory_used = memory_usage();
  RecalcFlushInitiationSize();
  ReevaluateNeedForMoreFlushesLockHeld(curr_memory_used);
}

void WriteBufferManager::RecalcFlushInitiationSize() {
  flushes_mu_->AssertHeld();

  if (num_running_flushes_ + num_flushes_to_initiate_ >=
      flush_initiation_options_.max_num_parallel_flushes) {
    additional_flush_initiation_size_ = buffer_size();
  } else {
    additional_flush_initiation_size_ =
        flush_initiation_start_size_ +
        additional_flush_step_size_ *
            (num_running_flushes_ + num_flushes_to_initiate_);
  }
}

void WriteBufferManager::ReevaluateNeedForMoreFlushesNoLockHeld(
    size_t curr_memory_used) {
  flushes_mu_->Lock();
  ReevaluateNeedForMoreFlushesLockHeld(curr_memory_used);
}

void WriteBufferManager::ReevaluateNeedForMoreFlushesLockHeld(
    size_t curr_memory_used) {
  assert(enabled());
  flushes_mu_->AssertHeld();

  if (ShouldInitiateAnotherFlush(curr_memory_used)) {
    // need to schedule more
    ++num_flushes_to_initiate_;
    RecalcFlushInitiationSize();
    WakeupFlushInitiationThreadLockHeld();
  } else {
    flushes_mu_->Unlock();
  }
}

uint64_t WriteBufferManager::FindInitiator(void* initiator) const {
  flushes_initiators_mu_->AssertHeld();

  for (auto i = 0U; i < flush_initiators_.size(); ++i) {
    if (flush_initiators_[i].initiator == initiator) {
      return i;
    }
  }

  return kInvalidInitiatorIdx;
}

void WriteBufferManager::WakeupFlushInitiationThreadNoLockHeld() {
  flushes_mu_->Lock();
  WakeupFlushInitiationThreadLockHeld();
}

// Assumed the lock is held
// Releases the lock upon exit
void WriteBufferManager::WakeupFlushInitiationThreadLockHeld() {
  flushes_mu_->AssertHeld();

  new_flushes_wakeup_ = true;

  // Done modifying the shared data. Release the lock so that when the flush
  // initiation thread it may acquire the mutex immediately
  flushes_mu_->Unlock();
  flushes_wakeup_cv_->Signal();
}

void WriteBufferManager::UpdateNextCandidateInitiatorIdx() {
  flushes_initiators_mu_->AssertHeld();

  if (flush_initiators_.empty() == false) {
    if (next_candidate_initiator_idx_ != kInvalidInitiatorIdx) {
      next_candidate_initiator_idx_ =
          ((next_candidate_initiator_idx_ + 1) % flush_initiators_.size());
    } else {
      next_candidate_initiator_idx_ = 0U;
    }
  } else {
    next_candidate_initiator_idx_ = kInvalidInitiatorIdx;
  }
}

bool WriteBufferManager::IsInitiatorIdxValid(uint64_t initiator_idx) const {
  flushes_initiators_mu_->AssertHeld();

  return (initiator_idx < flush_initiators_.size());
}

void WriteBufferManager::TEST_WakeupFlushInitiationThread() {
  WakeupFlushInitiationThreadNoLockHeld();
}

}  // namespace ROCKSDB_NAMESPACE
