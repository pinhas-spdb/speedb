// Copyright (C) 2022 Speedb Ltd. All rights reserved.
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

#ifndef ROCKSDB_LITE

#include "plugin/speedb/memtable/hash_spd_rep.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>  // std::condition_variable
#include <list>
#include <vector>

#include "db/memtable.h"
#include "memory/arena.h"
#include "memtable/stl_wrappers.h"
#include "monitoring/histogram.h"
#include "plugin/speedb/memtable/spdb_sorted_vector.h"
#include "port/port.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/slice.h"
#include "rocksdb/slice_transform.h"
#include "rocksdb/utilities/options_type.h"
#include "util/hash.h"
#include "util/heap.h"
#include "util/murmurhash.h"

namespace ROCKSDB_NAMESPACE {
namespace {

using namespace stl_wrappers;

constexpr size_t kMergedVectorsMax = 8;

struct SpdbKeyHandle {
  SpdbKeyHandle* next;
  char key[1];
};

struct BucketHeader {
  SpdbKeyHandle* items_;

  BucketHeader() : items_(nullptr) {}

  bool Contains(const MemTableRep::KeyComparator& comparator,
                const char* check_key) const {
    for (auto k = items_; k != nullptr; k = k->next) {
      const int cmp_res = comparator(k->key, check_key);
      if (cmp_res == 0) {
        return true;
      }
      if (cmp_res > 0) {
        break;
      }
    }
    return false;
  }

  bool Add(SpdbKeyHandle* val, const MemTableRep::KeyComparator& comparator) {
    SpdbKeyHandle** loc = &items_;

    for (; *loc != nullptr; loc = &(*loc)->next) {
      const int cmp_res = comparator((*loc)->key, val->key);
      if (cmp_res == 0) {
        return false;
      }
      if (cmp_res > 0) {
        break;
      }
    }

    val->next = *loc;
    *loc = val;
    return true;
  }
};

struct SpdbHashTable {
  std::vector<BucketHeader> buckets_;
  std::vector<port::Mutex> mutexes_;

  SpdbHashTable(size_t n_buckets, size_t n_mutexes)
      : buckets_(n_buckets), mutexes_(n_mutexes) {}

  bool Add(SpdbKeyHandle* val, const MemTableRep::KeyComparator& comparator) {
    auto mutex_and_bucket = GetMutexAndBucket(val->key, comparator);
    BucketHeader* bucket = mutex_and_bucket.second;
    MutexLock l(mutex_and_bucket.first);
    return bucket->Add(val, comparator);
  }

  bool Contains(const char* check_key,
                const MemTableRep::KeyComparator& comparator) const {
    auto mutex_and_bucket = GetMutexAndBucket(check_key, comparator);
    BucketHeader* bucket = mutex_and_bucket.second;
    MutexLock l(mutex_and_bucket.first);
    return bucket->Contains(comparator, check_key);
  }

  void Get(const LookupKey& k, const MemTableRep::KeyComparator& comparator,
           void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) const {
    auto mutex_and_bucket = GetMutexAndBucket(k.internal_key(), comparator);
    MutexLock l(mutex_and_bucket.first);

    auto iter = mutex_and_bucket.second->items_;

    for (; iter != nullptr; iter = iter->next) {
      if (comparator(iter->key, k.internal_key()) >= 0) {
        break;
      }
    }

    for (; iter != nullptr; iter = iter->next) {
      if (!callback_func(callback_args, iter->key)) {
        break;
      }
    }
  }

 private:
  static size_t GetHash(const Slice& user_key_without_ts) {
    return MurmurHash(user_key_without_ts.data(),
                      static_cast<int>(user_key_without_ts.size()), 0);
  }

  static Slice UserKeyWithoutTimestamp(
      const Slice internal_key, const MemTableRep::KeyComparator& compare) {
    auto key_comparator = static_cast<const MemTable::KeyComparator*>(&compare);
    const Comparator* user_comparator =
        key_comparator->comparator.user_comparator();
    const size_t ts_sz = user_comparator->timestamp_size();
    return ExtractUserKeyAndStripTimestamp(internal_key, ts_sz);
  }

  std::pair<port::Mutex*, BucketHeader*> GetMutexAndBucket(
      const char* key, const MemTableRep::KeyComparator& comparator) const {
    return GetMutexAndBucket(comparator.decode_key(key), comparator);
  }

  std::pair<port::Mutex*, BucketHeader*> GetMutexAndBucket(
      const Slice& internal_key,
      const MemTableRep::KeyComparator& comparator) const {
    const size_t hash =
        GetHash(UserKeyWithoutTimestamp(internal_key, comparator));
    port::Mutex* mutex =
        const_cast<port::Mutex*>(&mutexes_[hash % mutexes_.size()]);
    BucketHeader* bucket =
        const_cast<BucketHeader*>(&buckets_[hash % buckets_.size()]);
    return {mutex, bucket};
  }
};

// SpdbVector implemntation

bool SpdbVector::Add(const char* key) {
  const size_t location = n_elements_.fetch_add(1, std::memory_order_relaxed);
  if (location < items_.size()) {
    items_[location] = key;
    return true;
  }
  return false;
}

bool SpdbVector::Sort(const MemTableRep::KeyComparator& comparator) {
  if (n_elements_ == 0) {
    return false;
  }
  if (sorted_.load(std::memory_order_relaxed)) {
    return true;
  }

  MutexLock l(&mutex_);
  if (!sorted_.load(std::memory_order_acquire)) {
    const size_t num_elements = std::min(n_elements_.load(), items_.size());
    n_elements_.store(num_elements);
    if (num_elements < items_.size()) {
      items_.resize(num_elements);
    }
    std::sort(items_.begin(), items_.end(), Compare(comparator));
    sorted_.store(true, std::memory_order_release);
  }
  return true;
}

SpdbVector::Iterator SpdbVector::Seek(
    const MemTableRep::KeyComparator& comparator, const Slice* seek_key) {
  if (!IsEmpty()) {
    assert(sorted_);
    if (seek_key == nullptr || comparator(items_.front(), *seek_key) >= 0) {
      return items_.begin();
    } else if (comparator(items_.back(), *seek_key) >= 0) {
      return std::lower_bound(items_.begin(), items_.end(), *seek_key,
                              Compare(comparator));
    }
  }
  return items_.end();
}

SpdbVector::Iterator SpdbVector::SeekBackword(
    const MemTableRep::KeyComparator& comparator, const Slice* seek_key) {
  if (!IsEmpty()) {
    assert(sorted_);
    if (seek_key == nullptr || comparator(items_.back(), *seek_key) <= 0) {
      return std::prev(items_.end());
    } else if (comparator(items_.front(), *seek_key) <= 0) {
      auto ret = std::lower_bound(items_.begin(), items_.end(), *seek_key,
                                  Compare(comparator));
      if (comparator(*ret, *seek_key) > 0) {
        --ret;
      }
      return ret;
    }
  }
  return items_.end();
}

SpdbVector::Iterator SpdbVector::Seek(
    const MemTableRep::KeyComparator& comparator, const Slice* seek_key,
    SeekOption seek_op) {
  assert(sorted_);
  SpdbVector::Iterator ret;
  switch (seek_op) {
    case SEEK_INIT_FORWARD_OP:
    case SEEK_SWITCH_FORWARD_OP:
      ret = Seek(comparator, seek_key);
      break;
    case SEEK_INIT_BACKWARD_OP:
    case SEEK_SWITCH_BACKWARD_OP:
      ret = SeekBackword(comparator, seek_key);
      break;
  }
  return ret;
}

// SpdbVectorContainer implemanmtation
bool SpdbVectorContainer::InternalInsert(const char* key) {
  return spdb_vectors_.back()->Add(key);
}

void SpdbVectorContainer::Insert(const char* key) {
  num_elements_.fetch_add(1, std::memory_order_relaxed);
  {
    ReadLock rl(&spdb_vectors_rwlock_);

    if (InternalInsert(key)) {
      return;
    }
  }

  // add wasnt completed. need to add new add vector
  bool notify_sort_thread = false;
  {
    WriteLock wl(&spdb_vectors_rwlock_);

    if (InternalInsert(key)) {
      return;
    }

    SpdbVectorPtr spdb_vector(new SpdbVector(switch_spdb_vector_limit_));
    spdb_vectors_.push_back(spdb_vector);
    notify_sort_thread = true;

    if (!InternalInsert(key)) {
      assert(false);
      return;
    }
  }
  if (notify_sort_thread) {
    sort_thread_cv_.notify_one();
  }
}
bool SpdbVectorContainer::IsEmpty() const { return num_elements_.load() == 0; }

// copy the list of vectors to the iter_anchors
bool SpdbVectorContainer::InitIterator(IterAnchors& iter_anchor) {
  bool immutable = immutable_.load();
  if (!immutable) {
    spdb_vectors_rwlock_.WriteLock();
  }

  auto last_iter = std::prev(spdb_vectors_.end());
  bool notify_sort_thread = false;
  if (!immutable) {
    if (!(*last_iter)->IsEmpty()) {
      SpdbVectorPtr spdb_vector(new SpdbVector(switch_spdb_vector_limit_));
      spdb_vectors_.push_back(spdb_vector);
      notify_sort_thread = true;
    } else {
      --last_iter;
    }
  }
  ++last_iter;
  InitIterator(iter_anchor, spdb_vectors_.begin(), last_iter);
  if (!immutable) {
    spdb_vectors_rwlock_.WriteUnlock();
    if (notify_sort_thread) {
      sort_thread_cv_.notify_one();
    }
  }
  return true;
}

void SpdbVectorContainer::InitIterator(
    IterAnchors& iter_anchor, std::list<SpdbVectorPtr>::iterator start,
    std::list<SpdbVectorPtr>::iterator last) {
  for (auto iter = start; iter != last; ++iter) {
    SortHeapItem* item = new SortHeapItem(*iter, (*iter)->End());
    iter_anchor.push_back(item);
  }
}

void SpdbVectorContainer::SeekIter(const IterAnchors& iter_anchor,
                                   IterHeapInfo* iter_heap_info,
                                   const Slice* seek_key, SeekOption seek_op) {
  iter_heap_info->Reset(seek_op == SEEK_INIT_FORWARD_OP ||
                        seek_op == SEEK_SWITCH_FORWARD_OP);
  for (auto const& iter : iter_anchor) {
    if (iter->spdb_vector_->Sort(comparator_)) {
      iter->curr_iter_ =
          iter->spdb_vector_->Seek(comparator_, seek_key, seek_op);
      if (iter->Valid()) {
        iter_heap_info->Insert(iter);
      }
    }
  }
}

void SpdbVectorContainer::Merge(std::list<SpdbVectorPtr>::iterator& begin,
                                std::list<SpdbVectorPtr>::iterator& end) {
  SpdbVectorIterator iterator(this, comparator_, begin, end);
  const size_t num_elements = std::accumulate(
      begin, end, 0,
      [](size_t n, const SpdbVectorPtr& vec) { return n + vec->Size(); });
  if (num_elements > 0) {
    SpdbVector::Vec merged;
    merged.reserve(num_elements);

    for (iterator.SeekToFirst(); iterator.Valid(); iterator.Next()) {
      merged.emplace_back(iterator.key());
    }

    const size_t actual_elements_count = merged.size();
    SpdbVectorPtr new_vector(
        new SpdbVector(std::move(merged), actual_elements_count));

    // now replace
    WriteLock wl(&spdb_vectors_rwlock_);
    spdb_vectors_.insert(begin, std::move(new_vector));
    spdb_vectors_.erase(begin, end);
  }
}

bool SpdbVectorContainer::TryMergeVectors(
    std::list<SpdbVectorPtr>::iterator last) {
  std::list<SpdbVectorPtr>::iterator start = spdb_vectors_.begin();
  const size_t merge_threshold = switch_spdb_vector_limit_ * 75 / 100;

  size_t count = 0;
  for (auto s = start; s != last; ++s) {
    if ((*s)->Size() > merge_threshold) {
      if (count > 1) {
        last = s;
        break;
      }

      count = 0;
      start = std::next(s);
    } else {
      ++count;
      if (count == kMergedVectorsMax) {
        last = std::next(s);
        break;
      }
    }
  }
  if (count > 1) {
    Merge(start, last);
    return true;
  }
  return false;
}

void SpdbVectorContainer::SortThread() {
  std::unique_lock<std::mutex> lck(sort_thread_mutex_);
  std::list<SpdbVectorPtr>::iterator sort_iter_anchor = spdb_vectors_.begin();

  for (;;) {
    sort_thread_cv_.wait(lck);

    if (immutable_) {
      break;
    }

    std::list<SpdbVectorPtr>::iterator last;
    {
      ReadLock rl(&spdb_vectors_rwlock_);
      last = std::prev(spdb_vectors_.end());
    }

    if (last == sort_iter_anchor) {
      continue;
    }

    for (; sort_iter_anchor != last; ++sort_iter_anchor) {
      (*sort_iter_anchor)->Sort(comparator_);
    }

    if (spdb_vectors_.size() > kMergedVectorsMax) {
      if (TryMergeVectors(last)) {
        sort_iter_anchor = spdb_vectors_.begin();
      }
    }
  }
}

class HashSpdRep : public MemTableRep {
 public:
  HashSpdRep(const MemTableRep::KeyComparator& compare, Allocator* allocator,
             size_t bucket_size, size_t spdb_vector_limit_size);

  KeyHandle Allocate(const size_t len, char** buf) override;

  void Insert(KeyHandle handle) override { InsertKey(handle); }

  bool InsertKey(KeyHandle handle) override;

  bool InsertKeyWithHint(KeyHandle handle, void**) override {
    return InsertKey(handle);
  }

  bool InsertKeyWithHintConcurrently(KeyHandle handle, void**) override {
    return InsertKey(handle);
  }

  bool InsertKeyConcurrently(KeyHandle handle) override {
    return InsertKey(handle);
  }

  void MarkReadOnly() override;

  bool Contains(const char* key) const override;

  size_t ApproximateMemoryUsage() override;

  void Get(const LookupKey& k, void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) override;

  ~HashSpdRep() override;

  MemTableRep::Iterator* GetIterator(Arena* arena = nullptr) override;

 private:
  SpdbHashTable spdb_hash_table_;
  const MemTableRep::KeyComparator& compare_;
  std::shared_ptr<SpdbVectorContainer> spdb_vectors_cont_;
};

HashSpdRep::HashSpdRep(const MemTableRep::KeyComparator& compare,
                       Allocator* allocator, size_t bucket_size,
                       size_t add_list_limit_size)
    : MemTableRep(allocator),
      spdb_hash_table_(bucket_size, 1024),
      compare_(compare),
      spdb_vectors_cont_(
          new SpdbVectorContainer(compare, add_list_limit_size)) {}

HashSpdRep::~HashSpdRep() { MarkReadOnly(); }

KeyHandle HashSpdRep::Allocate(const size_t len, char** buf) {
  constexpr size_t kInlineDataSize =
      sizeof(SpdbKeyHandle) - offsetof(SpdbKeyHandle, key);
  const size_t alloc_size =
      std::max(len, kInlineDataSize) - kInlineDataSize + sizeof(SpdbKeyHandle);
  SpdbKeyHandle* h =
      reinterpret_cast<SpdbKeyHandle*>(allocator_->AllocateAligned(alloc_size));
  *buf = h->key;
  return h;
}

bool HashSpdRep::InsertKey(KeyHandle handle) {
  SpdbKeyHandle* spdb_handle = static_cast<SpdbKeyHandle*>(handle);
  if (!spdb_hash_table_.Add(spdb_handle, compare_)) {
    return false;
  }
  // insert to later sorter list
  spdb_vectors_cont_->Insert(spdb_handle->key);
  return true;
}

bool HashSpdRep::Contains(const char* key) const {
  return spdb_hash_table_.Contains(key, this->compare_);
}

void HashSpdRep::MarkReadOnly() { spdb_vectors_cont_->MarkReadOnly(); }

size_t HashSpdRep::ApproximateMemoryUsage() {
  // Memory is always allocated from the allocator.
  return 0;
}

void HashSpdRep::Get(const LookupKey& k, void* callback_args,
                     bool (*callback_func)(void* arg, const char* entry)) {
  spdb_hash_table_.Get(k, compare_, callback_args, callback_func);
}

MemTableRep::Iterator* HashSpdRep::GetIterator(Arena* arena) {
  const bool empty_iter = spdb_vectors_cont_->IsEmpty();

  if (arena != nullptr) {
    void* mem;
    if (empty_iter) {
      mem = arena->AllocateAligned(sizeof(SpdbVectorIteratorEmpty));
      return new (mem) SpdbVectorIteratorEmpty();
    } else {
      mem = arena->AllocateAligned(sizeof(SpdbVectorIterator));
      return new (mem) SpdbVectorIterator(spdb_vectors_cont_, compare_);
    }
  }
  if (empty_iter) {
    return new SpdbVectorIteratorEmpty();
  } else {
    return new SpdbVectorIterator(spdb_vectors_cont_, compare_);
  }
}

static std::unordered_map<std::string, OptionTypeInfo> hash_spd_factory_info = {
#ifndef ROCKSDB_LITE
    {"bucket_count",
     {0, OptionType::kSizeT, OptionVerificationType::kNormal,
      OptionTypeFlags::kDontSerialize /*Since it is part of the ID*/}},
#endif
};
}  // namespace

HashSpdRepFactory::HashSpdRepFactory(size_t bucket_count)
    : bucket_count_(bucket_count) {
  RegisterOptions("", &bucket_count_, &hash_spd_factory_info);
}

MemTableRep* HashSpdRepFactory::CreateMemTableRep(
    const MemTableRep::KeyComparator& compare, Allocator* allocator,
    const SliceTransform* /*transform*/, Logger* /*logger*/) {
  return new HashSpdRep(compare, allocator, bucket_count_, 10000);
}

}  // namespace ROCKSDB_NAMESPACE

#endif  // ROCKSDB_LITE
