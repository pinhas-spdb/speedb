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
//

#include "rocksdb/table_pinning_policy.h"

#include <array>

#include "rocksdb/status.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/customizable_util.h"
#include "rocksdb/utilities/object_registry.h"
#include "table/block_based/default_pinning_policy.h"
#include "table/block_based/recording_pinning_policy.h"

namespace ROCKSDB_NAMESPACE {

namespace pinning {

namespace {

const std::array<std::string, kNumHierarchyCategories>
    kHierarchyCategoryToHyphenString{{"top-level", "partition", "other"}};

}  // Unnamed namespace

std::string GetHierarchyCategoryName(HierarchyCategory category) {
  return kHierarchyCategoryToHyphenString[static_cast<size_t>(category)];
}

} // namespace pinning 

TablePinningInfo::TablePinningInfo(int _level, bool _is_last_level_with_data,
                   Cache::ItemOwnerId _item_owner_id, size_t _file_size,
                   size_t _max_file_size_for_l0_meta_pin)
      : level(_level),
        is_last_level_with_data(_is_last_level_with_data),
        item_owner_id(_item_owner_id),
        file_size(_file_size),
        max_file_size_for_l0_meta_pin(_max_file_size_for_l0_meta_pin) {
  // Validate / Sanitize the level + is_last_level_with_data combination
  if (is_last_level_with_data) {
    if (level <= 0) {
      assert(level > 0);
      is_last_level_with_data = false;
    }
  }
}

std::string TablePinningInfo::ToString() const {
  std::string result;

  result.append("level=").append(std::to_string(level)).append(", ");
  result.append("is_last_level_with_data=")
      .append(std::to_string(is_last_level_with_data))
      .append(", ");
  result.append("item_owner_id=")
      .append(std::to_string(item_owner_id))
      .append(", ");
  result.append("file_size=").append(std::to_string(file_size)).append(", ");
  result.append("max_file_size_for_l0_meta_pin=").append(std::to_string(max_file_size_for_l0_meta_pin)).append("\n");

  return result;
}

PinnedEntry::PinnedEntry(int _level, bool _is_last_level_with_data,
                         pinning::HierarchyCategory _category,
                         Cache::ItemOwnerId _item_owner_id,
                         CacheEntryRole _role, size_t _size)
    : level(_level),
      is_last_level_with_data(_is_last_level_with_data),
      category(_category),
      item_owner_id(_item_owner_id),
      role(_role),
      size(_size) {}

std::string PinnedEntry::ToString() const {
  std::string result;

  result.append("level=").append(std::to_string(level)).append(", ");
  result.append("is_last_level_with_data=")
      .append(std::to_string(is_last_level_with_data))
      .append(", ");
  result.append("category=")
      .append(pinning::GetHierarchyCategoryName(category))
      .append(", ");
  result.append("item_owner_id=")
      .append(std::to_string(item_owner_id))
      .append(", ");
  result.append("role=").append(GetCacheEntryRoleName(role)).append(", ");
  result.append("size=").append(std::to_string(size)).append("\n");

  return result;
}

DefaultPinningPolicy::DefaultPinningPolicy() {
  //**TODO: Register options?
}

DefaultPinningPolicy::DefaultPinningPolicy(const BlockBasedTableOptions& bbto)
    : DefaultPinningPolicy(bbto.metadata_cache_options,
                           bbto.pin_top_level_index_and_filter,
                           bbto.pin_l0_filter_and_index_blocks_in_cache) {}

DefaultPinningPolicy::DefaultPinningPolicy(const MetadataCacheOptions& mdco,
                                           bool pin_top, bool pin_l0)
    : cache_options_(mdco),
      pin_top_level_index_and_filter_(pin_top),
      pin_l0_index_and_filter_(pin_l0) {
  //**TODO: Register options?
}

bool DefaultPinningPolicy::CheckPin(const TablePinningInfo& tpi,
                                    pinning::HierarchyCategory category,
                                    CacheEntryRole /*role*/, size_t /*size*/,
                                    size_t /*limit*/) const {
  if (tpi.level < 0) {
    return false;
  } else if (category == pinning::HierarchyCategory::TOP_LEVEL) {
    return IsPinned(tpi, cache_options_.top_level_index_pinning,
                    pin_top_level_index_and_filter_ ? PinningTier::kAll
                                                    : PinningTier::kNone);
  } else if (category == pinning::HierarchyCategory::PARTITION) {
    return IsPinned(tpi, cache_options_.partition_pinning,
                    pin_l0_index_and_filter_ ? PinningTier::kFlushedAndSimilar
                                             : PinningTier::kNone);
  } else {
    return IsPinned(tpi, cache_options_.unpartitioned_pinning,
                    pin_l0_index_and_filter_ ? PinningTier::kFlushedAndSimilar
                                             : PinningTier::kNone);
  }
}

bool DefaultPinningPolicy::IsPinned(const TablePinningInfo& tpi,
                                    PinningTier pinning_tier,
                                    PinningTier fallback_pinning_tier) const {
  // Fallback to fallback would lead to infinite recursion. Disallow it.
  assert(fallback_pinning_tier != PinningTier::kFallback);

  switch (pinning_tier) {
    case PinningTier::kFallback:
      return IsPinned(tpi, fallback_pinning_tier,
                      PinningTier::kNone /* fallback_pinning_tier */);
    case PinningTier::kNone:
      return false;
    case PinningTier::kFlushedAndSimilar: {
      bool answer = (tpi.level == 0 &&
                     tpi.file_size <= tpi.max_file_size_for_l0_meta_pin);
      return answer;
    }
    case PinningTier::kAll:
      return true;
    default:
      assert(false);
      return false;
  }
}

TablePinningPolicy* NewDefaultPinningPolicy(
    const BlockBasedTableOptions& bbto) {
  return new DefaultPinningPolicy(bbto);
}

static int RegisterBuiltinPinningPolicies(ObjectLibrary& library,
                                          const std::string& /*arg*/) {
  library.AddFactory<TablePinningPolicy>(
      DefaultPinningPolicy::kClassName(),
      [](const std::string& /*uri*/, std::unique_ptr<TablePinningPolicy>* guard,
         std::string* /* errmsg */) {
        guard->reset(new DefaultPinningPolicy(BlockBasedTableOptions()));
        return guard->get();
      });
  size_t num_types;
  return static_cast<int>(library.GetFactoryCount(&num_types));
}

Status TablePinningPolicy::CreateFromString(
    const ConfigOptions& options, const std::string& value,
    std::shared_ptr<TablePinningPolicy>* policy) {
  static std::once_flag loaded;
  std::call_once(loaded, [&]() {
    RegisterBuiltinPinningPolicies(*(ObjectLibrary::Default().get()), "");
  });
  return LoadManagedObject<TablePinningPolicy>(options, value, policy);
}

}  // namespace ROCKSDB_NAMESPACE
