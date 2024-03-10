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

#include "plugin/speedb/pinning_policy/scoped_pinning_policy.h"

#include <inttypes.h>

#include <cstdio>
#include <string>
#include <unordered_map>

#include "port/port.h"
#include "rocksdb/utilities/options_type.h"

namespace ROCKSDB_NAMESPACE {

namespace {
std::unordered_map<std::string, OptionTypeInfo> scoped_pinning_type_info = {
    {"capacity",
     {offsetof(struct ScopedPinningOptions, capacity), OptionType::kSizeT,
      OptionVerificationType::kNormal, OptionTypeFlags::kNone}},
    {"last_level_with_data_percent",
     {offsetof(struct ScopedPinningOptions, last_level_with_data_percent),
      OptionType::kUInt32T, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
    {"mid_percent",
     {offsetof(struct ScopedPinningOptions, mid_percent), OptionType::kUInt32T,
      OptionVerificationType::kNormal, OptionTypeFlags::kNone}},
};

}  // unnamed namespace

ScopedPinningPolicy::ScopedPinningPolicy()
    : ScopedPinningPolicy(ScopedPinningOptions()) {}

ScopedPinningPolicy::ScopedPinningPolicy(const ScopedPinningOptions& options)
    : options_(options) {
  RegisterOptions(&options_, &scoped_pinning_type_info);
}

std::string ScopedPinningPolicy::GetId() const {
  return GenerateIndividualId();
}

bool ScopedPinningPolicy::CheckPin(const TablePinningInfo& tpi,
                                   pinning::HierarchyCategory /* category */,
                                   CacheEntryRole /* role */, size_t size,
                                   size_t usage) const {
  auto proposed = usage + size;

  if (tpi.is_last_level_with_data &&
      options_.last_level_with_data_percent > 0) {
    if (proposed >
        (options_.capacity * options_.last_level_with_data_percent / 100)) {
      return false;
    }
  } else if (tpi.level > 0 && options_.mid_percent > 0) {
    if (proposed > (options_.capacity * options_.mid_percent / 100)) {
      return false;
    }
  } else if (proposed > options_.capacity) {
    return false;
  }

  return true;
}

std::string ScopedPinningPolicy::GetPrintableOptions() const {
  std::string ret;

  ret.append("    capacity: ").append(std::to_string(options_.capacity));
  ret.append("    last_level_with_data_percent: ")
      .append(std::to_string(options_.last_level_with_data_percent));
  ret.append("    mid_percent: ").append(std::to_string(options_.mid_percent));

  return ret;
}

}  // namespace ROCKSDB_NAMESPACE
