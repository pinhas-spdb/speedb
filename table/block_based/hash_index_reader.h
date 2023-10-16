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
#pragma once

#include "table/block_based/index_reader_common.h"

namespace ROCKSDB_NAMESPACE {
// Index that leverages an internal hash table to quicken the lookup for a given
// key.
class HashIndexReader : public BlockBasedTable::IndexReaderCommon {
 public:
  static Status Create(const BlockBasedTable* table, const ReadOptions& ro,
                       const TablePinningOptions& tpo,
                       FilePrefetchBuffer* prefetch_buffer,
                       InternalIterator* meta_index_iter, bool use_cache,
                       bool prefetch, bool pin,
                       BlockCacheLookupContext* lookup_context,
                       std::unique_ptr<IndexReader>* index_reader);

  InternalIteratorBase<IndexValue>* NewIterator(
      const ReadOptions& read_options, bool disable_prefix_seek,
      IndexBlockIter* iter, GetContext* get_context,
      BlockCacheLookupContext* lookup_context) override;

  size_t ApproximateMemoryUsage() const override {
    size_t usage = ApproximateIndexBlockMemoryUsage();
#ifdef ROCKSDB_MALLOC_USABLE_SIZE
    usage += malloc_usable_size(const_cast<HashIndexReader*>(this));
#else
    if (prefix_index_) {
      usage += prefix_index_->ApproximateMemoryUsage();
    }
    usage += sizeof(*this);
#endif  // ROCKSDB_MALLOC_USABLE_SIZE
    return usage;
  }

 private:
  HashIndexReader(const BlockBasedTable* t, CachableEntry<Block>&& index_block,
                  std::unique_ptr<PinnedEntry>&& pinned)
      : IndexReaderCommon(t, std::move(index_block), std::move(pinned)) {}

  std::unique_ptr<BlockPrefixIndex> prefix_index_;
};
}  // namespace ROCKSDB_NAMESPACE
