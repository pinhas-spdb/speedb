//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//

#pragma once

#include <algorithm>
#include <iostream>

#include "rocksdb/memory_allocator.h"
#ifdef ROCKSDB_MALLOC_USABLE_SIZE
#include <malloc.h>
#endif
namespace ROCKSDB_NAMESPACE {
#ifdef MEMORY_REPORTING
struct blockfetchermem {
  inline static std::atomic_int64_t mem = {0};
};
#endif
struct CustomDeleter {
  CustomDeleter(MemoryAllocator* a = nullptr) : allocator(a) {}

  void operator()(char* ptr) const {
#ifdef MEMORY_REPORTING
    ROCKSDB_NAMESPACE::blockfetchermem::mem.fetch_sub(
        malloc_usable_size(const_cast<void*>(static_cast<const void*>(ptr))));
#endif
    if (allocator) {
      allocator->Deallocate(reinterpret_cast<void*>(ptr));
    } else {
      delete[] ptr;
    }
  }

  MemoryAllocator* allocator;
  size_t bytes_ = 0;
};

using CacheAllocationPtr = std::unique_ptr<char[], CustomDeleter>;

inline CacheAllocationPtr AllocateBlock(size_t size,
                                        MemoryAllocator* allocator) {
  if (allocator) {
#ifdef MEMORY_REPORTING
    ROCKSDB_NAMESPACE::blockfetchermem::mem.fetch_add(size);
#endif
    auto block = reinterpret_cast<char*>(allocator->Allocate(size));
    return CacheAllocationPtr(block, {allocator});
  }
  char* cache = new char[size];
#ifdef MEMORY_REPORTING
  ROCKSDB_NAMESPACE::blockfetchermem::mem.fetch_add(
      malloc_usable_size(const_cast<void*>(static_cast<const void*>(cache))));
#endif
  return CacheAllocationPtr(cache);
}

inline CacheAllocationPtr AllocateAndCopyBlock(const Slice& data,
                                               MemoryAllocator* allocator) {
  CacheAllocationPtr cap = AllocateBlock(data.size(), allocator);
  std::copy_n(data.data(), data.size(), cap.get());
  return cap;
}

}  // namespace ROCKSDB_NAMESPACE
