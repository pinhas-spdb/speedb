//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include <set>

#include "table/internal_iterator.h"
#include "test_util/sync_point.h"

namespace ROCKSDB_NAMESPACE {

// A internal wrapper class with an interface similar to Iterator that caches
// the valid() and key() results for an underlying iterator.
// This can help avoid virtual function calls and also gives better
// cache locality.
template <class TValue = Slice>
class IteratorWrapperBase {
 public:
  IteratorWrapperBase() : iter_(nullptr), valid_(false) {}
  explicit IteratorWrapperBase(InternalIteratorBase<TValue>* _iter)
      : iter_(nullptr) {
    Set(_iter);
  }
  ~IteratorWrapperBase() {}
  InternalIteratorBase<TValue>* iter() const { return iter_; }

  // Set the underlying Iterator to _iter and return
  // previous underlying Iterator.
  InternalIteratorBase<TValue>* Set(InternalIteratorBase<TValue>* _iter) {
    InternalIteratorBase<TValue>* old_iter = iter_;

    iter_ = _iter;
    if (iter_ == nullptr) {
      valid_ = false;
    } else {
      Update();
    }
    return old_iter;
  }

  void DeleteIter(bool is_arena_mode) {
    if (iter_) {
      if (!is_arena_mode) {
        delete iter_;
      } else {
        iter_->~InternalIteratorBase<TValue>();
      }
    }
  }

  // Iterator interface methods
  bool Valid() const { return valid_; }
  Slice key() const {
    assert(Valid());
    return result_.key;
  }
  TValue value() const {
    assert(Valid());
    return iter_->value();
  }
  // Methods below require iter() != nullptr
  Status status() const {
    assert(iter_);
    return iter_->status();
  }
  bool PrepareValue() {
    assert(Valid());
    if (result_.value_prepared) {
      return true;
    }
    if (iter_->PrepareValue()) {
      result_.value_prepared = true;
      return true;
    }

    assert(!iter_->Valid());
    valid_ = false;
    return false;
  }
  void Next() {
    assert(iter_);
    valid_ = iter_->NextAndGetResult(&result_);
    assert(!valid_ || iter_->status().ok());
  }
  bool NextAndGetResult(IterateResult* result) {
    assert(iter_);
    valid_ = iter_->NextAndGetResult(&result_);
    *result = result_;
    assert(!valid_ || iter_->status().ok());
    return valid_;
  }
  void Prev() {
    assert(iter_);
    iter_->Prev();
    Update();
  }
  void Seek(const Slice& k) {
    assert(iter_);
    iter_->Seek(k);
    Update();
  }
  void SeekForPrev(const Slice& k) {
    assert(iter_);
    iter_->SeekForPrev(k);
    Update();
  }
  void SeekToFirst() {
    assert(iter_);
    iter_->SeekToFirst();
    Update();
  }
  void SeekToLast() {
    assert(iter_);
    iter_->SeekToLast();
    Update();
  }

  bool MayBeOutOfLowerBound() {
    assert(Valid());
    return iter_->MayBeOutOfLowerBound();
  }

  IterBoundCheck UpperBoundCheckResult() {
    assert(Valid());
    return result_.bound_check_result;
  }

  void SetPinnedItersMgr(PinnedIteratorsManager* pinned_iters_mgr) {
    assert(iter_);
    iter_->SetPinnedItersMgr(pinned_iters_mgr);
  }
  bool IsKeyPinned() const {
    assert(Valid());
    return iter_->IsKeyPinned();
  }
  bool IsValuePinned() const {
    assert(Valid());
    return iter_->IsValuePinned();
  }

  bool IsValuePrepared() const {
    return result_.value_prepared;
  }

  Slice user_key() const {
    assert(Valid());
    return iter_->user_key();
  }

  void UpdateReadaheadState(InternalIteratorBase<TValue>* old_iter) {
    if (old_iter && iter_) {
      ReadaheadFileInfo readahead_file_info;
      old_iter->GetReadaheadState(&readahead_file_info);
      iter_->SetReadaheadState(&readahead_file_info);
    }
  }

  IteratorTargetState ValidateRange(const Slice& target,
                                    const Comparator* cmp) {
    return iter_->ValidateRange(target, cmp);
  }

  void SetRange(const Slice& smallest_key, Slice& largest_key) {
    iter_->SetRange(smallest_key, largest_key);
  }

  Slice GetSmallestKeyRange() const { return iter_->GetSmallestKeyRange(); }

 private:
  void Update() {
    valid_ = iter_->Valid();
    if (valid_) {
      assert(iter_->status().ok());
      result_.key = iter_->key();
      result_.bound_check_result = IterBoundCheck::kUnknown;
      result_.value_prepared = false;
    }
  }

  InternalIteratorBase<TValue>* iter_;
  IterateResult result_;
  bool valid_;
};

using IteratorWrapper = IteratorWrapperBase<Slice>;

class Arena;
// Return an empty iterator (yields nothing) allocated from arena.
template <class TValue = Slice>
extern InternalIteratorBase<TValue>* NewEmptyInternalIterator(Arena* arena);

}  // namespace ROCKSDB_NAMESPACE
