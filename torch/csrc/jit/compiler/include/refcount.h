#pragma once

#include <stdio.h>
#include <atomic>

#include "torch/csrc/jit/compiler/include/logging.h"

namespace torch {
namespace jit {
namespace compiler {

// A refcounted object.
// Callers can call "Ref()" and "Unref" to increment and decrement its reference
// count.
// When the refrence count goes this zero, "this" object will be deleted through
// the local "delete". This assumes the object is created through "new" on the
// same heap.
class RefCounted {
 public:
  // Initial reference count is one.
  RefCounted() : ref_(1) {}

  // Increments reference count by one.
  void Ref() const {
    DCHECK_GE(ref_.load(), 1);
    ref_.fetch_add(1, std::memory_order_relaxed);
  }

  // Decrements reference count by one.
  void Unref() const {
    DCHECK_GT(ref_.load(), 0);
    // If ref_==1, this object is owned only by the caller. Bypass a locked op
    // in that case.
    if (RefCountIsOne() || ref_.fetch_sub(1) == 1) {
      DCHECK((ref_.store(0), true));
      // TODO: switch to a generic deleter. This assumes this object instance is
      // created through new.
      delete this;
    }
  }

  // Return whether the reference count is one.
  bool RefCountIsOne() const {
    return (ref_.load(std::memory_order_acquire) == 1);
  }

 protected:
  // Make destructor protected so that RefCounted objects cannot
  // be instantiated directly. Only subclasses can be instantiated.
  virtual ~RefCounted() {
    DCHECK_EQ(ref_.load(), 0);
  }

 private:
  mutable std::atomic_int_fast32_t ref_;

  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};

template <class NodeType>
class RefHandle {
 public:
  bool is_null() const {
    return node_ == nullptr;
  }

 protected:
  virtual ~RefHandle() {
    reset();
  }

  RefHandle() {}
  RefHandle(NodeType* node) : node_(node) {}

  RefHandle(const RefHandle& other) {
    this->reset();
    node_ = other.node_;
    node_->Ref();
  }

  RefHandle(RefHandle&& other) {
    node_ = other.node_;
    other.node_ = nullptr;
  }

  RefHandle& operator=(const RefHandle& other) {
    this->reset();
    node_ = other.node_;
    node_->Ref();
  }

  RefHandle& operator=(RefHandle&& other) {
    node_ = other.node_;
    other.node_ = nullptr;
  }

  void reset() {
    if (node_) {
      node_->Unref();
    }
    node_ = nullptr;
  }

  const NodeType* node() const {
    return node_;
  }
  NodeType* node() {
    return node_;
  }

 private:
  NodeType* node_ = nullptr;
};

} // namespace compiler
} // namespace jit
} // namespace torch
