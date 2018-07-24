#pragma once

#include <atomic>

namespace at {

// base class for refcounted things, allows for collects of generic
// refcounted objects that include tensors
struct Retainable {
  Retainable(): refcount(1), weak_refcount(1) {}
  void retain() {
    ++refcount;
  }
  void release() {
    if(--refcount == 0) {
      // If we know that this is the last reference then we can skip
      // all the decrements and release_resources().
      if (weak_refcount == 1) {
        delete this;
      } else {
        release_resources();
        weak_release();
      }
    }
  }
  void weak_retain() {
    ++weak_refcount;
  }
  void weak_release() {
    if (--weak_refcount == 0) {
      delete this;
    }
  }
  bool weak_lock() {
    for (;;) {
      auto current_refcount = refcount.load();
      if (current_refcount == 0) return false;
      if (refcount.compare_exchange_strong(current_refcount, current_refcount + 1)) break;
    }
    return true;
  }
  uint32_t use_count() const {
    return refcount.load();
  }
  uint32_t weak_use_count() const {
    return weak_refcount.load();
  }

  virtual void release_resources() {};
  virtual ~Retainable() {}
private:
  // INVARIANT: once refcount reaches 0 it can never go up
  // INVARIANT: weak_refcount = number of weak references + (refcount > 0 ? 1 : 0)
  std::atomic<uint32_t> refcount;
  std::atomic<uint32_t> weak_refcount;
};

}
