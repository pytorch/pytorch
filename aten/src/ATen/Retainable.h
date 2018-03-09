#pragma once

#include <atomic>

namespace at {

// base class for refcounted things, allows for collects of generic
// refcounted objects that include tensors
struct Retainable {
  Retainable(): refcount(1) {}
  void retain() {
    ++refcount;
  }
  void release() {
    if(--refcount == 0) {
      delete this;
    }
  }
  int use_count() const {
    return refcount.load();
  }
  virtual ~Retainable() {}
private:
  std::atomic<int> refcount;
};

}
