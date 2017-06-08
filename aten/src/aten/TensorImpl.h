#pragma once

#include <atomic>

#include "TensorLib/Scalar.h"
#include "TensorLib/Type.h"

namespace tlib {

class Type;
struct TensorImpl {
  TensorImpl(Type * type)
  : type_(type), refcount(1) {}
  Type & type() const {
    return *type_;
  }

  virtual const char * toString() const = 0;
  void retain() {
    ++refcount;
  }
  virtual void release() {
    if(--refcount == 0) {
      delete this;
    }
  }
  virtual ~TensorImpl() {}

  friend class Type;
private:
  std::atomic<int> refcount;
  Type * type_;
};

}
