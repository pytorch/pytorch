#pragma once

#include "TH/TH.h"
#include "TH/THStorage.hpp"

namespace at {

// Returns true if size represents an "no element" or "empty" tensor.
// See Note [empty versus 0-dim tensors]
static inline bool is_noelem_tensor_size(ArrayRef<int64_t> size) {
  return size.size() == 1 && size[0] == 0;
}

enum class THLongStorageViewKind {
  SIZE,
  // noelem_to_empty is to differentiate strides of empty tensors vs scalars.  In ATen, both may have strides [1],
  // but in TH an empty tensor should have stride [], while a scalar should have stride [1].
  STRIDE_EMPTY_TENSOR,  // noelem_to_empty = true
  STRIDE_SCALAR,  // noelem_to_empty = false
  LENGTH,
};

// make a fake storage out of a size, pointer pair...
// used as an argument where THSize and THStride are passed into TH
class THLongStorageView {
public:
  operator THLongStorage*() {
    if (storage.size == 0 && zero_dim_to_null) {
      return nullptr;
    }
    return &storage;
  }

  /*
  // This is done as an enum, and not as these static constructors, as there
  // is no move/copy constructor for THLongStorageView

  static THLongStorageView makeFromSize(ArrayRef<int64_t> ref) {
    return THLongStorageView(ref, true, false, false);
  }
  static THLongStorageView makeFromStride(ArrayRef<int64_t> ref, bool noelem_to_empty) {
    return THLongStorageView(ref, false, true, noelem_to_empty);
  }
  static THLongStorageView makeFromLength(ArrayRef<int64_t> ref) {
    return THLongStorageView(ref, false, false, false);
  }
  */

  THLongStorageView(ArrayRef<int64_t> ref, THLongStorageViewKind kind)
  : zero_dim_to_null(false)
  {
    // zero_dim_to_one converts an empty ArrayRef into [1]
    // zero_dim_to_null converts an empty ArrayRef into a null THLongStorage
    // noelem_to_empty makes an ArrayRef of [0] into an empty THLongStorage
    bool zero_dim_to_one = false;
    bool noelem_to_empty = false;
    switch (kind) {
      case THLongStorageViewKind::SIZE:
        zero_dim_to_one = true;
        break;
      case THLongStorageViewKind::STRIDE_EMPTY_TENSOR:
        zero_dim_to_null = true;
        noelem_to_empty = true;
        break;
      case THLongStorageViewKind::STRIDE_SCALAR:
        zero_dim_to_null = true;
        break;
      case THLongStorageViewKind::LENGTH:
        break;
    }
    if(zero_dim_to_one && ref.size() == 0) {
      // make storage of size 0 actually a 1-length storage with 1 element
      // so that our 0-dim tensors get allocated as 1-dim inside TH
      one = 1;
      storage.data = &one;
      storage.size = 1;
    } else if (noelem_to_empty && is_noelem_tensor_size(ref)) {
      storage.data = (int64_t*)(ref.data());
      storage.size = 0;
    }
    else {
      storage.data = (int64_t*)(ref.data());
      storage.size = ref.size();
    }
    storage.refcount = 0;
    storage.flag = 0;
    storage.allocator = nullptr;
    storage.allocatorContext = nullptr;
  }
private:
  int64_t one;
  THLongStorage storage;
  bool zero_dim_to_null;
};

}
