#pragma once

#include "TH/TH.h"
#include "TH/THStorage.hpp"
#include "TH/THTypeConversion.hpp"

namespace at {

// Returns true if size represents an "no element" or "empty" tensor.
// See Note [empty versus 0-dim tensors]
static inline bool is_noelem_tensor_size(ArrayRef<int64_t> size) {
  return size.size() == 1 && size[0] == 0;
}

enum class THLongStorageViewKind {
  SIZE_STRIDE,  // represents a size or stride.
  LENGTH,
};

// make a fake storage out of a size, pointer pair...
// used as an argument where THSize and THStride are passed into TH
class THLongStorageView {
public:
  operator THLongStorage*() {
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
  {
    // zero_dim_to_one converts an empty ArrayRef into [1]
    bool zero_dim_to_one = kind == THLongStorageViewKind::SIZE_STRIDE;

    if(zero_dim_to_one && ref.size() == 0) {
      // make storage of size 0 actually a 1-length storage with 1 element
      // so that our 0-dim tensors get allocated as 1-dim inside TH
      one = 1;
      storage.data_ptr = &one;
      storage.size = 1;
    } else {
      storage.data_ptr = (void*)(ref.data());
      storage.size = ref.size();
    }
    storage.scalar_type = at::CTypeToScalarType<th::from_type<int64_t>>::to();
    storage.refcount = 0;
    storage.flag = 0;
    storage.allocator = nullptr;
    storage.allocatorContext = nullptr;
  }
private:
  int64_t one;
  THLongStorage storage;
};

}
