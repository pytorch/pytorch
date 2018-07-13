#pragma once

#include "TH/TH.h"
#include "TH/THStorage.hpp"
#include "TH/THTypeConversion.hpp"

namespace at {

enum class THLongStorageViewKind {
  SIZE,
  STRIDE,
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
  // This is done as an enum, and not as static constructors, as there
  // is no move/copy constructor for THLongStorageView

  static THLongStorageView makeFromSize(ArrayRef<int64_t> ref) {
    ...
  }

  static THLongStorageView makeFromLength(ArrayRef<int64_t> ref) {
    ...
  }
  */

  THLongStorageView(ArrayRef<int64_t> ref, THLongStorageViewKind kind)
  : zero_dim_to_null(false)
  {
    // zero_dim_to_one converts an empty ArrayRef into [1]
    // zero_dim_to_null converts an empty ArrayRef into a null THLongStorage
    bool zero_dim_to_one = false;
    bool noelem_to_empty = false;
    switch (kind) {
      case THLongStorageViewKind::SIZE:
        zero_dim_to_one = true;
        break;
      case THLongStorageViewKind::STRIDE:
        zero_dim_to_null = true;
        break;
      case THLongStorageViewKind::LENGTH:
        break;
    }

    if(zero_dim_to_one && ref.size() == 0) {
      // make storage of size 0 actually a 1-length storage with 1 element
      // so that our 0-dim tensors get allocated as 1-dim inside TH
      one = 1;
      storage.data_ptr = {&one, kCPU}; // non-owning
      storage.size = 1;
    } else {
      storage.data_ptr = {const_cast<void*>(static_cast<const void*>(ref.data())), kCPU}; // non-owning
      storage.size = ref.size();
    }
    storage.scalar_type = at::CTypeToScalarType<th::from_type<int64_t>>::to();
    storage.refcount = 0;
    storage.flag = 0;
  }
private:
  int64_t one;
  THLongStorage storage;
  bool zero_dim_to_null;
};

}
