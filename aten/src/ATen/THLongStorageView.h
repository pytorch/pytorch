#pragma once

#include <ATen/StorageImpl.h>
#include "TH/TH.h"
#include "TH/THStorageFunctions.hpp"
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
  operator StorageImpl*() {
    if (storage.pImpl()->size() == 0 && zero_dim_to_null) {
      return nullptr;
    }
    return storage.pImpl();
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
      : storage(nullptr), zero_dim_to_null(false) {
    // zero_dim_to_one converts an empty ArrayRef into [1]
    // zero_dim_to_null converts an empty ArrayRef into a null THLongStorage
    bool zero_dim_to_one = false;
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

    if (zero_dim_to_one && ref.size() == 0) {
      // make storage of size 0 actually a 1-length storage with 1 element
      // so that our 0-dim tensors get allocated as 1-dim inside TH

      one = 1;
      storage.set_pImpl(new StorageImpl(
          at::CTypeToScalarType<th::from_type<int64_t>>::to(),
          1,
          {&one, kCPU}, // non-owning
          nullptr,
          false));
    } else {
      storage.set_pImpl(new StorageImpl(
          at::CTypeToScalarType<th::from_type<int64_t>>::to(),
          ref.size(),
          {const_cast<void*>(static_cast<const void*>(ref.data())),
           kCPU}, // non-owning
          nullptr,
          false));
    }
  }
private:
  int64_t one;
  // NB: The lifetime of objects like one are tied to the lifetime of an
  // instance of this class. That means if storage is used after an instance of
  // this class dies, it'll be corrupted.
  Storage storage;
  bool zero_dim_to_null;
};

}
