#pragma once

#include "TH/TH.h"

namespace at {

// make a fake storage out of a size, pointer pair...
// used as an argument where THSize and THStride are passed into TH
class THLongStorageView {
public:
  static THLongStorageView make(ArrayRef<int64_t> ref, bool zero_dim_to_one = false) {
    return THLongStorageView(ref,zero_dim_to_one);
  }
  operator THLongStorage*() {
    return &storage;
  }
private:
  THLongStorageView(ArrayRef<int64_t> ref, bool zero_dim_to_one) {
    if(zero_dim_to_one && ref.size() == 0) {
      // make storage of size 0 actually a 1-length storage with 1 element
      // so that our 0-dim tensors get allocated as 1-dim inside TH
      one = 1;
      storage.data = &one;
      storage.size = 1;
    } else {
      storage.data = (long*)(ref.data());
      storage.size = ref.size();
    }
    storage.refcount = 0;
    storage.flag = 0;
    storage.allocator = nullptr;
    storage.allocatorContext = nullptr;
  }
  long one;
  THLongStorage storage;
};

}
