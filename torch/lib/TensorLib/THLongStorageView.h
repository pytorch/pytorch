#pragma once

#include "TH/TH.h"

namespace tlib {

// make a fake storage out of a size, pointer pair...
class THLongStorageView {
public:
  static THLongStorageView make(ArrayRef<int64_t> ref) {
    return THLongStorageView(ref);
  }
  operator THLongStorage*() {
    return &storage;
  }
private:
  THLongStorageView(ArrayRef<int64_t> ref) {
    storage.data = (long*)(ref.data());
    storage.size = ref.size();
    storage.refcount = 0;
    storage.flag = 0;
    storage.allocator = nullptr;
    storage.allocatorContext = nullptr;
  }

  THLongStorage storage;
};

}
