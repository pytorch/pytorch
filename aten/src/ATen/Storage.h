#pragma once

#include "ATen/Scalar.h"
#include <TH/THStorageFunctions.hpp>

namespace at {

struct Type;

struct Storage {
  static const char RESIZABLE = 2;

  Storage() {}
  Storage(THStorage* storage)
      : storage(storage) {}
  Storage(const Storage& other) = delete;
  virtual ~Storage() {
    THStorage_free(storage);
  }
  void operator=(const Storage&) = delete;

  virtual size_t elementSize() const = 0;
  size_t size() const {
    return storage->size;
  };
  void* data() {
    return storage->data_ptr.get();
  };
  const void* data() const {
    return storage->data_ptr.get();
  };
  void* unsafeGetTH(bool retain_) const {
    if (retain_) {
      THStorage_retain(storage);
    }
    return storage;
  }
  void retain() {
    THStorage_retain(storage);
  }
  virtual Type & type() const = 0;
  int getDevice() const {
    return storage->data_ptr.device().index();
  }
  void clear_flag(char flag) {
    THStorage_clearFlag(storage, flag);
  }

 protected:
  THStorage *storage;
};

} // namespace at
