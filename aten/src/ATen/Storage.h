#pragma once

#include "ATen/Scalar.h"
#include <TH/THStorageFunctions.hpp>

namespace at {

struct Type;

struct Storage {
  Storage() {}
  Storage(THStorage* storage)
      : storage(storage) {}
  Storage(Storage&) = delete;
  Storage(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage(const Storage&&) = delete;
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
  void set_resizable(bool resizable) {
    THStorage_setResizable(storage, resizable);
  }

 protected:
  THStorage *storage;
};

} // namespace at
