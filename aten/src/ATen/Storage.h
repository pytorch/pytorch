#pragma once

#include "ATen/Scalar.h"

namespace at {

struct Type;

struct Storage {
  static const char RESIZABLE = 2;

  Storage() {}
  Storage(const Storage& other) = delete;
  void operator=(const Storage&) = delete;

  virtual ~Storage() {};
  virtual size_t elementSize() const = 0;
  virtual size_t size() const = 0;
  virtual void* data() = 0;
  virtual const void* data() const = 0;
  virtual void * unsafeGetTH(bool retain) const = 0;

  virtual Type & type() const = 0;
  virtual int getDevice() const = 0;
  virtual void clear_flag(char flag) = 0;
};

} // namespace at
