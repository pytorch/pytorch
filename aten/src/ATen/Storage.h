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
  virtual std::size_t elementSize() const = 0;
  virtual std::size_t size() const = 0;
  virtual void* data() = 0;
  virtual const void* data() const = 0;
  virtual Storage& retain() = 0;
  virtual Storage& free() = 0;
  virtual void * unsafeGetTH(bool retain) const = 0;

  virtual Storage& resize(int64_t new_size) = 0;

  virtual Type & type() const = 0;
  virtual int getDevice() const = 0;
  virtual const char * toString() const = 0;

  virtual Storage& fill(Scalar value) = 0;
  virtual Storage& set(std::size_t ind, Scalar value) = 0;
  virtual Storage& fast_set(std::size_t ind, Scalar value) = 0;
  virtual Scalar get(std::size_t ind) = 0;
  virtual Scalar fast_get(std::size_t ind) = 0;

  virtual void set_flag(char flag) = 0;
  virtual void clear_flag(char flag) = 0;
};

} // namespace at
