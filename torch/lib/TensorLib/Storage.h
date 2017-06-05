#pragma once

#include "TensorLib/Scalar.h"
#include "TensorLib/Type.h"

namespace tlib {

struct Storage {
  Storage() {};
  Storage(const Storage& other) = delete;
  Storage(Storage&& other) = delete;
  virtual ~Storage() {};

  virtual std::size_t elementSize() const = 0;
  virtual std::size_t size() const = 0;
  virtual void* data() = 0;
  virtual const void* data() const = 0;
  virtual Storage& retain() = 0;
  virtual Storage& free() = 0;

  virtual Storage& resize(long new_size) = 0;

  virtual Type & type() const = 0;
  virtual int getDevice() const = 0;

  virtual Storage& fill(Scalar value) = 0;
  virtual Storage& set(std::size_t ind, Scalar value) = 0;
  virtual Storage& fast_set(std::size_t ind, Scalar value) = 0;
  virtual Scalar get(std::size_t ind) = 0;
  virtual Scalar fast_get(std::size_t ind) = 0;

};

} // namespace tlib
