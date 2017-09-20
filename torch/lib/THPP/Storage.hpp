#pragma once

#include "Type.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace thpp {

struct Tensor;

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

  virtual Storage& resize(int64_t new_size) = 0;

  virtual thpp::Type type() const = 0;
  virtual bool isCuda() const = 0;
  virtual int getDevice() const = 0;

  virtual std::unique_ptr<Tensor> newTensor() const = 0;
};

template<typename real>
struct StorageScalarInterface : public Storage {
  using scalar_type = real;
  virtual StorageScalarInterface& fill(scalar_type value) = 0;
  virtual StorageScalarInterface& set(std::size_t ind, scalar_type value) = 0;
  virtual StorageScalarInterface& fast_set(std::size_t ind, scalar_type value) = 0;
  virtual scalar_type get(std::size_t ind) = 0;
  virtual scalar_type fast_get(std::size_t ind) = 0;
};

using FloatStorage = StorageScalarInterface<double>;
using IntStorage = StorageScalarInterface<int64_t>;

} // namespace thpp

#include "Tensor.hpp"
