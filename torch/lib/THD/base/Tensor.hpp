#pragma once

#include "../master_worker/master/THDTensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace thd {

/*
 * The following notation comes from:
 * docs.python.org/3.5/library/struct.html#module-struct
 * except from 'T', which stands for Tensor
 */

enum class TensorType : char {
  CHAR = 'c',
  FLOAT = 'f',
  DOUBLE = 'd',
  SHORT = 'h',
  USHORT = 'H',
  INT = 'i',
  UINT = 'I',
  LONG = 'l',
  ULONG = 'L',
  LONG_LONG = 'q',
  ULONG_LONG = 'Q',
  TENSOR = 'T',
};


template<typename T>
struct tensor_type_traits {};

template<>
struct tensor_type_traits<char> {
  static constexpr TensorType type = TensorType::CHAR;
};

template<>
struct tensor_type_traits<float> {
  static constexpr TensorType type = TensorType::FLOAT;
};

template<>
struct tensor_type_traits<double> {
  static constexpr TensorType type = TensorType::DOUBLE;
};

template<>
struct tensor_type_traits<short> {
  static constexpr TensorType type = TensorType::SHORT;
};

template<>
struct tensor_type_traits<unsigned short> {
  static constexpr TensorType type = TensorType::USHORT;
};

template<>
struct tensor_type_traits<uint8_t> {
  static constexpr TensorType type = TensorType::USHORT;
};

template<>
struct tensor_type_traits<int> {
  static constexpr TensorType type = TensorType::INT;
};

template<>
struct tensor_type_traits<unsigned int> {
  static constexpr TensorType type = TensorType::UINT;
};

template<>
struct tensor_type_traits<long> {
  static constexpr TensorType type = TensorType::LONG;
};

template<>
struct tensor_type_traits<unsigned long> {
  static constexpr TensorType type = TensorType::ULONG;
};

template<>
struct tensor_type_traits<long long> {
  static constexpr TensorType type = TensorType::LONG_LONG;
};

template<>
struct tensor_type_traits<unsigned long long> {
  static constexpr TensorType type = TensorType::ULONG_LONG;
};

template<>
struct tensor_type_traits<THDTensor> {
  static constexpr TensorType type = TensorType::TENSOR;
};


static const std::unordered_map<char, TensorType> format_to_type = {
  {'c', TensorType::CHAR},
  {'f', TensorType::FLOAT},
  {'d', TensorType::DOUBLE},
  {'h', TensorType::SHORT},
  {'H', TensorType::USHORT},
  {'i', TensorType::INT},
  {'I', TensorType::UINT},
  {'l', TensorType::LONG},
  {'L', TensorType::ULONG},
  {'q', TensorType::LONG_LONG},
  {'Q', TensorType::ULONG_LONG},
  {'T', TensorType::TENSOR},
};

} // namespace thd

struct Tensor {
  using long_range = std::vector<long>;

  Tensor() {};
  Tensor(const Tensor& other) = delete;
  Tensor(Tensor&& other) = delete;
  virtual ~Tensor() {};

  virtual Tensor* clone() const = 0;

  virtual int nDim() const = 0;
  virtual long_range sizes() const = 0;
  virtual long_range strides() const = 0;
  virtual const long* rawSizes() const = 0;
  virtual const long* rawStrides() const = 0;
  virtual size_t storageOffset() const = 0;
  virtual size_t elementSize() const = 0;
  virtual long long numel() const = 0;
  virtual bool isContiguous() const = 0;
  virtual void* data() = 0;
  virtual const void* data() const = 0;

  virtual Tensor& resize(const std::initializer_list<long>& new_size) = 0;
  virtual Tensor& resize(const std::vector<long>& new_size) = 0;

  virtual thd::TensorType type() const = 0;
};

template<typename real>
struct TensorScalarInterface : public Tensor {
  using scalar_type = real;
  virtual TensorScalarInterface& fill(scalar_type value) = 0;
  virtual TensorScalarInterface& add(const Tensor& source, scalar_type salar) = 0;
};

using FloatTensor = TensorScalarInterface<double>;
using IntTensor = TensorScalarInterface<long long>;

template<typename real>
struct tensor_traits {
  using scalar_type = typename std::conditional<
    std::is_floating_point<real>::value,
    double,
    long long>::type;
  using interface_type = TensorScalarInterface<scalar_type>;
};
