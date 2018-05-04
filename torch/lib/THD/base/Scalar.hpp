#pragma once

#include <cstddef>

#include "RPCType.hpp"

namespace thd {

struct Scalar {
  Scalar() {}
  Scalar(const Scalar& other) = delete;
  Scalar(Scalar&& other) = delete;
  virtual ~Scalar() {}

  virtual std::size_t elementSize() const = 0;
  virtual void* data() = 0;
  virtual const void* data() const = 0;
  virtual RPCType type() const = 0;
  virtual Scalar* clone() const = 0;
};

template<typename real>
struct ScalarWrapper : Scalar {
  ScalarWrapper() {}
  ScalarWrapper(real value) : _value(value) {}
  virtual ~ScalarWrapper() {}

  virtual std::size_t elementSize() const override {
    return sizeof(real);
  }

  virtual void* data() override {
    return &_value;
  }

  virtual const void* data() const override {
    return &_value;
  }

  virtual RPCType type() const override {
    return type_traits<real>::type;
  }

  virtual ScalarWrapper* clone() const override {
    return new ScalarWrapper(value());
  }

  real value() const {
    return _value;
  }

private:
  real _value;
};

using FloatScalar = ScalarWrapper<double>;
using IntScalar = ScalarWrapper<int64_t>;

} // namespace thd
