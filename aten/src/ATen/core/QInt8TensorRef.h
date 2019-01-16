#pragma once

namespace at {

class Tensor;
struct QInt8TensorRef {
  explicit QInt8TensorRef(const Tensor& t): tref(t) {}
  const Tensor& tref;
};

}
