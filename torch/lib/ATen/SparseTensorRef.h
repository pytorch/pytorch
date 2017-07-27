#pragma once

namespace at {

struct Tensor;
struct SparseTensor {
  explicit SparseTensor(const Tensor& t): tref(t) {}
  const Tensor& tref;
};

}
