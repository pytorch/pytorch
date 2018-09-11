#pragma once

namespace at {

struct Tensor;
struct SparseTensorRef {
  explicit SparseTensorRef(const Tensor& t): tref(t) {}
  const Tensor& tref;
};

}
