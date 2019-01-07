#pragma once

namespace at {

class Tensor;
struct SparseTensorRef {
  explicit SparseTensorRef(const Tensor& t): tref(t) {}
  const Tensor& tref;
};

}
