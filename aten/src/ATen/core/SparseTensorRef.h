#pragma once

namespace at {

class Tensor;

using SparseTensorRef = Tensor;

// struct SparseTensorRef {
//   explicit SparseTensorRef(const Tensor& t): tref(t) {}
//   Tensor tref;
// };

}
