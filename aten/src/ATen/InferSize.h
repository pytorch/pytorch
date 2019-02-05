#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <sstream>
#include <vector>

namespace at {

// Infers the size of a dim with size -1, if it exists. Also checks that new
// shape is compatible with the number of elements.
static std::vector<int64_t> infer_size(IntArrayRef shape, int64_t numel) {
  auto res = shape.vec();
  int64_t newsize = 1;
  auto infer_dim = c10::optional<int64_t>();
  for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
    if (shape[dim] == -1) {
      if (infer_dim) {
        throw std::runtime_error("only one dimension can be inferred");
      }
      infer_dim = dim;
    } else if (shape[dim] >= 0) {
      newsize *= shape[dim];
    } else {
      AT_ERROR("invalid shape dimension ", shape[dim]);
    }
  }

  if (numel == newsize || (infer_dim && newsize > 0 && numel % newsize == 0)) {
    if (infer_dim) {
      // we have a degree of freedom here to select the dimension size; follow NumPy semantics
      // and just bail.
      AT_CHECK(newsize != 0, "cannot reshape tensor of 0 elements into shape ", shape);
      res[*infer_dim] = numel / newsize;
    }
    return res;
  }

  std::ostringstream ss;
  ss << "shape '" << shape << "' is invalid for input of size " << numel;
  throw std::runtime_error(ss.str());
}

}
