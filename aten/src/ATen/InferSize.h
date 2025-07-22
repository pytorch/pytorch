#pragma once

#include <ATen/DimVector.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/DimVector.h>
#include <optional>
#include <sstream>
#include <vector>

namespace at {

// Infers the size of a dim with size -1, if it exists. Also checks that new
// shape is compatible with the number of elements.
//
// templated to handle std::vector<int64_t> and DimVector use cases, see
// below
//
template <typename InputArrayRef, typename NumelType, typename ResultVec>
inline void infer_size_impl(
    InputArrayRef shape,
    NumelType numel,
    ResultVec& res) {
  NumelType newsize = 1;
  // N.B. this is an index, not a sym dim!
  std::optional<int64_t> infer_dim;
  for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
    if (TORCH_GUARD_OR_FALSE(sym_eq(shape[dim], -1))) {
      if (infer_dim) {
        throw std::runtime_error("only one dimension can be inferred");
      }
      infer_dim = dim;
    } else {
      // in case of unbacked shape[dim] we assume it's not -1 and add a runtime
      // assertion.
      TORCH_MAYBE_SYM_CHECK(
          sym_gt(shape[dim], -1),
          "invalid shape dimension ",
          shape[dim],
          " at index ",
          dim,
          " of shape ",
          shape);
      newsize *= shape[dim];
    }
  }

  auto set_infer_dim = [&]() {
    // We have a degree of freedom here to select the dimension size; follow
    // NumPy semantics and just bail.  However, a nice error message is needed
    // because users often use `view` as a way to flatten & unflatten
    // dimensions and will otherwise be confused why
    //   empty_tensor.view( 0, 0)
    // works yet
    //   empty_tensor.view(-1, 0)
    // doesn't.
    TORCH_CHECK(
        newsize != 0,
        "cannot reshape tensor of 0 elements into shape ",
        shape,
        " because the unspecified dimension size -1 can be any "
        "value and is ambiguous");
    res[*infer_dim] = numel / newsize;
    return;
  };

  if (infer_dim && newsize > 0 && numel % newsize == 0) {
    set_infer_dim();
    return;
  }

  TORCH_MAYBE_SYM_CHECK(
      sym_eq(numel, newsize),
      "shape '",
      shape,
      "' is invalid for input of size ",
      numel);
  if (infer_dim) {
    set_infer_dim();
  }
}

inline std::vector<int64_t> infer_size(IntArrayRef shape, int64_t numel) {
  auto res = shape.vec();
  infer_size_impl(shape, numel, res);
  return res;
}

inline at::DimVector infer_size_dv(IntArrayRef shape, int64_t numel) {
  auto res = at::DimVector(shape);
  infer_size_impl(shape, numel, res);
  return res;
}

inline at::SymDimVector infer_size_dv(
    c10::SymIntArrayRef shape,
    c10::SymInt numel) {
  auto res = at::SymDimVector(shape);
  infer_size_impl<c10::SymIntArrayRef, c10::SymInt, at::SymDimVector>(
      shape, std::move(numel), res);
  return res;
}

} // namespace at
