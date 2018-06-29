#pragma once

#include "torch/csrc/utils/pybind.h"

#include "torch/csrc/jit/variable_tensor_list.h"

namespace torch { namespace jit {

namespace {

// we cannot use the default py:cast<autograd::Variable> because it currently
// unwraps the data tensor in the conversion process
// TODO: replace with bs type
variable_tensor_list createVariableTensorList(py::tuple tuple, size_t reserve_extra_space = 0) {
  variable_tensor_list result;
  result.reserve(tuple.size() + reserve_extra_space);
  for(auto e : tuple) {
    result.push_back(py::cast<autograd::Variable>(e));
  }
  return result;
}

}  // namespace

} }  // namespace torch::jit
