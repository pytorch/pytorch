#pragma once

#include "torch/csrc/utils/pybind.h"

namespace torch { namespace jit {

inline Stack createStack(const py::tuple& tuple, size_t reserve_extra_space = 0) {
  Stack result;
  result.reserve(tuple.size() + reserve_extra_space);
  for(auto e : tuple) {
    result.push_back(py::cast<autograd::Variable>(e));
  }
  return result;
}

inline py::object wrapStack(Stack&& outputs) {
  if (outputs.size() == 0) {
    return py::none();
  } else if (outputs.size() == 1) {
    JIT_ASSERT(outputs[0].isTensor());
    return py::cast(autograd::as_variable_ref(std::move(outputs[0]).toTensor()));
  } else {
    py::tuple tuple(outputs.size());
    for(size_t i = 0; i < outputs.size(); i++) {
      JIT_ASSERT(outputs[i].isTensor());
      tuple[i] = py::cast(autograd::as_variable_ref(std::move(outputs[i]).toTensor()));
    }
    return tuple;
  }
}

} }  // namespace torch::jit
