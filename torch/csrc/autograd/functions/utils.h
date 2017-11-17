#pragma once

#include <Python.h>
#include <functional>
#include <memory>
#include <array>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

using function_constructor = std::function<std::shared_ptr<Function>(FunctionFlags)>;

template<typename ...Args>
inline variable_list as_variable_list(Args&& ... args) {
  std::array<variable_list::value_type, sizeof...(args)> arr = { {std::move(args)...} };
  return variable_list(std::make_move_iterator(arr.begin()),
                       std::make_move_iterator(arr.end()));
}

template<typename ...Args>
inline tensor_list as_tensor_list(Args&& ... args) {
  std::array<tensor_list::value_type, sizeof...(args)> arr = { {std::move(args)...} };
  return tensor_list(std::make_move_iterator(arr.begin()),
                     std::make_move_iterator(arr.end()));
}


/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           function_constructor ctr);

/**
 * Checks that inputs contains exactly `args` items and that the first `required_args`
 * items are not NULL. If not specified, `required_args` defaults to `args`.
 */
void check_input_variables(const char* name, const variable_list& inputs, int args, int required_args=-1);

}}
