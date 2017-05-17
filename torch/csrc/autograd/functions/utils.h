#pragma once

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


variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           function_constructor ctr);

}}
