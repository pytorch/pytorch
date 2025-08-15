#pragma once

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch::aot_inductor {

enum class DynamicArgType : int {
  TensorType = 0,
  ListTensorType = 1,
  ListOptionalTensorType = 2,
  IntType = 3,
  ListIntType = 4,
  NoneType = 5,
};

inline bool isTensorType(DynamicArgType arg_type) {
  return arg_type == DynamicArgType::TensorType ||
      arg_type == DynamicArgType::ListTensorType ||
      arg_type == DynamicArgType::ListOptionalTensorType;
}

class ProxyExecutor {
 public:
  ProxyExecutor() = default;
  virtual ~ProxyExecutor() = default;

  virtual void call_function(
      int extern_node_index,
      int num_ints,
      int64_t* flatten_int_args,
      int num_tensors,
      AtenTensorHandle* flatten_tensor_args) = 0;
};

} // namespace torch::aot_inductor
