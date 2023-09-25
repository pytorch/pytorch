#pragma once

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <string>

namespace torch {
namespace aot_inductor {

class TORCH_API ProxyExecutor : public torch::CustomClassHolder {
 public:
  ProxyExecutor() {}
  virtual ~ProxyExecutor() {}

  virtual void call_function(
      int extern_node_index,
      int num_ints,
      int64_t* flatten_int_args,
      int num_tensors,
      void** flatten_tensor_args) = 0;
};

} // namespace aot_inductor
} // namespace torch
