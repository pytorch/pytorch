#pragma once

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch::aot_inductor {

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
