#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace torch {
namespace aot_inductor {

enum DynamicArgType : int {
  TensorType = 0,
  ListTensorType = 1,
  ListOptionalTensorType = 2,
  IntType = 3,
  ListIntType = 4,
};

inline bool isTensorType(DynamicArgType arg_type) {
  return arg_type == DynamicArgType::TensorType ||
      arg_type == DynamicArgType::ListTensorType ||
      arg_type == DynamicArgType::ListOptionalTensorType;
}

struct DynamicArg {
  DynamicArg(
      int arg_index,
      DynamicArgType arg_type,
      int length,
      std::string serialized_arg_type)
      : arg_index(arg_index),
        arg_type(arg_type),
        length(length),
        serialized_arg_type(serialized_arg_type) {}
  int arg_index;
  DynamicArgType arg_type;
  int length;
  std::string serialized_arg_type;
};

struct OpKernel {
  OpKernel(const std::string& target, const c10::OperatorHandle& op_handle)
      : target_(target), op_handle_(op_handle) {}

  std::string target_;
  c10::OperatorHandle op_handle_;
  std::vector<DynamicArg> dynamic_args_;
  std::vector<DynamicArg> outputs_;
  std::vector<c10::IValue> stack_;

  int num_output_tensors() const {
    int num_output_tensors = 0;
    for (const auto& output : outputs_) {
      if (isTensorType(output.arg_type)) {
        num_output_tensors += output.length;
      }
    }
    return num_output_tensors;
  }
};

class OSSProxyExecutor: public ProxyExecutor {
 public:
  explicit OSSProxyExecutor(const std::string& json_path, bool is_cpu);

  void call_function(
      int extern_node_index,
      int num_ints,
      int64_t* flatten_int_args,
      int num_tensors,
      AtenTensorHandle* flatten_tensor_args) override;

 private:
  void prefill_stack_with_static_arguments(
      int index,
      at::TypePtr schema_arg_type,
      json thrift_arg,
      OpKernel& op_kernel);

  void get_input_info_from_serialized(
      const std::vector<c10::Argument>& schema_args,
      json serialized_node,
      OpKernel& op_kernel);

  void get_output_info_from_serialized(
      const std::vector<c10::Argument>& schema_returns,
      json serialized_node,
      OpKernel& op_kernel);

  std::vector<OpKernel> op_kernels_;
  std::unique_ptr<c10::Device> device_;
};

} // namespace aot_inductor
} // namespace torch
