#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>
#include <iostream>
#include <utility>

namespace torch::aot_inductor {

enum class DynamicArgType : int {
  TensorType = 0,
  ListTensorType = 1,
  ListOptionalTensorType = 2,
  IntType = 3,
  ListIntType = 4,
};

inline std::ostream& operator<<(std::ostream& os, DynamicArgType arg_type) {
  os << static_cast<int>(arg_type);
  return os;
}

inline bool isTensorType(DynamicArgType arg_type) {
  return arg_type == DynamicArgType::TensorType ||
      arg_type == DynamicArgType::ListTensorType ||
      arg_type == DynamicArgType::ListOptionalTensorType;
}

struct OSSDynamicArg {
  OSSDynamicArg(
      int arg_index,
      DynamicArgType arg_type,
      int length,
      std::optional<std::vector<std::string>> list_item_types = std::nullopt)
      : arg_index(arg_index),
        arg_type(arg_type),
        length(length),
        list_item_types(std::move(list_item_types)) {}
  int arg_index;
  DynamicArgType arg_type;
  int length;
  std::optional<std::vector<std::string>>
      list_item_types; // only used for parsing list of optional tensors
};

struct OSSOpKernel {
  OSSOpKernel(std::string target, c10::OperatorHandle op_handle)
      : target_(std::move(target)), op_handle_(std::move(op_handle)) {}

  std::string target_;
  c10::OperatorHandle op_handle_;
  std::vector<OSSDynamicArg> dynamic_args_;
  std::vector<OSSDynamicArg> outputs_;
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

class OSSProxyExecutor : public ProxyExecutor {
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
      const at::TypePtr& schema_arg_type,
      const nlohmann::json& serialized_arg,
      OSSOpKernel& op_kernel);

  void get_input_info_from_serialized(
      const std::vector<c10::Argument>& schema_args,
      const nlohmann::json& serialized_node,
      OSSOpKernel& op_kernel);

  void get_output_info_from_serialized(
      const std::vector<c10::Argument>& schema_returns,
      const nlohmann::json& serialized_node,
      OSSOpKernel& op_kernel);

  std::vector<OSSOpKernel> op_kernels_;
  std::unique_ptr<c10::Device> device_;
};

} // namespace torch::aot_inductor
