#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

#include <torch/csrc/inductor/aoti_torch/oss_proxy_executor.h>

namespace {
at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle) {
  return reinterpret_cast<at::Tensor*>(handle);
}
} // namespace

namespace torch::aot_inductor {

void OSSProxyExecutor::prefill_stack_with_static_arguments(
    int index,
    const at::TypePtr& schema_arg_type,
    const nlohmann::json& serialized_arg,
    OSSOpKernel& op_kernel) {
  auto& stack = op_kernel.stack_;
  auto& dynamic_args = op_kernel.dynamic_args_;

  TORCH_CHECK(serialized_arg.size() == 1);
  std::string serialized_arg_type = serialized_arg.begin().key();
  auto& serialized_arg_val = serialized_arg.begin().value();

  switch (schema_arg_type->kind()) {
    case c10::TypeKind::TensorType: {
      TORCH_CHECK(serialized_arg_type == "as_tensor");
      stack.emplace_back();
      dynamic_args.emplace_back(
          index, DynamicArgType::TensorType, 1, serialized_arg_val);
      break;
    }
    // TODO: handle the other input types
    default:
      TORCH_CHECK(false, "Unsupported input type ", serialized_arg_type);
  }
}

// Populates op_kernel.stack_, op_kernel.dynamic_args_
void OSSProxyExecutor::get_input_info_from_serialized(
    const std::vector<c10::Argument>& schema_args,
    const nlohmann::json& serialized_node,
    OSSOpKernel& op_kernel) {
  int index = 0;
  for (const auto& named_argument : serialized_node["inputs"]) {
    const auto& arg = named_argument["arg"];
    auto& schema_arg = schema_args[index];

    prefill_stack_with_static_arguments(
        index++, schema_arg.real_type(), arg, op_kernel);
  }

  // TODO: prefill default values
}

// Populates op_kernel.outputs_
void OSSProxyExecutor::get_output_info_from_serialized(
    const std::vector<c10::Argument>& schema_returns,
    const nlohmann::json& serialized_node,
    OSSOpKernel& op_kernel) {
  std::vector<OSSDynamicArg>& outputs = op_kernel.outputs_;

  TORCH_CHECK(
      schema_returns.size() == serialized_node["outputs"].size(),
      "Serialized node doesn't match op's schema outputs.");

  size_t output_index = 0;
  for (const auto& serialized_output : serialized_node["outputs"]) {
    TORCH_CHECK(serialized_output.size() == 1);
    std::string serialized_output_type = serialized_output.begin().key();
    auto& serialized_output_val = serialized_output.begin().value();

    auto& schema_return = schema_returns[output_index];
    const at::TypePtr& schema_return_type = schema_return.real_type();

    switch (schema_return_type->kind()) {
      case c10::TypeKind::TensorType: {
        TORCH_CHECK(
            serialized_output_type == "as_tensor",
            serialized_node["target"],
            " got serialized_output_type of ",
            serialized_output_type);
        outputs.emplace_back(
            output_index,
            DynamicArgType::TensorType,
            1,
            serialized_output_type);
        break;
      }
      case c10::TypeKind::ListType: {
        if (schema_return_type->isSubtypeOf(at::ListType::ofTensors())) {
          TORCH_CHECK(
              serialized_output_type == "as_tensors",
              serialized_node["target"],
              " got serialized_output_type of ",
              serialized_output_type);
          outputs.emplace_back(
              output_index,
              DynamicArgType::ListTensorType,
              serialized_output_val.size(),
              serialized_output_type);
        } else {
          TORCH_CHECK(
              false,
              "Unsupported return list type ",
              schema_return_type->repr_str());
        }
        break;
      }
      default: {
        TORCH_CHECK(
            false, "Unsupported return type ", schema_return_type->repr_str());
      }
    }

    output_index++;
  }
}

OSSProxyExecutor::OSSProxyExecutor(const std::string& json_path, bool is_cpu) {
  if (is_cpu) {
    device_ = std::make_unique<c10::Device>(c10::DeviceType::CPU);
  } else {
    int device_idx = -1;
    device_ = std::make_unique<c10::Device>(c10::DeviceType::CUDA, device_idx);
  }

  std::ifstream json_file(json_path);
  TORCH_CHECK(json_file.is_open());

  // Parse file into a json object
  nlohmann::json json_obj;
  json_file >> json_obj;

  // Access data
  for (auto const& serialized_extern_node : json_obj["nodes"]) {
    auto const& serialized_node = serialized_extern_node["node"];

    const std::string& target = serialized_node["target"];

    std::string opName;
    std::string overloadName;
    size_t pos = target.find('.');
    if (pos == std::string::npos) {
      opName = target;
      overloadName = "";
    } else {
      // There should be no more periods
      size_t pos2 = target.find('.', pos);
      TORCH_CHECK(pos2 == std::string::npos);

      opName = target.substr(0, pos);
      overloadName = target.substr(pos + 1, target.length() - pos);
    }

    c10::OperatorHandle op_handle =
        c10::Dispatcher::singleton().findSchemaOrThrow(
            opName.c_str(), overloadName.c_str());
    const c10::FunctionSchema& schema = op_handle.schema();

    const auto& schema_args = schema.arguments();
    const auto& schema_returns = schema.returns();

    OSSOpKernel op_kernel(target, op_handle);
    get_input_info_from_serialized(schema_args, serialized_node, op_kernel);
    get_output_info_from_serialized(schema_returns, serialized_node, op_kernel);

    op_kernels_.emplace_back(std::move(op_kernel));
  }
}

void OSSProxyExecutor::call_function(
    int extern_node_index,
    int num_ints,
    int64_t* flatten_int_args,
    int num_tensors,
    AtenTensorHandle* flatten_tensor_args) {
  TORCH_CHECK(
      extern_node_index < static_cast<int>(op_kernels_.size()),
      "Invalid extern node index");
  OSSOpKernel& op_kernel = op_kernels_[extern_node_index];

  std::vector<c10::IValue> stack = op_kernel.stack_;
  auto& dynamic_args = op_kernel.dynamic_args_;

  int tensor_id = 0;
  int int_id = 0;
  for (auto& dynamic_arg : dynamic_args) {
    int arg_index = dynamic_arg.arg_index;
    DynamicArgType dynamic_arg_type = dynamic_arg.arg_type;
    int length = dynamic_arg.length;

    if (length == 0) {
      continue;
    }

    switch (dynamic_arg_type) {
      case DynamicArgType::TensorType: {
        at::Tensor* tensor =
            tensor_handle_to_tensor_pointer(flatten_tensor_args[tensor_id++]);
        stack[arg_index] = *tensor;
        break;
      }
      // TODO: handle other dynamic arg types
      default:
        TORCH_CHECK(false, "Unsupported dynamic arg type: ", dynamic_arg_type);
    }
  }

  int num_output_tensors = op_kernel.num_output_tensors();
  TORCH_CHECK(
      tensor_id == num_tensors - num_output_tensors,
      "Mismatch between tensors consumed and num of input tensor, got tensor_id = .",
      tensor_id,
      ", expected num = ",
      num_tensors - num_output_tensors);
  TORCH_CHECK(
      int_id == num_ints,
      "Mismatch between ints consumed and num_ints, got int_id = ",
      int_id,
      ", num_ints = ",
      num_ints);

  // Call the op with the prepared stack.
  const c10::OperatorHandle& op = op_kernel.op_handle_;
  op.callBoxed(stack);

  const c10::FunctionSchema& schema = op.schema();
  const auto& schema_returns = schema.returns();

  TORCH_CHECK(op_kernel.outputs_.size() == stack.size());
  // TODO: what about optional outputs? This assert may not hold
  TORCH_CHECK(stack.size() == schema_returns.size());

  int index = 0;
  for (const auto& schema_return : schema_returns) {
    if (schema_return.type()->kind() == c10::TypeKind::TensorType) {
      at::Tensor* tensor =
          tensor_handle_to_tensor_pointer(flatten_tensor_args[tensor_id++]);
      *tensor = stack[index++].toTensor();
      // TODO: handle tensor list returns
    } else {
      TORCH_CHECK(
          false,
          "NYI: Unsupported return type for schema: ",
          schema_return.type()->repr_str());
    }
  }

  TORCH_CHECK(
      tensor_id == num_tensors,
      "Mismatch between tensors consumed and num_tensors, got tensor_id = ",
      tensor_id,
      ", expected num = ",
      num_tensors);
}

} // namespace torch::aot_inductor
