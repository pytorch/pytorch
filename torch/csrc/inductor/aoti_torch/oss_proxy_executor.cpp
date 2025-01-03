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
      TORCH_CHECK(
          serialized_arg_type == "as_tensor",
          "Expected extern kernel ",
          op_kernel.target_,
          " to have serialized argument type as_tensor for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.emplace_back();
      dynamic_args.emplace_back(index, DynamicArgType::TensorType, 1);
      break;
    }
    case c10::TypeKind::IntType: {
      TORCH_CHECK(
          serialized_arg_type == "as_int",
          "Expected extern kernel ",
          op_kernel.target_,
          " to have serialized argument type as_int for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.emplace_back();
      dynamic_args.emplace_back(index, DynamicArgType::IntType, 1);
      break;
    }
    case c10::TypeKind::SymIntType: {
      TORCH_CHECK(
          serialized_arg_type == "as_int" ||
              serialized_arg_type == "as_sym_int",
          "Expected extern kernel ",
          op_kernel.target_,
          " to have serialized argument type as_int or as_sym_int for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.emplace_back();
      dynamic_args.emplace_back(index, DynamicArgType::IntType, 1);
      break;
    }
    case c10::TypeKind::FloatType: {
      TORCH_CHECK(
          serialized_arg_type == "as_float",
          "Expected extern kernel ",
          op_kernel.target_,
          " to have serialized argument type as_float for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.emplace_back(serialized_arg_val.get<double>());
      break;
    }
    case c10::TypeKind::BoolType: {
      TORCH_CHECK(
          serialized_arg_type == "as_bool",
          "Expected extern kernel ",
          op_kernel.target_,
          " to have serialized argument type as_bool for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.emplace_back(serialized_arg_val.get<bool>());
      break;
    }
    case c10::TypeKind::NumberType: {
      if (serialized_arg_type == "as_int") {
        // Only int Scalar is treated as dynamic arg for now
        stack.emplace_back();
        dynamic_args.emplace_back(index, DynamicArgType::IntType, 1);
      } else if (serialized_arg_type == "as_float") {
        stack.emplace_back(serialized_arg_val.get<double>());
      } else if (serialized_arg_type == "as_bool") {
        stack.emplace_back(serialized_arg_val.get<bool>());
      } else {
        TORCH_CHECK(
            false,
            "Expected extern kernel ",
            op_kernel.target_,
            " to have a scalar input for argument ",
            index,
            " but got ",
            serialized_arg_type);
      }
      break;
    }
    case c10::TypeKind::StringType: {
      TORCH_CHECK(
          serialized_arg_type == "as_string",
          "Expected extern kernel ",
          op_kernel.target_,
          " to have serialized argument type as_string for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.emplace_back(serialized_arg_val.get<std::string>());
      break;
    }
    case c10::TypeKind::DeviceObjType: {
      TORCH_CHECK(
          serialized_arg_type == "as_device",
          "Expected extern kernel ",
          op_kernel.target_,
          " to have serialized argument type as_device for argument ",
          index,
          " but got ",
          serialized_arg_type);

      std::string device_string = serialized_arg_val["type"].get<std::string>();
      if (serialized_arg_val["index"].is_number()) {
        device_string += ":" + serialized_arg_val["index"].get<std::string>();
      }

      c10::Device device(device_string);

      if (device != *device_) {
        VLOG(1) << "ProxyExecutor is using " << *device_ << " for "
                << op_kernel.target_ << " argument #" << index
                << ", which is different from the one serialized in thrift: "
                << device << ". Please ensure this is intentional.";
      }

      stack.emplace_back(*device_);
      break;
    }
    case c10::TypeKind::ListType: {
      if (schema_arg_type->isSubtypeOf(at::ListType::ofTensors())) {
        TORCH_CHECK(
            serialized_arg_type == "as_tensors",
            "Expected extern kernel ",
            op_kernel.target_,
            " to have serialized argument type as_tensors for argument ",
            index,
            " but got ",
            serialized_arg_type);
        TORCH_CHECK(serialized_arg_type == "as_tensors");
        stack.emplace_back();
        dynamic_args.emplace_back(
            index, DynamicArgType::ListTensorType, serialized_arg_val.size());
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofInts())) {
        TORCH_CHECK(
            serialized_arg_type == "as_ints",
            "Expected extern kernel ",
            op_kernel.target_,
            " to have serialized argument type as_ints for argument ",
            index,
            " but got ",
            serialized_arg_type);
        dynamic_args.emplace_back(
            index, DynamicArgType::ListIntType, serialized_arg_val.size());
        stack.emplace_back();
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofSymInts())) {
        TORCH_CHECK(
            serialized_arg_type == "as_ints" ||
                serialized_arg_type == "as_sym_ints",
            "Expected extern kernel ",
            op_kernel.target_,
            " to have serialized argument type as_ints or as_sym_ints for argument ",
            index,
            " but got ",
            serialized_arg_type);
        dynamic_args.emplace_back(
            index, DynamicArgType::ListIntType, serialized_arg_val.size());
        stack.emplace_back();
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofFloats())) {
        TORCH_CHECK(
            serialized_arg_type == "as_floats",
            "Expected extern kernel ",
            op_kernel.target_,
            " to have serialized argument type as_floats for argument ",
            index,
            " but got ",
            serialized_arg_type);
        std::vector<double> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<double>());
        }
        stack.emplace_back(ret);
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofBools())) {
        TORCH_CHECK(
            serialized_arg_type == "as_bools",
            "Expected extern kernel ",
            op_kernel.target_,
            " to have serialized argument type as_bools for argument ",
            index,
            " but got ",
            serialized_arg_type);
        std::vector<bool> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<bool>());
        }
        stack.emplace_back(ret);
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofNumbers())) {
        if (serialized_arg_type == "as_ints") {
          dynamic_args.emplace_back(
              index, DynamicArgType::ListIntType, serialized_arg_val.size());
          stack.emplace_back();
        } else if (serialized_arg_type == "as_floats") {
          std::vector<double> ret;
          for (const auto& arg : serialized_arg_val) {
            ret.push_back(arg);
          }
          stack.emplace_back(ret);
        } else if (serialized_arg_type == "as_bools") {
          std::vector<bool> ret;
          for (const auto& arg : serialized_arg_val) {
            ret.push_back(arg);
          }
          stack.emplace_back(ret);
        } else {
          TORCH_CHECK(
              false,
              "Expected extern kernel ",
              op_kernel.target_,
              " to have a List[Scalar] input for argument ",
              index,
              " but got ",
              serialized_arg_type);
        }
      } else if (schema_arg_type->isSubtypeOf(
                     at::ListType::ofOptionalTensors())) {
        if (serialized_arg_type == "as_optional_tensors") {
          std::vector<std::string> list_item_types;
          for (const auto& arg : serialized_arg_val) {
            list_item_types.push_back(arg.begin().key());
          }
          stack.emplace_back();
          dynamic_args.emplace_back(
              index,
              DynamicArgType::ListOptionalTensorType,
              serialized_arg_val.size(),
              list_item_types);
        } else if (serialized_arg_type == "as_tensors") {
          stack.emplace_back();
          dynamic_args.emplace_back(
              index, DynamicArgType::ListTensorType, serialized_arg_val.size());
        } else {
          TORCH_CHECK(
              false,
              "Expected extern kernel ",
              op_kernel.target_,
              " to have a Tensor?[] input for argument ",
              index,
              " but got ",
              serialized_arg_type);
        }
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofStrings())) {
        TORCH_CHECK(
            serialized_arg_type == "as_strings",
            "Expected extern kernel ",
            op_kernel.target_,
            " to have serialized argument type as_strings for argument ",
            index,
            " but got ",
            serialized_arg_type);
        std::vector<std::string> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<std::string>());
        }
        stack.emplace_back(ret);
      } else {
        TORCH_CHECK(
            false,
            "NYI: Unsupported list type ",
            serialized_arg_type,
            " for extern kernel ",
            op_kernel.target_,
            " argument ",
            index);
      }
      break;
    }
    case c10::TypeKind::OptionalType: {
      auto inner_type =
          schema_arg_type->castRaw<at::OptionalType>()->getElementType();

      if (serialized_arg_type == "as_none") {
        stack.emplace_back(std::nullopt);
        if (inner_type->kind() == c10::TypeKind::TensorType) {
          // Tensor is None
          dynamic_args.emplace_back(index, DynamicArgType::TensorType, 0);
        } else if (
            inner_type->kind() == c10::TypeKind::IntType ||
            inner_type->kind() == c10::TypeKind::SymIntType) {
          // Int or SymInt is None
          dynamic_args.emplace_back(index, DynamicArgType::IntType, 0);
        } else if (
            inner_type->kind() == c10::TypeKind::ListType &&
            schema_arg_type->isSubtypeOf(at::ListType::ofTensors())) {
          // List[Tensor] is None
          dynamic_args.emplace_back(index, DynamicArgType::ListTensorType, 0);
        } else if (
            inner_type->kind() == c10::TypeKind::ListType &&
            schema_arg_type->isSubtypeOf(at::ListType::ofSymInts())) {
          // List[SymInt] is None
          dynamic_args.emplace_back(index, DynamicArgType::ListIntType, 0);
        }
      } else {
        prefill_stack_with_static_arguments(
            index, inner_type, serialized_arg, op_kernel);
      }
      break;
    }
    // TODO: handle the other input types
    default:
      TORCH_CHECK(
          false,
          "Unsupported input type ",
          serialized_arg_type,
          " for extern kernel ",
          op_kernel.target_,
          " argument ",
          index);
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
      "Serialized node doesn't match operator ",
      serialized_node["target"],
      "'s schema outputs.");

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
            "Expected extern kernel ",
            serialized_node["target"],
            " to have serialized output type as_tensor, ",
            " but got ",
            serialized_output_type);
        outputs.emplace_back(output_index, DynamicArgType::TensorType, 1);
        break;
      }
      case c10::TypeKind::ListType: {
        if (schema_return_type->isSubtypeOf(at::ListType::ofTensors())) {
          TORCH_CHECK(
              serialized_output_type == "as_tensors",
              "Expected extern kernel ",
              serialized_node["target"],
              " to have serialized output type as_tensors, ",
              " but got ",
              serialized_output_type);
          outputs.emplace_back(
              output_index,
              DynamicArgType::ListTensorType,
              serialized_output_val.size());
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
            false,
            "Unsupported return type ",
            schema_return_type->repr_str(),
            " for extern kernel ",
            op_kernel.target_);
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

  std::string extern_kernel_nodes_serialized;

  std::ifstream json_file(json_path);
  TORCH_CHECK(json_file.is_open(), "Unable to open file ", json_path);

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
      case DynamicArgType::IntType: {
        int64_t val = flatten_int_args[int_id++];
        stack[arg_index] = val;
        break;
      }
      case DynamicArgType::ListTensorType: {
        std::vector<at::Tensor> tensor_list;
        for (int j = 0; j < length; j++) {
          at::Tensor* tensor =
              tensor_handle_to_tensor_pointer(flatten_tensor_args[tensor_id++]);
          tensor_list.push_back(*tensor);
        }
        stack[arg_index] = tensor_list;
        break;
      }
      case DynamicArgType::ListOptionalTensorType: {
        std::vector<std::optional<at::Tensor>> optional_tensor_list;
        auto& list_item_types = dynamic_arg.list_item_types;
        TORCH_CHECK(
            list_item_types.has_value(),
            "Could not find list of item types for optional tensor list input");

        for (const std::string& item_type : list_item_types.value()) {
          if (item_type == "as_tensor") {
            at::Tensor* tensor = tensor_handle_to_tensor_pointer(
                flatten_tensor_args[tensor_id++]);
            optional_tensor_list.emplace_back(*tensor);
          } else if (item_type == "as_none") {
            optional_tensor_list.emplace_back(std::nullopt);
          }
        }
        stack[arg_index] = optional_tensor_list;
        break;
      }
      case DynamicArgType::ListIntType: {
        std::vector<int64_t> vals;
        vals.reserve(length);
        for (int j = 0; j < length; j++) {
          vals.push_back(flatten_int_args[int_id++]);
        }
        stack[arg_index] = vals;
        break;
      }
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
    } else if (
        schema_return.type()->kind() == c10::TypeKind::ListType &&
        schema_return.type()->isSubtypeOf(at::ListType::ofTensors())) {
      auto tensors = stack[index++].toTensorList();
      for (auto&& t : tensors) {
        at::Tensor* tensor =
            tensor_handle_to_tensor_pointer(flatten_tensor_args[tensor_id++]);
        *tensor = t;
      }
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
