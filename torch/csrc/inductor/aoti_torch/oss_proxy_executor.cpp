#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/oss_proxy_executor.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace {
at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle) {
  return reinterpret_cast<at::Tensor*>(handle);
}

bool has_key(
    const std::unordered_map<std::string, c10::IValue>& map,
    const std::string& key) {
  return map.find(key) != map.end();
}

#ifdef _WIN32
const std::string k_separator = "\\";
#else
const std::string k_separator = "/";
#endif

} // namespace

namespace torch::aot_inductor {

void OSSProxyExecutor::prefill_stack_with_static_arguments(
    size_t index,
    const at::TypePtr& schema_arg_type,
    const nlohmann::json& serialized_arg,
    OSSOpKernel* op_kernel,
    const std::string& torchbind_obj_name) {
  auto& stack = op_kernel->stack_;
  auto& dynamic_args = op_kernel->dynamic_args_;
  auto& torchbind_args = op_kernel->torchbind_args_;

  TORCH_CHECK(serialized_arg.size() == 1);
  std::string serialized_arg_type = serialized_arg.begin().key();
  auto& serialized_arg_val = serialized_arg.begin().value();

  switch (schema_arg_type->kind()) {
    case c10::TypeKind::ClassType: {
      TORCH_CHECK(
          serialized_arg_type == "as_custom_obj",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_custom_obj for argument ",
          index,
          " but got ",
          serialized_arg_type);

      TORCH_CHECK(
          has_key(custom_objs_, torchbind_obj_name),
          "ProxyExecutor does not have a custom object named ",
          torchbind_obj_name,
          " from extern kernel ",
          op_kernel->target_,
          " argument ",
          index);

      LOG(INFO) << "Prefilling stack with torchbind argument "
                << torchbind_obj_name;
      torchbind_args.emplace_back(index, torchbind_obj_name);
      break;
    }
    case c10::TypeKind::TensorType: {
      TORCH_CHECK(
          serialized_arg_type == "as_tensor",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_tensor for argument ",
          index,
          " but got ",
          serialized_arg_type);
      dynamic_args.emplace_back(index, DynamicArgType::TensorType, 1);
      break;
    }
    case c10::TypeKind::IntType: {
      TORCH_CHECK(
          serialized_arg_type == "as_int",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_int for argument ",
          index,
          " but got ",
          serialized_arg_type);
      dynamic_args.emplace_back(index, DynamicArgType::IntType, 1);
      break;
    }
    case c10::TypeKind::SymIntType: {
      TORCH_CHECK(
          serialized_arg_type == "as_int" ||
              serialized_arg_type == "as_sym_int",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_int or as_sym_int for argument ",
          index,
          " but got ",
          serialized_arg_type);
      dynamic_args.emplace_back(index, DynamicArgType::IntType, 1);
      break;
    }
    case c10::TypeKind::FloatType: {
      TORCH_CHECK(
          serialized_arg_type == "as_float",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_float for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.at(index) = serialized_arg_val.get<double>();
      break;
    }
    case c10::TypeKind::BoolType: {
      TORCH_CHECK(
          serialized_arg_type == "as_bool",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_bool for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.at(index) = serialized_arg_val.get<bool>();
      break;
    }
    case c10::TypeKind::NumberType: {
      if (serialized_arg_type == "as_int") {
        // Only int Scalar is treated as dynamic arg for now
        dynamic_args.emplace_back(index, DynamicArgType::IntType, 1);
      } else if (serialized_arg_type == "as_float") {
        stack.at(index) = serialized_arg_val.get<double>();
      } else if (serialized_arg_type == "as_bool") {
        stack.at(index) = serialized_arg_val.get<bool>();
      } else {
        TORCH_CHECK(
            false,
            "Expected extern kernel ",
            op_kernel->target_,
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
          op_kernel->target_,
          " to have serialized argument type as_string for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.at(index) = serialized_arg_val.get<std::string>();
      break;
    }
    case c10::TypeKind::ScalarTypeType: {
      TORCH_CHECK(
          serialized_arg_type == "as_scalar_type",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_scalar_type for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.at(index) = serialized_arg_val.get<c10::ScalarType>();
      break;
    }
    case c10::TypeKind::MemoryFormatType: {
      TORCH_CHECK(
          serialized_arg_type == "as_memory_format",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_memory_format for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.at(index) = serialized_arg_val.get<c10::MemoryFormat>();
      break;
    }
    case c10::TypeKind::LayoutType: {
      TORCH_CHECK(
          serialized_arg_type == "as_layout",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_layout for argument ",
          index,
          " but got ",
          serialized_arg_type);
      stack.at(index) = serialized_arg_val.get<c10::Layout>();
      break;
    }
    case c10::TypeKind::DeviceObjType: {
      TORCH_CHECK(
          serialized_arg_type == "as_device",
          "Expected extern kernel ",
          op_kernel->target_,
          " to have serialized argument type as_device for argument ",
          index,
          " but got ",
          serialized_arg_type);

      std::string device_string = serialized_arg_val["type"].get<std::string>();
      if (serialized_arg_val.contains("index") &&
          serialized_arg_val["index"].is_number()) {
        auto index = serialized_arg_val["index"].get<int>();
        device_string += ":" + std::to_string(index);
        device_->set_index(static_cast<int8_t>(index));
      }

      c10::Device device(device_string);

      if (device.type() != device_->type()) {
        VLOG(1) << "ProxyExecutor is using " << *device_ << " for "
                << op_kernel->target_ << " argument #" << index
                << ", which is different from the one serialized in thrift: "
                << device << ". Please ensure this is intentional.";
      }

      stack.at(index) = *device_;
      break;
    }
    case c10::TypeKind::ListType: {
      if (schema_arg_type->isSubtypeOf(at::ListType::ofTensors())) {
        TORCH_CHECK(
            serialized_arg_type == "as_tensors",
            "Expected extern kernel ",
            op_kernel->target_,
            " to have serialized argument type as_tensors for argument ",
            index,
            " but got ",
            serialized_arg_type);
        TORCH_CHECK(serialized_arg_type == "as_tensors");
        dynamic_args.emplace_back(
            index, DynamicArgType::ListTensorType, serialized_arg_val.size());
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofInts())) {
        TORCH_CHECK(
            serialized_arg_type == "as_ints",
            "Expected extern kernel ",
            op_kernel->target_,
            " to have serialized argument type as_ints for argument ",
            index,
            " but got ",
            serialized_arg_type);
        dynamic_args.emplace_back(
            index, DynamicArgType::ListIntType, serialized_arg_val.size());
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofSymInts())) {
        TORCH_CHECK(
            serialized_arg_type == "as_ints" ||
                serialized_arg_type == "as_sym_ints",
            "Expected extern kernel ",
            op_kernel->target_,
            " to have serialized argument type as_ints or as_sym_ints for argument ",
            index,
            " but got ",
            serialized_arg_type);
        dynamic_args.emplace_back(
            index, DynamicArgType::ListIntType, serialized_arg_val.size());
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofFloats())) {
        TORCH_CHECK(
            serialized_arg_type == "as_floats",
            "Expected extern kernel ",
            op_kernel->target_,
            " to have serialized argument type as_floats for argument ",
            index,
            " but got ",
            serialized_arg_type);
        std::vector<double> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<double>());
        }
        stack.at(index) = std::move(ret);
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofBools())) {
        TORCH_CHECK(
            serialized_arg_type == "as_bools",
            "Expected extern kernel ",
            op_kernel->target_,
            " to have serialized argument type as_bools for argument ",
            index,
            " but got ",
            serialized_arg_type);
        std::vector<bool> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<bool>());
        }
        stack.at(index) = std::move(ret);
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofNumbers())) {
        if (serialized_arg_type == "as_ints") {
          dynamic_args.emplace_back(
              index, DynamicArgType::ListIntType, serialized_arg_val.size());
        } else if (serialized_arg_type == "as_floats") {
          std::vector<double> ret;
          for (const auto& arg : serialized_arg_val) {
            ret.push_back(arg);
          }
          stack.at(index) = std::move(ret);
        } else if (serialized_arg_type == "as_bools") {
          std::vector<bool> ret;
          for (const auto& arg : serialized_arg_val) {
            ret.push_back(arg);
          }
          stack.at(index) = std::move(ret);
        } else {
          TORCH_CHECK(
              false,
              "Expected extern kernel ",
              op_kernel->target_,
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
          dynamic_args.emplace_back(
              index,
              DynamicArgType::ListOptionalTensorType,
              serialized_arg_val.size(),
              list_item_types);
        } else if (serialized_arg_type == "as_tensors") {
          dynamic_args.emplace_back(
              index, DynamicArgType::ListTensorType, serialized_arg_val.size());
        } else {
          TORCH_CHECK(
              false,
              "Expected extern kernel ",
              op_kernel->target_,
              " to have a Tensor?[] input for argument ",
              index,
              " but got ",
              serialized_arg_type);
        }
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofStrings())) {
        TORCH_CHECK(
            serialized_arg_type == "as_strings",
            "Expected extern kernel ",
            op_kernel->target_,
            " to have serialized argument type as_strings for argument ",
            index,
            " but got ",
            serialized_arg_type);
        std::vector<std::string> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<std::string>());
        }
        stack.at(index) = std::move(ret);
      } else {
        TORCH_CHECK(
            false,
            "NYI: Unsupported list type ",
            serialized_arg_type,
            " for extern kernel ",
            op_kernel->target_,
            " argument ",
            index);
      }
      break;
    }
    case c10::TypeKind::OptionalType: {
      auto inner_type =
          schema_arg_type->castRaw<at::OptionalType>()->getElementType();

      if (serialized_arg_type == "as_none") {
        stack.at(index) = c10::IValue{};
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
            index, inner_type, serialized_arg, op_kernel, torchbind_obj_name);
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
          op_kernel->target_,
          " argument ",
          index);
  }
}

// Populates op_kernel.stack_, op_kernel.dynamic_args_
void OSSProxyExecutor::get_input_info_from_serialized(
    const std::vector<c10::Argument>& schema_args,
    const nlohmann::json& serialized_node,
    OSSOpKernel& op_kernel) {
  std::vector<bool> filled(schema_args.size(), false);
  TORCH_CHECK(op_kernel.stack_.empty());
  op_kernel.stack_.resize(schema_args.size());
  for (const auto& named_argument : serialized_node["inputs"]) {
    const auto& arg = named_argument["arg"];
    const auto& name = named_argument["name"].get<std::string>();

    std::string custom_obj_name = "";
    if (arg.contains("as_custom_obj")) {
      custom_obj_name = arg["as_custom_obj"]["name"].get<std::string>();
    }

    // Doing a linear lookup in the schema to find the index
    // of a static argument. Should be fine performance wise
    // because we usually only have small amount of arguments.
    for (size_t index = 0; index < schema_args.size(); index++) {
      auto& schema_arg = schema_args[index];
      if (schema_arg.name() == name) {
        prefill_stack_with_static_arguments(
            index, schema_arg.real_type(), arg, &op_kernel, custom_obj_name);
        filled[index] = true;
        break;
      }
    }
  }

  // If an argument is not filled and has a default value, we should
  // also prefill the default value.
  for (size_t index = 0; index < schema_args.size(); index++) {
    auto default_value = schema_args[index].default_value();
    if (!filled[index] && default_value.has_value()) {
      op_kernel.stack_.at(index) = std::move(default_value.value());
    }
  }
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
      case c10::TypeKind::NoneType: {
        TORCH_CHECK(
            serialized_output_type == "as_none",
            "Expected extern kernel ",
            serialized_node["target"],
            " to have serialized output type as_none, ",
            " but got ",
            serialized_output_type);
        outputs.emplace_back(output_index, DynamicArgType::NoneType, 1);
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
      case c10::TypeKind::OptionalType: {
        auto inner_type =
            schema_return_type->castRaw<at::OptionalType>()->getElementType();
        if (inner_type->kind() == c10::TypeKind::TensorType) {
          TORCH_CHECK(serialized_output_type == "as_optional_tensor");
          if (serialized_output_val.begin().key() == "as_none") {
            outputs.emplace_back(output_index, DynamicArgType::NoneType, 1);
          } else if (serialized_output_val.begin().key() == "as_tensor") {
            outputs.emplace_back(output_index, DynamicArgType::TensorType, 1);
          } else {
            TORCH_CHECK(
                false,
                "Only as_none or as_tensor is supported for as_optional_tensor");
          }
        }
        break;
      }
      case c10::TypeKind::IntType: {
        TORCH_CHECK(
            serialized_output_type == "as_int",
            "Expected extern kernel ",
            serialized_node["target"],
            " to have serialized output type as_int, ",
            " but got ",
            serialized_output_type);
        outputs.emplace_back(output_index, DynamicArgType::IntType, 1);
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

std::unique_ptr<OSSCallTorchBindKernel> OSSProxyExecutor::
    get_call_torch_bind_kernel(const nlohmann::json& serialized_node) {
  // const std::string& target = serialized_node["target"].get<std::string>();
  TORCH_CHECK(
      serialized_node["inputs"].size() > 1,
      "Expects higher_order.call_torchbind to only have at least 2 attributes, object and methodName");

  const auto first_input = serialized_node["inputs"][0]["arg"]["as_custom_obj"];
  const std::string torchbind_obj_name = first_input["name"].get<std::string>();
  const std::string class_fqn = first_input["class_fqn"].get<std::string>();
  const std::string method_name =
      serialized_node["inputs"][1]["arg"]["as_string"].get<std::string>();

  auto customClassType_ = torch::jit::getCustomClass(class_fqn);
  auto method = customClassType_->findMethod(method_name);

  CHECK(method != nullptr) << "method not found: " << method_name;

  TORCH_CHECK(
      has_key(custom_objs_, torchbind_obj_name),
      "ProxyExecutor does not have a custom object named ",
      torchbind_obj_name,
      " from call_torchbind ");

  const c10::FunctionSchema& schema = method->getSchema();

  const auto& schema_args = schema.arguments();
  const auto& schema_returns = schema.returns();

  std::unique_ptr<OSSCallTorchBindKernel> op_kernel =
      std::make_unique<OSSCallTorchBindKernel>("call_torchbind", method);
  auto modified_serialized_node = serialized_node;
  // Remove the second elements (the method string) from inputs because they
  // are only for HOP
  auto& inputs = modified_serialized_node["inputs"];
  // Erase the second element (index 1)
  inputs.erase(inputs.begin() + 1);

  get_input_info_from_serialized(
      schema_args, modified_serialized_node, *op_kernel);
  get_output_info_from_serialized(schema_returns, serialized_node, *op_kernel);
  return op_kernel;
}

OSSProxyExecutor::OSSProxyExecutor(
    const std::string& json_path,
    bool is_cpu,
    std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs) {
  if (is_cpu) {
    device_ = std::make_unique<c10::Device>(c10::DeviceType::CPU);
  } else {
    int device_idx = -1;
    device_ = std::make_unique<c10::Device>(c10::DeviceType::CUDA, device_idx);
  }

  // If custom_objs is provided, use it instead of loading from
  // custom_objs_config.json If custom_objs is not provided, try to load from
  // custom_objs_config.json
  if (custom_objs.has_value()) {
    custom_objs_ = std::move(custom_objs.value());
  } else {
    // Load custom objects from custom_objs_config.json file
    // Get the constants json path from the extern_kernel_nodes .json file

    size_t lastSlash = json_path.find_last_of("/\\");
    std::string folder_path = json_path.substr(0, lastSlash);
    std::string custom_objs_json_path =
        folder_path + k_separator + "custom_objs_config.json";
    LOG(INFO) << "Loading custom_objs_config .json file from "
              << custom_objs_json_path;

    std::ifstream custom_objs_json_file(custom_objs_json_path);

    if (!custom_objs_json_file.is_open()) {
      // BC-compatible with old files that don't have custom_objs_config.json
      LOG(INFO) << "Unable to open custom objs json file "
                << custom_objs_json_path;
    } else {
      nlohmann::json custom_objs_json;
      custom_objs_json_file >> custom_objs_json;
      // Load custom objects from binary torchbind file
      for (auto& [customObjName, file_name] : custom_objs_json.items()) {
        std::string customObjPath =
            folder_path + k_separator + file_name.get<std::string>();
        LOG(INFO) << "Loading custom object to FbProxyExecutor from: "
                  << customObjPath;

        std::ifstream custom_obj_file(customObjPath, std::ios::binary);
        TORCH_CHECK(
            custom_obj_file.is_open(), "Failed to open custom obj file");
        std::vector<char> customObjData(
            (std::istreambuf_iterator<char>(custom_obj_file)),
            std::istreambuf_iterator<char>());
        custom_obj_file.close();

        std::string customObjBytes(customObjData.data(), customObjData.size());

        c10::IValue custom_obj = torch::jit::pickle_load_obj(customObjBytes);
        CHECK(custom_obj.isCustomClass());
        CHECK(!custom_obj.isNone());
        custom_objs_[customObjName] = std::move(custom_obj);
      }
    }
  }

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
      size_t pos2 = target.find('.', pos + 1);
      TORCH_CHECK(pos2 == std::string::npos);

      opName = target.substr(0, pos);
      overloadName = target.substr(pos + 1, target.length() - pos);
    }

    if (target == "call_torchbind") {
      // Special handling for CallTorchBind HOP
      std::unique_ptr<OSSCallTorchBindKernel> op_kernel =
          get_call_torch_bind_kernel(serialized_node);
      op_kernels_.emplace_back(std::move(op_kernel));
    } else {
      c10::OperatorHandle op_handle =
          c10::Dispatcher::singleton().findSchemaOrThrow(
              opName.c_str(), overloadName.c_str());
      const c10::FunctionSchema& schema = op_handle.schema();

      const auto& schema_args = schema.arguments();
      const auto& schema_returns = schema.returns();

      std::unique_ptr<OSSOpKernelOperator> op_kernel =
          std::make_unique<OSSOpKernelOperator>(target, op_handle);
      get_input_info_from_serialized(schema_args, serialized_node, *op_kernel);
      get_output_info_from_serialized(
          schema_returns, serialized_node, *op_kernel);
      op_kernels_.emplace_back(std::move(op_kernel));
    }
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
  auto& op_kernel = op_kernels_[extern_node_index];

  std::vector<c10::IValue> stack = op_kernel->stack_;
  auto& dynamic_args = op_kernel->dynamic_args_;
  auto& torchbind_args = op_kernel->torchbind_args_;

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

  for (auto& torchbind_arg : torchbind_args) {
    int arg_index = torchbind_arg.arg_index;
    stack[arg_index] = custom_objs_[torchbind_arg.arg_name];
  }

  int num_output_tensors = op_kernel->num_output_tensors();
  TORCH_CHECK(
      tensor_id == num_tensors - num_output_tensors,
      "Mismatch between tensors consumed and num of input tensor, got tensor_id = ",
      tensor_id,
      ", expected num = ",
      num_tensors - num_output_tensors);

  int num_output_ints = op_kernel->num_output_ints();
  TORCH_CHECK(
      int_id == num_ints - num_output_ints,
      "Mismatch between ints consumed and num_ints, got int_id = ",
      int_id,
      ", num_ints = ",
      num_ints - num_output_ints);

  // Call the op with the prepared stack.
  op_kernel->run(stack);

  const c10::FunctionSchema& schema = op_kernel->schema();
  const auto& schema_returns = schema.returns();

  TORCH_CHECK(op_kernel->outputs_.size() == stack.size());
  TORCH_CHECK(stack.size() == schema_returns.size());

  int index = 0;
  for (const auto& schema_return : schema_returns) {
    if (schema_return.type()->kind() == c10::TypeKind::TensorType) {
      at::Tensor* tensor =
          tensor_handle_to_tensor_pointer(flatten_tensor_args[tensor_id++]);
      *tensor = stack[index++].toTensor();
    } else if (schema_return.type()->kind() == c10::TypeKind::NoneType) {
      continue;
    } else if (
        schema_return.type()->kind() == c10::TypeKind::ListType &&
        schema_return.type()->isSubtypeOf(at::ListType::ofTensors())) {
      auto tensors = stack[index++].toTensorList();
      for (auto&& t : tensors) {
        at::Tensor* tensor =
            tensor_handle_to_tensor_pointer(flatten_tensor_args[tensor_id++]);
        *tensor = t;
      }
    } else if (
        schema_return.type()->kind() == c10::TypeKind::OptionalType &&
        schema_return.type()
                ->castRaw<at::OptionalType>()
                ->getElementType()
                ->kind() == c10::TypeKind::TensorType) {
      if (op_kernel->outputs_[index].arg_type == DynamicArgType::TensorType) {
        auto stack_tensor = stack[index++].toOptional<at::Tensor>();
        at::Tensor* tensor =
            tensor_handle_to_tensor_pointer(flatten_tensor_args[tensor_id++]);
        if (stack_tensor.has_value()) {
          *tensor = stack_tensor.value();
        } else {
          TORCH_CHECK(false, "Expected tensor, got None");
        }
      } else {
        index++;
      }
    } else if (schema_return.real_type()->kind() == c10::TypeKind::IntType) {
      // need to use real_type() to differentiate between IntType and SymIntType
      // for int type, it is already specialized in downstream kernels. So we
      // don't need to do anything here.
      auto returned_int_value = stack[index++].toInt();
      auto serialized_int_value = flatten_int_args[int_id++];
      TORCH_CHECK(
          returned_int_value == serialized_int_value,
          "Expect returned int value to match the serialized int value, but got returned int value: ",
          returned_int_value,
          " and serialized int value: ",
          serialized_int_value);
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

  TORCH_CHECK(
      int_id == num_ints,
      "Mismatch between tensors consumed and num_ints, got tensor_id = ",
      int_id,
      ", expected num = ",
      num_ints);
}

} // namespace torch::aot_inductor
