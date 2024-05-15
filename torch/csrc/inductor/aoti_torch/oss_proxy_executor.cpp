#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

#include <torch/csrc/inductor/aoti_torch/oss_proxy_executor.h>

using json = nlohmann::json;

namespace {
at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle) {
  return reinterpret_cast<at::Tensor*>(handle);
}
} // namespace

namespace torch {
namespace aot_inductor {

c10::ScalarType OSSProxyExecutor::convertSerializedScalarType(int scalarType) {
  // Numberings match with torch._export.serde.schema.ScalarType enum
  switch (scalarType) {
    case 0: // ScalarType.UNKNOWN
      TORCH_CHECK(false, "Unknown scalar type");
    case 1: // ScalarType.BYTE
      return c10::ScalarType::Byte;
    case 2: // ScalarType.CHAR
      return c10::ScalarType::Char;
    case 3: // ScalarType.SHORT
      return c10::ScalarType::Short;
    case 4: // ScalarType.INT
      return c10::ScalarType::Int;
    case 5: // ScalarType.LONG
      return c10::ScalarType::Long;
    case 6: // ScalarType.HALF
      return c10::ScalarType::Half;
    case 7: // ScalarType.FLOAT
      return c10::ScalarType::Float;
    case 8: // ScalarType.DOUBLE
      return c10::ScalarType::Double;
    case 9: // ScalarType.COMPLEXHALF
      return c10::ScalarType::ComplexHalf;
    case 10: // ScalarType.COMPLEXFLOAT
      return c10::ScalarType::ComplexFloat;
    case 11: // ScalarType.COMPLEXDOUBLE
      return c10::ScalarType::ComplexDouble;
    case 12: // ScalarType.BOOL
      return c10::ScalarType::Bool;
    case 13: // ScalarType.BFLOAT16
      return c10::ScalarType::BFloat16;
    default:
      TORCH_CHECK(false, "Unknown scalar type", scalarType);
  }
}

c10::MemoryFormat OSSProxyExecutor::convertSerializedMemoryFormat(
    int memoryFormat) {
  // Numberings match with torch._export.serde.schema.MemoryFormat enum
  switch (memoryFormat) {
    case 0: // MemoryFormat.Unknown
      TORCH_CHECK(false, "Unknown memory format", memoryFormat);
    case 1: // MemoryFormat.ContiguousFormat
      return c10::MemoryFormat::Contiguous;
    case 2: // MemoryFormat.ChannelsLast
      return c10::MemoryFormat::ChannelsLast;
    case 3: // MemoryFormat.ChannelsLast3d
      return c10::MemoryFormat::ChannelsLast3d;
    case 4: // MemoryFormat.PreserveFormat
      return c10::MemoryFormat::Preserve;
    default:
      TORCH_CHECK(false, "Unknown memory format", memoryFormat);
  }
}

c10::Layout OSSProxyExecutor::convertSerializedLayout(int layout) {
  // Numberings match with torch._export.serde.schema.Layout enum
  switch (layout) {
    case 0: // Layout.Unknown:
      TORCH_CHECK(false, "Got unknown layout", layout);
    case 1: // Layout.SparseCoo:
      // TODO is this the right translation
      return c10::Layout::Sparse;
    case 2: // Layout.SparseCsr:
      return c10::Layout::SparseCsr;
    case 3: // Layout.SparseCsc:
      return c10::Layout::SparseCsc;
    case 4: // Layout.SparseBsr:
      return c10::Layout::SparseBsr;
    case 5: // Layout.SparseBsc:
      return c10::Layout::SparseBsc;
    case 6: // Layout._mkldnn:
      return c10::Layout::Mkldnn;
    case 7: // Layout.Strided:
      return c10::Layout::Strided;
    default:
      TORCH_CHECK(false, "Unknown layout", layout);
  }
}

void OSSProxyExecutor::prefill_stack_with_static_arguments(
    int index,
    at::TypePtr schema_arg_type,
    json serialized_arg,
    OpKernel& op_kernel) {
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
    case c10::TypeKind::IntType: {
      TORCH_CHECK(serialized_arg_type == "as_int");
      stack.emplace_back(c10::IValue());
      dynamic_args.emplace_back(
          index, DynamicArgType::IntType, 1, serialized_arg_val);
      break;
    }
    case c10::TypeKind::SymIntType: {
      TORCH_CHECK(
          serialized_arg_type == "as_int" ||
          serialized_arg_type == "as_sym_int");
      stack.emplace_back(c10::IValue());
      dynamic_args.emplace_back(
          index, DynamicArgType::IntType, 1, serialized_arg_val);
      break;
    }
    case c10::TypeKind::FloatType: {
      TORCH_CHECK(serialized_arg_type == "as_float");
      stack.emplace_back(serialized_arg_val.get<double>());
      break;
    }
    case c10::TypeKind::BoolType: {
      TORCH_CHECK(serialized_arg_type == "as_bool");
      stack.emplace_back(serialized_arg_val.get<bool>());
      break;
    }
    case c10::TypeKind::NumberType: {
      if (serialized_arg_type == "as_int") {
        // Only int Scalar is treated as dynamic arg for now
        stack.emplace_back();
        dynamic_args.emplace_back(
            index, DynamicArgType::IntType, 1, serialized_arg_val);
      } else if (serialized_arg_type == "as_float") {
        stack.emplace_back(serialized_arg_val.get<double>());
      } else if (serialized_arg_type == "as_bool") {
        stack.emplace_back(serialized_arg_val.get<bool>());
      } else {
        TORCH_CHECK(
            false,
            "Invalid serialized argument type found for Scalar input: ",
            serialized_arg_type);
      }
      break;
    }
    case c10::TypeKind::StringType: {
      TORCH_CHECK(serialized_arg_type == "as_string");
      stack.emplace_back(serialized_arg_val.get<std::string>());
      break;
    }
    case c10::TypeKind::ScalarTypeType: {
      TORCH_CHECK(serialized_arg_type == "as_scalar_type");
      c10::ScalarType scalar_type =
          convertSerializedScalarType(serialized_arg_val.get<int>());
      stack.emplace_back(scalar_type);
      break;
    }
    case c10::TypeKind::MemoryFormatType: {
      TORCH_CHECK(serialized_arg_type == "as_memory_format");
      c10::MemoryFormat memory_format =
          convertSerializedMemoryFormat(serialized_arg_val.get<int>());
      stack.emplace_back(memory_format);
      break;
    }
    case c10::TypeKind::LayoutType: {
      TORCH_CHECK(serialized_arg_type == "as_layout");
      c10::Layout layout =
          convertSerializedLayout(serialized_arg_val.get<int>());
      stack.emplace_back(layout);
      break;
    }
    case c10::TypeKind::DeviceObjType: {
      TORCH_CHECK(serialized_arg_type == "as_device");

      std::string device_string = serialized_arg_val["type"].get<std::string>();
      if (!serialized_arg_val["index"].is_null()) {
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
        TORCH_CHECK(serialized_arg_type == "as_tensors");
        stack.emplace_back();
        dynamic_args.emplace_back(
            index,
            DynamicArgType::ListTensorType,
            serialized_arg_val.size(),
            serialized_arg_val);
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofInts())) {
        TORCH_CHECK(serialized_arg_type == "as_ints");
        dynamic_args.emplace_back(
            index,
            DynamicArgType::ListIntType,
            serialized_arg_val.size(),
            serialized_arg_val);
        stack.emplace_back(c10::IValue());
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofSymInts())) {
        TORCH_CHECK(
            serialized_arg_type == "as_ints" or
            serialized_arg_type == "as_sym_ints");
        dynamic_args.emplace_back(
            index,
            DynamicArgType::ListIntType,
            serialized_arg_val.size(),
            serialized_arg_val);
        stack.emplace_back(c10::IValue());
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofFloats())) {
        TORCH_CHECK(serialized_arg_type == "as_floats");
        std::vector<double> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<double>());
        }
        stack.emplace_back(ret);
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofBools())) {
        TORCH_CHECK(serialized_arg_type == "as_bools");
        std::vector<bool> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<bool>());
        }
        stack.emplace_back(ret);
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofNumbers())) {
        if (serialized_arg_type == "as_ints") {
          dynamic_args.emplace_back(
              index,
              DynamicArgType::ListIntType,
              serialized_arg_val.size(),
              serialized_arg_val);
          stack.emplace_back(c10::IValue());
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
              "Invalid serialized argument type found for List[Scalar] ",
              serialized_arg_type);
        }
      } else if (schema_arg_type->isSubtypeOf(
                     at::ListType::ofOptionalTensors())) {
        if (serialized_arg_type == "as_optional_tensors") {
          stack.emplace_back();
          dynamic_args.emplace_back(
              index,
              DynamicArgType::ListOptionalTensorType,
              serialized_arg_val.size(),
              serialized_arg_val);
        } else if (serialized_arg_type == "as_tensors") {
          stack.emplace_back();
          dynamic_args.emplace_back(
              index,
              DynamicArgType::ListTensorType,
              serialized_arg_val.size(),
              serialized_arg_val);
        } else {
          TORCH_CHECK(
              false,
              "Invalid serialized type found for argument of type `Tensor?[]`",
              serialized_arg_type);
        }
      } else if (schema_arg_type->isSubtypeOf(at::ListType::ofStrings())) {
        TORCH_CHECK(serialized_arg_type == "as_strings");
        std::vector<std::string> ret;
        for (const auto& arg : serialized_arg_val) {
          ret.push_back(arg.get<std::string>());
        }
        stack.emplace_back(ret);
      } else {
        TORCH_CHECK(false, "NYI: Unsupported list type ", serialized_arg_type);
      }
      break;
    }
    case c10::TypeKind::OptionalType: {
      auto inner_type =
          schema_arg_type->castRaw<at::OptionalType>()->getElementType();

      if (serialized_arg_type == "as_none") {
        stack.emplace_back(c10::nullopt);
        if (inner_type->kind() == c10::TypeKind::TensorType) {
          // Tensor is None
          dynamic_args.emplace_back(
              index, DynamicArgType::TensorType, 0, serialized_arg_val);
        } else if (
            inner_type->kind() == c10::TypeKind::IntType ||
            inner_type->kind() == c10::TypeKind::SymIntType) {
          // Int or SymInt is None
          dynamic_args.emplace_back(
              index, DynamicArgType::IntType, 0, serialized_arg_val);
        } else if (
            inner_type->kind() == c10::TypeKind::ListType &&
            schema_arg_type->isSubtypeOf(at::ListType::ofTensors())) {
          // List[Tensor] is None
          dynamic_args.emplace_back(
              index, DynamicArgType::ListTensorType, 0, serialized_arg_val);
        } else if (
            inner_type->kind() == c10::TypeKind::ListType &&
            schema_arg_type->isSubtypeOf(at::ListType::ofSymInts())) {
          // List[SymInt] is None
          dynamic_args.emplace_back(
              index, DynamicArgType::ListIntType, 0, serialized_arg_val);
        }
      } else {
        prefill_stack_with_static_arguments(
            index, inner_type, serialized_arg, op_kernel);
      }
      break;
    }
    default:
      TORCH_CHECK(false, "NYI: Unsupported input type ", serialized_arg_type);
  }
}

// Populates op_kernel.stack_, op_kernel.dynamic_args_
void OSSProxyExecutor::get_input_info_from_serialized(
    const std::vector<c10::Argument>& schema_args,
    json serialized_node,
    OpKernel& op_kernel) {
  int index = 0;
  for (const auto& named_argument : serialized_node["inputs"]) {
    const auto& arg = named_argument["arg"];
    auto& schema_arg = schema_args[index];
    prefill_stack_with_static_arguments(
        index++, schema_arg.real_type(), arg, op_kernel);
  }
}

// Populates op_kernel.outputs_
void OSSProxyExecutor::get_output_info_from_serialized(
    const std::vector<c10::Argument>& schema_returns,
    json serialized_node,
    OpKernel& op_kernel) {
  std::vector<DynamicArg>& outputs = op_kernel.outputs_;

  TORCH_CHECK(
      schema_returns.size() == serialized_node["outputs"].size(),
      "Serialized node doesn't match op's schema outputs.");

  size_t output_index = 0;
  for (const auto& serialized_output : serialized_node["outputs"]) {
    TORCH_CHECK(serialized_output.size() == 1);
    std::string serialized_output_type = serialized_output.begin().key();
    auto& serialized_output_val = serialized_output.begin().value();

    auto& schema_return = schema_returns[output_index];
    at::TypePtr schema_return_type = schema_return.real_type();

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
            false,
            "NYI: Unsupported return type ",
            schema_return_type->repr_str());
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
  TORCH_CHECK(json_file.is_open());

  // Parse file into a json object
  json json_obj;
  json_file >> json_obj;

  // Access data
  for (auto const& serialized_extern_node : json_obj["nodes"]) {
    auto const& serialized_node = serialized_extern_node["node"];

    const std::string& target = serialized_node["target"];

    std::string opName;
    std::string overloadName;
    size_t pos = target.find(".");
    if (pos == std::string::npos) {
      opName = target;
      overloadName = "";
    } else {
      // There should be no more periods
      size_t pos2 = target.find(".", pos);
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

    OpKernel op_kernel(target, op_handle);
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
  OpKernel& op_kernel = op_kernels_[extern_node_index];

  std::vector<c10::IValue> stack = op_kernel.stack_;
  auto& dynamic_args = op_kernel.dynamic_args_;

  int tensor_id = 0;
  int int_id = 0;
  for (size_t i = 0; i < dynamic_args.size(); i++) {
    int arg_index = dynamic_args[i].arg_index;
    DynamicArgType dynamic_arg_type = dynamic_args[i].arg_type;
    int length = dynamic_args[i].length;

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
        auto& serialized_arg_val = dynamic_args[i].serialized_arg_val;

        for (const auto& arg : serialized_arg_val) {
          std::string arg_type = arg.begin().key();
          if (arg_type == "as_tensor") {
            at::Tensor* tensor = tensor_handle_to_tensor_pointer(
                flatten_tensor_args[tensor_id++]);
            optional_tensor_list.emplace_back(*tensor);
          } else if (arg_type == "as_none") {
            optional_tensor_list.emplace_back(c10::nullopt);
          }
        }
        stack[arg_index] = optional_tensor_list;
        break;
      }
      case DynamicArgType::ListIntType: {
        std::vector<int64_t> vals;
        for (int j = 0; j < length; j++) {
          vals.push_back(flatten_int_args[int_id++]);
        }
        stack[arg_index] = vals;
        break;
      }
      default:
        TORCH_CHECK(
            false, "NYI: Unsupported dynamic arg type: ", dynamic_arg_type);
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
      for (size_t i = 0; i < tensors.size(); ++i) {
        at::Tensor* tensor =
            tensor_handle_to_tensor_pointer(flatten_tensor_args[tensor_id++]);
        *tensor = tensors[i];
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

} // namespace aot_inductor
} // namespace torch
