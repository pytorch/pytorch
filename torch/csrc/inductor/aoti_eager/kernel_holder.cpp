#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/kernel_holder.h>

#include <ATen/ATen.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

namespace torch::inductor {

namespace {

inline void unpack_tensor_ivalue(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  inputs.push_back(ivalue.toTensor());
}

inline void unpack_optional_tensor_ivalue(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  auto ivalue_opt_tensor = ivalue.toOptional<at::Tensor>();
  if (ivalue_opt_tensor.has_value()) {
    inputs.push_back(ivalue_opt_tensor.value());
  }
}

inline void unpack_tensor_list_ivalue(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  for (const auto& item : ivalue.toListRef()) {
    inputs.push_back(item.toTensor());
  }
}

inline void unpack_optional_tensor_list_ivalue(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  for (const auto& item : ivalue.toListRef()) {
    unpack_optional_tensor_ivalue(item, device, inputs);
  }
}

std::vector<at::Tensor> unpack_tensors(
    const std::vector<c10::Argument>& arguments,
    const torch::jit::Stack& stack,
    const c10::Device& device) {
  std::vector<at::Tensor> inputs;
  for (size_t idx = 0; idx < stack.size(); idx++) {
    const auto& ivalue = stack[idx];
    const auto& ivalue_arg = arguments[idx];
    if (ivalue.isTensor()) {
      unpack_tensor_ivalue(ivalue, device, inputs);
    } else if (ivalue.isTensorList()) {
      unpack_tensor_list_ivalue(ivalue, device, inputs);
    } else if (ivalue.isOptionalTensorList()) {
      unpack_optional_tensor_list_ivalue(ivalue, device, inputs);
    } else if (
        *ivalue_arg.real_type() ==
        *c10::getTypePtr<std::optional<at::Tensor>>()) {
      // ivalue is std::optional<at::Tensor>
      unpack_optional_tensor_ivalue(ivalue, device, inputs);
    }
  }
  return inputs;
}

// Find the first positional argument that isn't defaulted
bool is_default_value(
    const c10::Argument& argument,
    const c10::IValue& ivalue) {
  if (!argument.default_value().has_value()) {
    return false;
  }
  const auto& default_ivalue = *argument.default_value();
  if (default_ivalue != ivalue) {
    return false;
  }

  return true;
}

std::vector<ParameterMetadata> unpack_input_parameters(
    const std::vector<c10::Argument>& arguments,
    const torch::jit::Stack& stack) {
  std::vector<ParameterMetadata> inputs_metadata;
  // Represent the order of argument and skip default parameter
  int64_t arg_order = 0;
  for (size_t idx = 0; idx < stack.size(); idx++) {
    // By default, the parameter will not be cached if its value is the default
    // value.
    //   - produce_aoti_kernel_lib utilizes parseIValuesToPyArgsKwargs to get
    //   args and kwargs.
    //   - parseIValuesToPyArgsKwargs skips the parameter if its value is the
    //   default value.
    if (is_default_value(arguments[idx], stack[idx])) {
      continue;
    }

    if (stack[idx].isScalar()) {
      // Beyond c10::Scalar, the floating value and interger value are also
      // represented as Scalar.
      inputs_metadata.emplace_back(stack[idx].toScalar(), arg_order);
    } else if (stack[idx].isTensorList()) {
      // tensor list
      inputs_metadata.emplace_back(stack[idx].toTensorList().vec(), arg_order);
    } else if (stack[idx].isOptionalTensorList()) {
      // optional tensor list: std::vector<std::optional<at::Tensor>>
      std::vector<at::Tensor> tensor_list;
      for (const auto& item : stack[idx].toListRef()) {
        if (item.toOptional<at::Tensor>().has_value()) {
          tensor_list.push_back(item.toOptional<at::Tensor>().value());
        }
      }
      inputs_metadata.emplace_back(tensor_list, arg_order);
    } else if (
        *arguments[idx].real_type() ==
        *c10::getTypePtr<std::optional<at::Tensor>>()) {
      // optional tensor
      if (stack[idx].toOptional<at::Tensor>().has_value()) {
        inputs_metadata.emplace_back(
            stack[idx].toOptional<at::Tensor>().value(), arg_order);
      }
    } else if (stack[idx].isTensor()) {
      inputs_metadata.emplace_back(stack[idx].toTensor(), arg_order);
    } else if (stack[idx].isString()) {
      inputs_metadata.emplace_back(stack[idx].toStringRef(), arg_order);
    } else if (stack[idx].isBool()) {
      inputs_metadata.emplace_back(c10::Scalar(stack[idx].toBool()), arg_order);
    } else if (stack[idx].isDevice()) {
      inputs_metadata.emplace_back(stack[idx].toDevice(), arg_order);
    } else {
      TORCH_CHECK_NOT_IMPLEMENTED(
          false,
          "Not implemented for operations that contain a parameter which is ",
          "not one of the following types: at::Tensor, at::TensorList, ",
          "std::optional<at::Tensor>, std::vector<std::optional<at::Tensor>> and c10::Scalar.",
          "The input type is ",
          stack[idx].type()->str());
    }

    arg_order++;
  }

  return inputs_metadata;
}

} // namespace

AOTIPythonKernelHolder::AOTIPythonKernelHolder(
    c10::DispatchKey dispatch_key,
    std::string_view ns,
    std::string_view op_name_with_overload)
    : dispatch_key_(dispatch_key),
      ns_(std::string(ns)),
      op_name_with_overload_(std::string(op_name_with_overload)),
      device_(c10::dispatchKeyToDeviceType(dispatch_key_), 0),
      pyinterpreter_(getPyInterpreter()) {
  auto device_name = c10::DeviceTypeName(device_.type());
  auto registered_aoti_runner = getAOTIModelRunnerRegistry();
  TORCH_CHECK(
      device_.type() == c10::DeviceType::CUDA ||
          device_.type() == c10::DeviceType::CPU ||
          registered_aoti_runner.find(device_name) !=
              registered_aoti_runner.end(),
      "AOTI for eager does not support ",
      c10::DeviceTypeName(device_.type()),
      " now.");

  init_aoti_kernel_cache();
}

void AOTIPythonKernelHolder::operator()(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  AOTIKernelMetadata aoti_kernel_metadata;
  if (cache_lookup(op, keyset, stack, aoti_kernel_metadata)) {
    cache_hit(aoti_kernel_metadata, op, keyset, stack);
  } else {
    cache_miss(op, keyset, stack);
  }
}

bool AOTIPythonKernelHolder::cache_lookup(
    const c10::OperatorHandle& op,
    const c10::DispatchKeySet& keyset,
    const torch::jit::Stack* stack,
    AOTIKernelMetadata& aoti_kernel_metadata) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      op.schema().returns().size() == 1,
      "Not implemented for operations that return either multiple values or no value.");
  TORCH_CHECK_NOT_IMPLEMENTED(
      op.schema().returns()[0].type()->isSubtypeOf(c10::TensorType::get()),
      "Not implemented for operations that return a non-Tensor value.");

  auto inputs_metadata =
      unpack_input_parameters(op.schema().arguments(), *stack);
  for (const auto& aoti_kernel_cache : aoti_kernel_cache_) {
    if (aoti_kernel_cache.check(inputs_metadata)) {
      aoti_kernel_metadata = aoti_kernel_cache;
      return true;
    }
  }

  return false;
}

void AOTIPythonKernelHolder::cache_hit(
    const AOTIKernelMetadata& aoti_kernel_metadata,
    const c10::OperatorHandle& op,
    const c10::DispatchKeySet& keyset,
    torch::jit::Stack* stack) {
  auto inputs = unpack_tensors(op.schema().arguments(), *stack, device_);
  torch::jit::drop(*stack, op.schema().arguments().size());

  auto outputs = aoti_kernel_metadata.kernel_runner_->run(inputs);
  for (auto& output : outputs) {
    stack->emplace_back(output);
  }
}

void AOTIPythonKernelHolder::init_aoti_kernel_cache() {
  if (device_.type() == c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES) {
    return;
  }

  py::gil_scoped_acquire gil;

  py::handle load_aoti_eager_cache_function =
      py::module::import("torch._inductor.aoti_eager")
          .attr("load_aoti_eager_cache");
  TORCH_INTERNAL_ASSERT(
      load_aoti_eager_cache_function.ptr() != nullptr,
      "Failed to import - torch._inductor.aoti_eager.load_aoti_eager_cache");

  auto result = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
      load_aoti_eager_cache_function.ptr(),
      py::str(ns_).ptr(),
      py::str(op_name_with_overload_).ptr(),
      py::str(c10::DeviceTypeName(device_.type(), true)).ptr(),
      nullptr));
  TORCH_INTERNAL_ASSERT(
      result.ptr() != nullptr && result.ptr() != Py_None,
      "Failed to load AOTI kernel. Operator Name is ",
      op_name_with_overload_);

  auto build_tensor_metadata = [](const py::dict& metadata) -> TensorMetadata {
    // Access the fields of each metadata dict
    auto is_dynamic = metadata["is_dynamic"].cast<bool>();
    auto device_type = metadata["device_type"].cast<std::string>();
    auto device_index = metadata["device_index"].cast<int8_t>();
    auto data_type_obj = metadata["dtype"].cast<py::object>();
    TORCH_INTERNAL_ASSERT(THPDtype_Check(data_type_obj.ptr()));
    auto data_type =
        reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;
    auto sizes = metadata["sizes"].cast<std::vector<int64_t>>();
    auto strides = metadata["strides"].cast<std::vector<int64_t>>();
    auto requires_grad = metadata["requires_grad"].cast<bool>();
    auto dispatch_key_set_raw_repr =
        metadata["dispatch_key_set"].cast<uint64_t>();
    auto dispatch_key_set = c10::DispatchKeySet(
        c10::DispatchKeySet::RAW, dispatch_key_set_raw_repr);
    auto device = c10::Device(device_type);
    device.set_index(device_index);

    auto tensor_metadata = TensorMetadata(
        is_dynamic,
        data_type,
        device,
        dispatch_key_set,
        sizes,
        strides,
        requires_grad);

    // Build guard for tensor check
    torch::dynamo::LocalState state;
    state.overrideDispatchKeySet(dispatch_key_set);
    tensor_metadata.build_guard(state);

    return tensor_metadata;
  };

  TORCH_INTERNAL_ASSERT(py::isinstance<py::list>(result));
  auto kernel_info_list = result.cast<py::list>();
  for (auto kernel_info : kernel_info_list) {
    TORCH_INTERNAL_ASSERT(py::isinstance<py::dict>(kernel_info));
    auto item_dict = kernel_info.cast<py::dict>();

    // Access the kernel_path field
    auto kernel_path = item_dict["kernel_path"].cast<std::string>();

    // Access the meta_info list
    auto inputs_metadata = item_dict["meta_info"].cast<py::list>();

    std::vector<ParameterMetadata> parameter_metadata_list;
    // Loop over the meta_info list
    for (auto item_metadata : inputs_metadata) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(py::isinstance<py::dict>(item_metadata));
      auto metadata = item_metadata.cast<py::dict>();
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(metadata.contains("arg_order"));
      uint64_t arg_idx = metadata["arg_order"].cast<uint64_t>();
      bool is_scalar = metadata.contains("scalar_value");
      bool is_tensor_list = metadata.contains("tensor_list");
      bool is_string = metadata.contains("string_value");
      bool is_device = metadata.contains("device_type_value");
      bool is_dtype = metadata.contains("dtype_value");
      bool is_layout = metadata.contains("layout_value");

      if (is_tensor_list) {
        // Tensor List
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            py::isinstance<py::list>(metadata["tensor_list"]));
        auto tensor_list = metadata["tensor_list"].cast<py::list>();
        std::vector<TensorMetadata> test_list_metadata;
        for (auto item_tensor : tensor_list) {
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
              py::isinstance<py::dict>(item_tensor));
          auto metadata = item_tensor.cast<py::dict>();
          auto tensor_metadata = build_tensor_metadata(metadata);
          test_list_metadata.push_back(tensor_metadata);
        }
        parameter_metadata_list.emplace_back(test_list_metadata, arg_idx);
      } else if (is_scalar) {
        // Scalar
        auto metadata = item_metadata.cast<py::dict>();
        auto dtype_obj = metadata["dtype"].cast<py::object>();
        TORCH_INTERNAL_ASSERT(THPDtype_Check(dtype_obj.ptr()));
        auto dtype_value =
            reinterpret_cast<THPDtype*>(dtype_obj.ptr())->scalar_type;

        c10::Scalar scalar;
        if (c10::isFloatingType(dtype_value)) {
          scalar = metadata["scalar_value"].cast<double>();
        } else if (c10::isIntegralType(dtype_value, false)) {
          scalar = metadata["scalar_value"].cast<int64_t>();
        } else if (dtype_value == c10::kBool) {
          scalar = metadata["scalar_value"].cast<bool>();
        } else {
          TORCH_CHECK_NOT_IMPLEMENTED(
              false,
              "Not implemented for operations that contain a scalar parameter which is ",
              dtype_value);
        }

        parameter_metadata_list.emplace_back(c10::Scalar(scalar), arg_idx);
      } else if (is_string) {
        // String
        auto metadata = item_metadata.cast<py::dict>();
        auto str_value = metadata["string_value"].cast<std::string>();
        parameter_metadata_list.emplace_back(str_value, arg_idx);
      } else if (is_dtype) {
        // Dtype
        auto metadata = item_metadata.cast<py::dict>();
        auto dtype_value_obj = metadata["dtype_value"].cast<py::object>();
        TORCH_INTERNAL_ASSERT(THPDtype_Check(dtype_value_obj.ptr()));
        auto dtype_value =
            reinterpret_cast<THPDtype*>(dtype_value_obj.ptr())->scalar_type;
        parameter_metadata_list.emplace_back(
            c10::Scalar(static_cast<int>(dtype_value)), arg_idx);
      } else if (is_device) {
        // Device
        auto metadata = item_metadata.cast<py::dict>();
        auto device_type_value =
            metadata["device_type_value"].cast<std::string>();
        auto device = c10::Device(device_type_value);
        if (metadata["device_index_value"].ptr() != Py_None) {
          auto device_index_value =
              metadata["device_index_value"].cast<c10::DeviceIndex>();
          device.set_index(device_index_value);
        }
        parameter_metadata_list.emplace_back(device, arg_idx);
      } else if (is_layout) {
        auto metadata = item_metadata.cast<py::dict>();
        auto layout_value_obj = metadata["layout_value"].cast<py::object>();
        TORCH_INTERNAL_ASSERT(THPLayout_Check(layout_value_obj.ptr()));
        auto layout_value =
            reinterpret_cast<THPLayout*>(layout_value_obj.ptr())->layout;
        parameter_metadata_list.emplace_back(
            c10::Scalar(static_cast<int>(layout_value)), arg_idx);
      } else {
        // Tensor
        auto metadata = item_metadata.cast<py::dict>();
        auto tensor_metadata = build_tensor_metadata(metadata);
        parameter_metadata_list.emplace_back(tensor_metadata, arg_idx);
      }
    }

    AOTIKernelMetadata aoti_kernel_metadata;
    aoti_kernel_metadata.parameter_metadata_list_ =
        std::move(parameter_metadata_list);
    aoti_kernel_metadata.kernel_runner_ = load_aoti_model_runner(kernel_path);
    aoti_kernel_cache_.push_back(aoti_kernel_metadata);
  }
}

std::shared_ptr<AOTIModelContainerRunner> AOTIPythonKernelHolder::
    load_aoti_model_runner(const std::string& so_path) {
  auto device_name = c10::DeviceTypeName(device_.type());
  auto registered_aoti_runner = getAOTIModelRunnerRegistry();
  TORCH_CHECK(
      device_.type() == c10::DeviceType::CUDA ||
          device_.type() == c10::DeviceType::CPU ||
          registered_aoti_runner.find(device_name) !=
              registered_aoti_runner.end(),
      "AOTI for eager does not support ",
      c10::DeviceTypeName(device_.type()),
      " now.");
  if (device_.type() == c10::DeviceType::CUDA) {
#ifdef USE_CUDA
    return std::make_shared<AOTIModelContainerRunnerCuda>(so_path);
#else
    return nullptr;
#endif
  } else if (device_.type() == c10::DeviceType::CPU) {
    return std::make_shared<AOTIModelContainerRunnerCpu>(so_path);
  } else {
    auto aoti_model_runer_fn = registered_aoti_runner[device_name];
    return aoti_model_runer_fn(so_path, 1, device_name, "");
  }
}

void AOTIPythonKernelHolder::cache_miss(
    const c10::OperatorHandle& op,
    const c10::DispatchKeySet& keyset,
    torch::jit::Stack* stack) {
  auto kernel_lib_path = produce_aoti_kernel_lib(op, keyset, stack);
  std::shared_ptr<AOTIModelContainerRunner> kernel = nullptr;
  kernel = load_aoti_model_runner(kernel_lib_path);
  TORCH_INTERNAL_ASSERT(
      kernel != nullptr,
      "Unsupported device: ",
      c10::DeviceTypeName(device_.type()));
  auto inputs = unpack_tensors(op.schema().arguments(), *stack, device_);
  auto outputs = kernel->run(inputs);
  torch::jit::drop(*stack, op.schema().arguments().size());
  // TODO: Get the output type of this operation and then convert to the
  // output type.
  for (auto& output : outputs) {
    torch::jit::push(*stack, std::move(output));
  }
}

std::string AOTIPythonKernelHolder::produce_aoti_kernel_lib(
    const c10::OperatorHandle& op,
    const c10::DispatchKeySet& keyset,
    const torch::jit::Stack* stack) {
  auto arguments = torch::jit::last(*stack, op.schema().arguments().size());

  const auto& schema = op.schema();
  const auto& qualified_name = op.operator_name().name;
  const auto& overload_name =
      schema.overload_name().empty() ? "default" : schema.overload_name();
  auto pos = qualified_name.find("::");
  TORCH_INTERNAL_ASSERT(pos != std::string::npos, qualified_name);
  std::string ns_str(
      qualified_name.begin(),
      qualified_name.begin() + static_cast<ptrdiff_t>(pos));
  std::string func_name(
      qualified_name.begin() + static_cast<ptrdiff_t>(pos + strlen("::")),
      qualified_name.end());

  py::gil_scoped_acquire gil;
  py::handle op_py_func = op.getPythonOp(pyinterpreter_, [&]() -> PyObject* {
    py::handle torch_api_function = py::module::import("torch")
                                        .attr("ops")
                                        .attr(ns_str.c_str())
                                        .attr(func_name.c_str());
    if (overload_name.empty()) {
      return torch_api_function.attr("default").ptr();
    } else {
      return torch_api_function.attr(overload_name.c_str()).ptr();
    }
  });

  TORCH_INTERNAL_ASSERT(
      op_py_func.ptr() != nullptr && op_py_func.ptr() != Py_None,
      "Failed to get python operation. Operator Name is ",
      op.operator_name().name,
      ", Overload Name is ",
      overload_name);

  py::handle aot_compile_function =
      py::module::import("torch._inductor.aoti_eager")
          .attr("aoti_compile_with_persistent_cache");
  TORCH_INTERNAL_ASSERT(
      aot_compile_function.ptr() != nullptr &&
          aot_compile_function.ptr() != Py_None,
      "Failed to import - torch._inductor.aoti_eager.aoti_compile_with_persistent_cache");

  // Pass the python operation to the AOT Inductor to generate the kernel
  // library.
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments.vec());
  auto result = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
      aot_compile_function.ptr(),
      py::str(ns_str).ptr(),
      py::str(op_name_with_overload_).ptr(),
      py::str(c10::DeviceTypeName(device_.type(), true)).ptr(),
      py::bool_(false).ptr(),
      op_py_func.ptr(),
      args_kwargs.first.ptr(),
      args_kwargs.second.ptr(),
      nullptr));
  TORCH_INTERNAL_ASSERT(result.ptr() != nullptr && result.ptr() != Py_None);

  auto kernel_lib_path = py::cast<std::string>(result);
  TORCH_CHECK(
      !kernel_lib_path.empty(),
      "Failed to produce kernel libarary by using AOTI for ",
      c10::DeviceTypeName(device_.type()),
      ". Operator Name is ",
      op.operator_name().name,
      ", Overload Name is ",
      op.schema().overload_name());

  return kernel_lib_path;
}

} // namespace torch::inductor
#endif
