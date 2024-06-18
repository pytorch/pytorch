#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/kernel_holder.h>

#include <ATen/ATen.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
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

inline void unpack_scalar_ivalue(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  inputs.push_back(at::scalar_tensor(
      ivalue.toScalar(),
      c10::TensorOptions().device(device).dtype(ivalue.toScalar().type())));
}

bool unpack_ivalue(
    const c10::Argument& argument,
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  if (ivalue.isTensor()) {
    unpack_tensor_ivalue(ivalue, device, inputs);
  } else if (ivalue.isTensorList()) {
    unpack_tensor_list_ivalue(ivalue, device, inputs);
  } else if (ivalue.isOptionalTensorList()) {
    unpack_optional_tensor_list_ivalue(ivalue, device, inputs);
  } else if (ivalue.isScalar()) {
    // ivalue is scalar
    unpack_scalar_ivalue(ivalue, device, inputs);
  } else if (
      *argument.real_type() == *c10::getTypePtr<std::optional<at::Tensor>>()) {
    // ivalue is std::optional<at::Tensor>
    unpack_optional_tensor_ivalue(ivalue, device, inputs);
  } else {
    // Unsupport IValue type.
    return false;
  }

  return true;
}

std::vector<at::Tensor> unpack_tensors(
    const std::vector<c10::Argument>& arguments,
    const torch::jit::Stack& stack,
    const c10::Device& device) {
  std::vector<at::Tensor> inputs;
  for (size_t idx = 0; idx < stack.size(); idx++) {
    auto ivalue = stack[idx];
    auto ivalue_arg = arguments[idx];
    if (ivalue.isTensor()) {
      unpack_tensor_ivalue(ivalue, device, inputs);
    } else if (ivalue.isTensorList()) {
      unpack_tensor_list_ivalue(ivalue, device, inputs);
    } else if (ivalue.isOptionalTensorList()) {
      unpack_optional_tensor_list_ivalue(ivalue, device, inputs);
    } else if (
        *ivalue_arg.real_type() ==
        *c10::getTypePtr<c10::optional<at::Tensor>>()) {
      // ivalue is c10::optional<at::Tensor>
      unpack_optional_tensor_ivalue(ivalue, device, inputs);
    }
  }
  return inputs;
}

std::vector<ParameterMetadata> unpack_input_parameters(
    const std::vector<c10::Argument>& arguments,
    const torch::jit::Stack& stack) {
  std::vector<ParameterMetadata> inputs_metadata;
  for (size_t idx = 0; idx < stack.size(); idx++) {
    if (stack[idx].isScalar()) {
      // scalar
      inputs_metadata.push_back(ParameterMetadata(stack[idx].toScalar(), idx));
    } else if (stack[idx].isTensorList()) {
      // tensor list
      inputs_metadata.push_back(
          ParameterMetadata(stack[idx].toTensorList().vec(), idx));
    } else if (stack[idx].isOptionalTensorList()) {
      // optional tensor list: std::vector<std::optional<at::Tensor>>
      std::vector<at::Tensor> tensor_list;
      for (const auto& item : stack[idx].toListRef()) {
        if (item.toOptional<at::Tensor>().has_value()) {
          tensor_list.push_back(item.toOptional<at::Tensor>().value());
        }
      }
      inputs_metadata.push_back(ParameterMetadata(tensor_list, idx));
    } else if (
        *arguments[idx].real_type() ==
        *c10::getTypePtr<std::optional<at::Tensor>>()) {
      // optional tensor
      if (stack[idx].toOptional<at::Tensor>().has_value()) {
        inputs_metadata.push_back(ParameterMetadata(
            stack[idx].toOptional<at::Tensor>().value(), idx));
      }
    } else if (stack[idx].isTensor()) {
      inputs_metadata.push_back(ParameterMetadata(stack[idx].toTensor(), idx));
    } else if (stack[idx].isString()) {
      inputs_metadata.push_back(
          ParameterMetadata(stack[idx].toStringRef(), idx));
    } else {
      TORCH_CHECK_NOT_IMPLEMENTED(
          false,
          "Not implemented for operations that contain a parameter which is ",
          "not one of the following types: at::Tensor, at::TensorList, ",
          "std::optional<at::Tensor>, std::vector<std::optional<at::Tensor>> and c10::Scalar.");
    }
  }

  return inputs_metadata;
}

} // namespace

AOTIPythonKernelHolder::AOTIPythonKernelHolder(
    c10::DispatchKey dispatch_key,
    c10::string_view ns,
    c10::string_view op_name_with_overload)
    : dispatch_key_(dispatch_key),
      ns_(std::string(ns)),
      op_name_with_overload_(std::string(op_name_with_overload)),
      device_(c10::dispatchKeyToDeviceType(dispatch_key_), 0),
      pyinterpreter_(getPyInterpreter()) {
  TORCH_CHECK(
      (device_.type() == c10::DeviceType::CPU) ||
          (device_.type() == c10::DeviceType::CUDA),
      "Unsupported device type");
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
    stack->push_back(output);
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
        parameter_metadata_list.push_back(
            ParameterMetadata(test_list_metadata, arg_idx));
      } else if (is_scalar) {
        // Scalar
        auto metadata = item_metadata.cast<py::dict>();
        // Always cast scalar value to double to simplify the comparison
        auto scalar_value = metadata["scalar_value"].cast<double>();
        parameter_metadata_list.push_back(
            ParameterMetadata(c10::Scalar(scalar_value), arg_idx));
      } else if (is_string) {
        // String
        auto metadata = item_metadata.cast<py::dict>();
        auto str_value = metadata["string_value"].cast<std::string>();
        parameter_metadata_list.push_back(
            ParameterMetadata(str_value, arg_idx));
      } else {
        // Tensor
        auto metadata = item_metadata.cast<py::dict>();
        auto tensor_metadata = build_tensor_metadata(metadata);
        parameter_metadata_list.push_back(
            ParameterMetadata(tensor_metadata, arg_idx));
      }
    }

    AOTIKernelMetadata aoti_kernel_metadata;
    aoti_kernel_metadata.parameter_metadata_list_ = parameter_metadata_list;
    aoti_kernel_metadata.kernel_runner_ = load_aoti_model_runner(kernel_path);
    aoti_kernel_cache_.push_back(aoti_kernel_metadata);
  }
}

std::shared_ptr<AOTIModelContainerRunner> AOTIPythonKernelHolder::
    load_aoti_model_runner(const std::string& so_path) {
  TORCH_CHECK(
      device_.type() == c10::DeviceType::CUDA ||
          device_.type() == c10::DeviceType::CPU,
      "AOTI for eager does not support ",
      c10::DeviceTypeName(device_.type()),
      " now.");
  if (device_.type() == c10::DeviceType::CUDA) {
#ifdef USE_CUDA
    return std::make_shared<AOTIModelContainerRunnerCpu>(so_path);
#else
    return nullptr;
#endif
  } else {
    return std::make_shared<AOTIModelContainerRunnerCpu>(so_path);
  }
}

void AOTIPythonKernelHolder::cache_miss(
    const c10::OperatorHandle& op,
    const c10::DispatchKeySet& keyset,
    torch::jit::Stack* stack) {
  auto kernel_lib_path = produce_aoti_kernel_lib(op, keyset, stack);
  std::shared_ptr<AOTIModelContainerRunner> kernel = nullptr;
  // TODO: To enable the plugin mechanism to allow registration for other
  // backends
  if (device_.type() == c10::DeviceType::CPU) {
    kernel = std::make_shared<AOTIModelContainerRunnerCpu>(kernel_lib_path);
  } else {
#ifdef USE_CUDA
    kernel = std::make_shared<AOTIModelContainerRunnerCuda>(kernel_lib_path);
#else
    TORCH_CHECK(false, "Unsupported CUDA device type");
#endif
  }

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
  std::string ns_str(qualified_name.begin(), qualified_name.begin() + pos);
  std::string func_name(
      qualified_name.begin() + pos + strlen("::"), qualified_name.end());

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
