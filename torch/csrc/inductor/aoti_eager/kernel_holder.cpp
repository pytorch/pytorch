#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/kernel_holder.h>

#include <ATen/ATen.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/csrc/jit/frontend/function_schema_parser.h>

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
      *argument.real_type() == *c10::getTypePtr<c10::optional<at::Tensor>>()) {
    // ivalue is c10::optional<at::Tensor>
    unpack_optional_tensor_ivalue(ivalue, device, inputs);
  } else {
    // Unsupport IValue type.
    return false;
  }

  return true;
}

bool unpack_tensors(
    const std::vector<c10::Argument>& arguments,
    const torch::jit::Stack& stack,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  for (size_t idx = 0; idx < stack.size(); idx++) {
    if (!unpack_ivalue(arguments[idx], stack[idx], device, inputs)) {
      return false;
    }
  }

  return true;
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
}

void AOTIPythonKernelHolder::operator()(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  if (cache_lookup(op, keyset, stack)) {
    cache_hit(op, keyset, stack);
  } else {
    cache_miss(op, keyset, stack);
  }
}

bool AOTIPythonKernelHolder::cache_lookup(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  // TODO: Always return false now to implement cache_miss. Later, we will add
  // cache lookup and implement cache hit.
  return false;
}

void AOTIPythonKernelHolder::cache_hit(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(false);
}

void AOTIPythonKernelHolder::cache_miss(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
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

  std::vector<at::Tensor> inputs;
  TORCH_INTERNAL_ASSERT(
      unpack_tensors(op.schema().arguments(), *stack, device_, inputs),
      "Failed to unpack tensors for the stack to run the AOTI kernel.");
  auto outputs = kernel->run(inputs);
  if (outputs.size() > 0) {
    torch::jit::drop(*stack, op.schema().arguments().size());
    // TODO: Get the output type of this operation and then convert to the
    // output type.
    for (auto& output : outputs) {
      torch::jit::push(*stack, std::move(output));
    }
  }
}

std::string AOTIPythonKernelHolder::produce_aoti_kernel_lib(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  auto arguments = torch::jit::last(*stack, op.schema().arguments().size());

  py::gil_scoped_acquire gil;

  // Get the corresponding python operation for the current operator and the
  // python operation will pass to the AOT Inductor to generate the kernel
  // library.
  const auto& schema = op.schema();
  const auto& qualified_name = op.operator_name().name;
  const auto& overload_name =
      schema.overload_name().empty() ? "default" : schema.overload_name();
  auto pos = qualified_name.find("::");
  TORCH_INTERNAL_ASSERT(pos != std::string::npos, qualified_name);
  // Make me some null terminated strings
  std::string ns_str = qualified_name.substr(0, pos);
  const char* ns = ns_str.c_str();
  const char* func_name = qualified_name.c_str() + pos + strlen("::");
  py::handle op_py_func = op.getPythonOp(pyinterpreter_, [&]() -> PyObject* {
    py::handle torch_api_function =
        py::module::import("torch").attr("ops").attr(ns).attr(func_name);
    return torch_api_function.attr(overload_name.c_str()).ptr();
  });

  TORCH_INTERNAL_ASSERT(
      op_py_func.ptr() != nullptr && op_py_func.ptr() != Py_None,
      "Failed to get python operation. Operator Name is ",
      op.operator_name().name,
      ", Overload Name is ",
      overload_name);

  py::handle aot_compile_function =
      py::module::import("torch._export").attr("aot_compile");
  TORCH_INTERNAL_ASSERT(
      aot_compile_function.ptr() != nullptr &&
          aot_compile_function.ptr() != Py_None,
      "Failed to import - torch._export.aot_compile");

  // Pass the python operation to the AOT Inductor to generate the kernel
  // library.
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments.vec());
  auto result = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
      aot_compile_function.ptr(),
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
