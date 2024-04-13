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
enum class IValueType : uint8_t {
  Tensor,
  TensorList,
  OptionalTensorList,
  Scalar,
  Invalid,
};

static bool HandleTensor(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  inputs.push_back(ivalue.toTensor());
  return true;
}

static bool HandleTensorList(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  for (const auto& item : ivalue.toListRef()) {
    if (!item.isNone()) {
      inputs.push_back(item.toTensor());
    }
  }
  return true;
}

static bool HandleOptionalTensorList(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  return HandleTensorList(ivalue, device, inputs);
}

static bool HandleScalar(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  inputs.push_back(at::scalar_tensor(
      ivalue.toScalar(),
      c10::TensorOptions().device(device).dtype(ivalue.toScalar().type())));
  return true;
}

typedef bool (*HandlerFunc)(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs);
std::unordered_map<IValueType, HandlerFunc> handlers_ = {
    {IValueType::Tensor, &HandleTensor},
    {IValueType::TensorList, &HandleTensorList},
    {IValueType::OptionalTensorList, &HandleOptionalTensorList},
    {IValueType::Scalar, &HandleScalar}};

bool HandleIValue(
    const c10::IValue& ivalue,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  IValueType ivalue_type = IValueType::Invalid;
  if (ivalue.isTensor()) {
    ivalue_type = IValueType::Tensor;
  } else if (ivalue.isTensorList()) {
    ivalue_type = IValueType::TensorList;
  } else if (ivalue.isOptionalTensorList()) {
    ivalue_type = IValueType::OptionalTensorList;
  } else if (ivalue.isScalar()) {
    ivalue_type = IValueType::Scalar;
  }

  auto it = handlers_.find(ivalue_type);
  if (it != handlers_.end()) {
    return it->second(ivalue, device, inputs);
  }

  // Handle unsupported types or add a default handler
  return false;
}

bool unpackTensors(
    const torch::jit::Stack& stack,
    const c10::Device& device,
    std::vector<at::Tensor>& inputs) {
  for (const auto& ivalue : stack) {
    if (!HandleIValue(ivalue, device, inputs)) {
      return false;
    }
  }
  return true;
}

} // namespace

AOTIPythonKernelHolder::AOTIPythonKernelHolder(
    py::object func,
    c10::DispatchKey dispatch_key,
    c10::string_view ns,
    c10::string_view op_name,
    c10::string_view op_overload_name,
    bool is_symbolic)
    : python_kernel_holder_(func, dispatch_key),
      dispatch_key_(dispatch_key),
      ns_(std::string(ns)),
      op_name_(std::string(op_name)),
      op_overload_name_(std::string(op_overload_name)),
      is_symbolic_(is_symbolic),
      has_fall_back_(func.ptr() != Py_None),
      device_opt_(c10::Device(c10::dispatchKeyToDeviceType(dispatch_key_), 0)),
      pyinterpreter_(getPyInterpreter()) {
  (void)is_symbolic_; // Suppress unused variable warning
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
  // Always return false not and will add cache lookup later.
  return false;
}

void AOTIPythonKernelHolder::cache_hit(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  TORCH_CHECK(has_fall_back_, "AOTI kernel failed to run and no fall back");
  python_kernel_holder_(op, keyset, stack);
}

void AOTIPythonKernelHolder::cache_miss(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  auto device_type = c10::dispatchKeyToDeviceType(dispatch_key_);
  TORCH_CHECK(
      (device_type == c10::DeviceType::CPU) ||
          (device_type == c10::DeviceType::CUDA),
      "Unsupported device type");

  auto kernel_lib_path = produce_aoti_kernel_lib(op, keyset, stack);
  if (kernel_lib_path.empty()) {
    TORCH_CHECK(
        has_fall_back_,
        "Failed to produce kernel libarary by using AOTI and no fall back");
    python_kernel_holder_(op, keyset, stack);
    return;
  }

  auto device_index = 0; // TODO: Get device index from other tensors.
  auto device = c10::Device(device_type, device_index);

  std::shared_ptr<AOTIModelContainerRunner> kernel = nullptr;
  if (device_type == c10::DeviceType::CPU) {
    kernel = std::make_shared<AOTIModelContainerRunnerCpu>(kernel_lib_path);
  } else {
#ifdef USE_CUDA
    kernel = std::make_shared<AOTIModelContainerRunnerCuda>(kernel_lib_path);
#else
    TORCH_CHECK(false, "Unsupported CUDA device type");
#endif
  }

  std::vector<at::Tensor> inputs;
  if (unpackTensors(*stack, device, inputs)) {
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
}

std::string AOTIPythonKernelHolder::produce_aoti_kernel_lib(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  auto arguments = torch::jit::last(*stack, op.schema().arguments().size());
  std::string kernel_lib_path("");

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

  if (op_py_func.ptr() == nullptr || op_py_func.ptr() == Py_None) {
    TORCH_WARN(
        "Failed to get python operation. Operator Name is ",
        op.operator_name().name,
        ", Overload Name is ",
        overload_name);
    return kernel_lib_path;
  }

  py::handle aot_compile_function =
      py::module::import("torch._export").attr("aot_compile");
  if (aot_compile_function.ptr() == nullptr ||
      aot_compile_function.ptr() == Py_None) {
    TORCH_WARN("Failed to import - torch._export.aot_compile");
    return kernel_lib_path;
  }

  // Pass the python operation to the AOT Inductor to generate the kernel
  // library.
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments.vec());
  auto result = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
      aot_compile_function.ptr(),
      op_py_func.ptr(),
      args_kwargs.first.ptr(),
      args_kwargs.second.ptr(),
      nullptr));
  if (result.ptr() != nullptr && result.ptr() != Py_None) {
    kernel_lib_path = py::cast<std::string>(result);
    if (kernel_lib_path.empty()) {
      TORCH_WARN(
          "Kernel library is not generated by AOTI for ",
          c10::DeviceTypeName(device_opt_.value().type()),
          ". Operator Name is ",
          op.operator_name().name,
          ", Overload Name is ",
          overload_name);
    }
  } else {
    TORCH_WARN(
        "AOTI kernel library is not generated for ",
        c10::DeviceTypeName(device_opt_.value().type()),
        ". Operator Name is ",
        op.operator_name().name,
        ", Overload Name is ",
        overload_name);
  }

  return kernel_lib_path;
}

} // namespace torch::inductor
#endif
