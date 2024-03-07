#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/aoti_kernel_holder.h>

#include <ATen/ATen.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/inductor/aoti_eager/aoti_kernel_holder.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>

namespace torch::inductor {

AOTIPythonKernelHolder::AOTIPythonKernelHolder(
    py::object func,
    c10::DispatchKey dispatch_key,
    c10::string_view op_name,
    bool is_symbolic)
    : python_kernel_holder_(func, dispatch_key),
      dispatch_key_(dispatch_key),
      op_name_(std::string(op_name)),
      is_symbolic_(is_symbolic),
      device_opt_(c10::nullopt) {
  // TODO: To provide a registration mechanim to avoid adding such if-else block
  if (dispatch_key_ == c10::DispatchKey::CUDA) {
    device_opt_ = c10::Device(c10::DeviceType::CUDA, 0);
  } else if (dispatch_key_ == c10::DispatchKey::XPU) {
    device_opt_ = c10::Device(c10::DeviceType::XPU, 0);
  } else {
    device_opt_ = c10::Device(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
  }

  canonicalizeOpName();
}

void AOTIPythonKernelHolder::operator()(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack) {
  // TODO(Eikan): By now, we always fallback to python python_kernel_holder_ to
  // simulate the cass miss
  //  behavior. We will add cache lookup later when the design is mature.
  python_kernel_holder_(op, keyset, stack);
}

void AOTIPythonKernelHolder::canonicalizeOpName() {
  // Canonicalize the op_name as a valid directory name
  std::replace(op_name_.begin(), op_name_.end(), '.', '_');
  const std::string to_remove = "aten::";
  size_t start_pos = op_name_.find(to_remove);
  if (start_pos != std::string::npos) {
    op_name_.replace(start_pos, to_remove.length(), "");
  }
}

} // namespace torch::inductor
#endif