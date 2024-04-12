#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/utils/python_dispatch.h>

#include <string>

namespace torch::inductor {

class AOTIPythonKernelHolder : public c10::OperatorKernel {
  torch::impl::dispatch::PythonKernelHolder python_kernel_holder_;
  c10::DispatchKey dispatch_key_;
  std::string ns_;
  std::string op_name_;
  std::string op_overload_name_;
  bool is_symbolic_;
  bool is_fall_back_;
  c10::optional<c10::Device> device_opt_;
  c10::impl::PyInterpreter* pyinterpreter_;

 public:
  AOTIPythonKernelHolder(
      py::object fall_back_func,
      c10::DispatchKey dispatch_key,
      c10::string_view ns,
      c10::string_view op_name,
      c10::string_view op_overload_name,
      bool is_symbolic = false);

  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);

 private:
  bool detect_cache(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);
  void cache_miss(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);
  void cache_hit(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);
  std::string produce_aot_kernel_lib(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);
};

} // namespace torch::inductor
#endif
