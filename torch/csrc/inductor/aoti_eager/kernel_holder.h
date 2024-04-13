#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/utils/python_dispatch.h>

#include <string>

namespace torch::inductor {

// The AOTIPythonKernelHolder class uses the AOT Inductor to generate a kernel
// for a specified operation. To speed up this process, the generated kernel
// library is cached on disk. Detailed information from the input tensors is
// used as the key for caching the kernel library. On subsequent runs, these
// input tensors are used to search the cache. If a cache hit occurs, the cached
// kernel library is loaded and executed. If a cache miss occurs, the AOT
// Inductor is called again to generate the kernel library.
class AOTIPythonKernelHolder : public c10::OperatorKernel {
  // A PythonKernelHolder object that holds the fallback PyTorch kernel in case
  // the AOT Inductor fails to generate a kernel.
  torch::impl::dispatch::PythonKernelHolder python_kernel_holder_;
  // A DispatchKey object that represents the dispatch key for the kernel.
  c10::DispatchKey dispatch_key_;
  // Namespace of the kernel.
  std::string ns_;
  // Name of the operation the kernel performs.
  std::string op_name_;
  // Name of the overloaded operation the kernel performs.
  std::string op_overload_name_;
  // Produce kernel w/ dynamic shapes.
  bool is_symbolic_;
  // Has a fallback function or not.
  bool has_fall_back_;
  // The device on which the kernel is to be executed.
  c10::optional<c10::Device> device_opt_;
  // The Python interpreter to get OpOverload object with the given op_name and
  // op_overload_name.
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
  bool cache_lookup(
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
  std::string produce_aoti_kernel_lib(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);
};

} // namespace torch::inductor
#endif
