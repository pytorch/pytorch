#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/boxing/KernelFunction.h>

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/utils/pybind.h>

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
  // A DispatchKey object that represents the dispatch key for the kernel.
  c10::DispatchKey dispatch_key_;
  // Namespace of the kernel.
  std::string ns_;
  // Name of the operation the kernel performs.
  std::string op_name_with_overload_;
  // The device on which the kernel is to be executed.
  c10::Device device_;
  // The Python interpreter to get OpOverload object with the given op_name and
  // op_overload_name.
  c10::impl::PyInterpreter* pyinterpreter_;

 public:
  AOTIPythonKernelHolder(
      c10::DispatchKey dispatch_key,
      c10::string_view ns,
      c10::string_view op_name_with_overload);

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
