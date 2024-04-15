#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/utils/python_dispatch.h>

#include <string>

namespace torch::inductor {

struct AOTIKernelState {
  std::shared_ptr<AOTIModelContainerRunner> kernel_runner_;
  std::vector<TensorCheck> tensor_checks_;
};

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
  c10::Device device_;
  // The Python interpreter to get OpOverload object with the given op_name and
  // op_overload_name.
  c10::impl::PyInterpreter* pyinterpreter_;

  std::
      unordered_map<AOTIKernelMetaInfo, AOTIKernelState, AOTIKernelMetaInfoHash>
          aoti_kernel_cache_;

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
      const c10::DispatchKeySet& keyset,
      const torch::jit::Stack* stack,
      AOTIKernelState& kernel_state);
  void cache_miss(
      const c10::OperatorHandle& op,
      const c10::DispatchKeySet& keyset,
      torch::jit::Stack* stack);
  void cache_hit(
      const AOTIKernelState& kernel_state,
      const c10::OperatorHandle& op,
      const c10::DispatchKeySet& keyset,
      torch::jit::Stack* stack);
  // Invoke python utility function on the Inductor side to produce AOTI kernel
  // for the given operation.
  //   Inductor utility function -
  //   torch._inductor.utils.aoti_compile_with_persistent_cache
  std::string produce_aoti_kernel_lib(
      const c10::OperatorHandle& op,
      const c10::DispatchKeySet& keyset,
      const torch::jit::Stack* stack);
  // Invoke python utility function on the Inductor side to load AOTI kernel for
  // the given operation.
  //   Inductor utility function - torch._inductor.utils.load_aoti_eager_cache
  void init_aoti_kernel_cache();
  // Abstract the meta information of each tensor for the given operation. The
  // meta infomation will be used for cache lookup as the key.
  AOTIKernelMetaInfo get_inputs_meta_info(const std::vector<at::Tensor>&);
  // Load the AOTIModelContainerRunner object from the given file path.
  std::shared_ptr<AOTIModelContainerRunner> load_aoti_model_runner(
      const std::string&);
};

} // namespace torch::inductor
#endif
