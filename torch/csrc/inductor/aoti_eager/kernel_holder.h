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

class AOTIPythonKernelHolder : public c10::OperatorKernel {
  torch::impl::dispatch::PythonKernelHolder python_kernel_holder_;
  c10::DispatchKey dispatch_key_;
  std::string ns_;
  std::string op_name_;
  std::string op_overload_name_;
  bool is_symbolic_;
  bool is_fall_back_;
  c10::Device device_;
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
  bool detect_cache(
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
  std::string produce_aoti_kernel_lib(
      const c10::OperatorHandle& op,
      const c10::DispatchKeySet& keyset,
      const torch::jit::Stack* stack);
  void init_aoti_kernel_cache();
  AOTIKernelMetaInfo get_inputs_meta_info(const std::vector<at::Tensor>&);
  std::shared_ptr<AOTIModelContainerRunner> load_aoti_model_runner(
      const std::string&);
};

} // namespace torch::inductor
#endif
