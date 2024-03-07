#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/inductor/aoti_eager/aoti_kernel_meta_info.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/utils/python_dispatch.h>

#include <string>

namespace torch::inductor {

class AOTIPythonKernelHolder : public c10::OperatorKernel {
  torch::impl::dispatch::PythonKernelHolder python_kernel_holder_;
  c10::DispatchKey dispatch_key_;
  std::string op_name_;
  bool is_symbolic_;
  c10::optional<c10::Device> device_opt_;

  std::unordered_map<
      AOTIKernelMetaInfo,
      std::shared_ptr<AOTIEagerKernelRunner>,
      AOTIKernelMetaInfoHash>
      aoti_kernel_cache_;

 public:
  AOTIPythonKernelHolder(
      py::object func,
      c10::DispatchKey dispatch_key,
      c10::string_view op_name,
      bool is_dynamic = false);

  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);

 private:
  void initAOTIKernelCache();
  void canonicalizeOpName();
  bool unpackTensors(
      const torch::jit::Stack& stack,
      std::vector<at::Tensor>& inputs);
  AOTIKernelMetaInfo getInputsMetaInfo(const std::vector<at::Tensor>&);
  std::shared_ptr<AOTIEagerKernelRunner> getAOTIEagerKernelRunner(
      const std::string&);
};

} // namespace torch::inductor
#endif