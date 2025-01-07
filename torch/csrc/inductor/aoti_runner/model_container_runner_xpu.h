#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <c10/xpu/XPUStream.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {

// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs

// HERE we use C10_EXPORT because libtorch_python needs this Symbol be exported.
// And `TORCH_API and `TORCH_XPU_API`` do not export the symbol in Windows
// build.
class C10_EXPORT AOTIModelContainerRunnerXpu : public AOTIModelContainerRunner {
 public:
  // @param device_str: xpu device string, e.g. "xpu", "xpu:0"
  AOTIModelContainerRunnerXpu(
      const std::string& model_so_path,
      size_t num_models = 1,
      const std::string& device_str = "xpu",
      const std::string& kernel_bin_dir = "");

  ~AOTIModelContainerRunnerXpu() override;

  std::vector<at::Tensor> run(
      const std::vector<at::Tensor>& inputs,
      void* stream_handle = nullptr) override;

  std::vector<at::Tensor> run_with_xpu_stream(
      std::vector<at::Tensor>& inputs,
      at::xpu::XPUStream xpu_stream);
};

} // namespace torch::inductor
#endif
