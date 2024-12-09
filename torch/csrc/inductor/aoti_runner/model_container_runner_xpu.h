#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <c10/xpu/XPUStream.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {

// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs
class TORCH_API AOTIModelContainerRunnerXpu : public AOTIModelContainerRunner {
 public:
  // @param device_str: xpu device string, e.g. "xpu", "xpu:0"
  AOTIModelContainerRunnerXpu(
      const std::string& model_so_path,
      size_t num_models = 1,
      const std::string& device_str = "xpu",
      const std::string& cubin_dir = "");

  ~AOTIModelContainerRunnerXpu();

  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs);

  std::vector<at::Tensor> run_with_xpu_stream(
      std::vector<at::Tensor>& inputs,
      at::xpu::XPUStream xpu_stream);
};

} // namespace torch::inductor
#endif
