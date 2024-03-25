#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
class TORCH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCpu(
      const std::string& model_so_path,
      size_t num_models = 1);

  ~AOTIModelContainerRunnerCpu();

  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs);
};

class TORCH_API AOTIEagerKernelRunnerCPU : public AOTIEagerKernelRunner {
 public:
  AOTIEagerKernelRunnerCPU(const std::string& kernel_so_path);
  ~AOTIEagerKernelRunnerCPU(){};

  void operator()(
      AtenTensorHandle* input_handles,
      AtenTensorHandle* output_handles);
};

} // namespace torch::inductor
#endif
