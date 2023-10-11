#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>

// Forward declare DynamicLibrary
namespace at {
struct DynamicLibrary;
}

namespace torch::inductor {

class TORCH_API AOTIModelRunner {
 public:
  AOTIModelRunner() = delete;
  AOTIModelRunner(const AOTIModelRunner& other) = delete;
  AOTIModelRunner(AOTIModelRunner&& other) = delete;
  AOTIModelRunner& operator=(const AOTIModelRunner& other) = delete;
  AOTIModelRunner& operator=(AOTIModelRunner&& other) = delete;

 protected:
  std::vector<at::Tensor> run(
      std::vector<at::Tensor> inputs,
      AOTInductorStreamHandle cuda_stream_handle,
      AOTIProxyExecutorHandle proxy_executor_handle);

  AOTIModelRunner(
      const char* model_path,
      size_t num_models,
      bool is_cpu,
      const char* cubin_dir);

  ~AOTIModelRunner();

  std::unique_ptr<at::DynamicLibrary> model_so_;
  decltype(&AOTInductorModelContainerCreate) create_func_{nullptr};
  decltype(&AOTInductorModelContainerDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumOutputs) get_num_outputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerRun) run_func_{nullptr};
  AOTInductorModelContainerHandle container_handle_ = nullptr;
};

class TORCH_API AOTIModelRunnerCpu : public AOTIModelRunner {
 public:
  AOTIModelRunnerCpu(const char* model_path, size_t num_models = 1)
      : AOTIModelRunner(model_path, num_models, true, nullptr) {}

  std::vector<at::Tensor> run(
      std::vector<at::Tensor> inputs,
      AOTIProxyExecutorHandle proxy_executor_handle = nullptr) {
    return AOTIModelRunner::run(inputs, nullptr, proxy_executor_handle);
  }
};

} // namespace torch::inductor
#endif
