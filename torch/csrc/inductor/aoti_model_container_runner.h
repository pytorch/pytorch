#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>

// Forward declare DynamicLibrary
namespace at {
struct DynamicLibrary;
}

namespace torch::inductor {

class TORCH_API AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunner() = delete;
  AOTIModelContainerRunner(const AOTIModelContainerRunner& other) = delete;
  AOTIModelContainerRunner(AOTIModelContainerRunner&& other) = delete;
  AOTIModelContainerRunner& operator=(const AOTIModelContainerRunner& other) =
      delete;
  AOTIModelContainerRunner& operator=(AOTIModelContainerRunner&& other) =
      delete;

 protected:
  std::vector<at::Tensor> run(
      std::vector<at::Tensor> inputs,
      AOTInductorStreamHandle cuda_stream_handle,
      AOTIProxyExecutorHandle proxy_executor_handle);

  AOTIModelContainerRunner(
      const char* model_path,
      size_t num_models,
      bool is_cpu,
      const char* cubin_dir);

  ~AOTIModelContainerRunner();

  std::unique_ptr<at::DynamicLibrary> model_so_;
  decltype(&AOTInductorModelContainerCreate) create_func_{nullptr};
  decltype(&AOTInductorModelContainerDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumOutputs) get_num_outputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerRun) run_func_{nullptr};
  AOTInductorModelContainerHandle container_handle_ = nullptr;
};

class TORCH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCpu(const char* model_path, size_t num_models = 1)
      : AOTIModelContainerRunner(model_path, num_models, true, nullptr) {}

  std::vector<at::Tensor> run(
      std::vector<at::Tensor> inputs,
      AOTIProxyExecutorHandle proxy_executor_handle = nullptr) {
    return AOTIModelContainerRunner::run(
        inputs, nullptr, proxy_executor_handle);
  }
};

} // namespace torch::inductor
#endif
