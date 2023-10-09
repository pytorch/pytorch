#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>

// Forward declare DynamicLibrary
namespace at {
struct DynamicLibrary;
}

namespace torch {
namespace inductor {

class TORCH_API AOTIModelRunner {
 public:
  AOTIModelRunner(
      const char* model_path,
      size_t num_models = 1,
      bool is_cpu = false,
      const char* cubin_dir = nullptr);

  ~AOTIModelRunner();

  std::vector<at::Tensor> run(
      std::vector<at::Tensor> inputs,
      AOTInductorStreamHandle cuda_stream_handle = nullptr,
      AOTIProxyExecutorHandle proxy_executor_handle = nullptr);

 private:
  std::unique_ptr<at::DynamicLibrary> model_so_;
  decltype(&AOTInductorModelContainerCreate) create_func_{nullptr};
  decltype(&AOTInductorModelContainerDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumOutputs) get_num_outputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerRun) run_func_{nullptr};
  AOTInductorModelContainerHandle container_handle_ = nullptr;
};

} // namespace inductor
} // namespace torch
