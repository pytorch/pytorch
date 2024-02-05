#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>

// Forward declare DynamicLibrary
namespace at {
struct DynamicLibrary;
}

namespace torch::inductor {
using TensorConstantMap = std::unordered_map<std::string, at::Tensor*>;

class TORCH_API AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunner() = delete;
  AOTIModelContainerRunner(const AOTIModelContainerRunner& other) = delete;
  AOTIModelContainerRunner(AOTIModelContainerRunner&& other) = delete;
  AOTIModelContainerRunner& operator=(const AOTIModelContainerRunner& other) =
      delete;
  AOTIModelContainerRunner& operator=(AOTIModelContainerRunner&& other) =
      delete;
  ~AOTIModelContainerRunner();

  std::vector<at::Tensor> run(
      std::vector<at::Tensor>& inputs,
      AOTInductorStreamHandle cuda_stream_handle = nullptr);

  std::unordered_map<std::string, std::string> getConstantNamesToOriginalFQNs()
      const;
  std::unordered_map<std::string, int32_t> getConstantNamesToDtypes() const;
  void update_inactive_constant_buffer(const TensorConstantMap& const_map);
  void update_constant_buffer(
      const TensorConstantMap& const_map,
      bool use_inactive,
      bool validate_full_updates);
  void run_const_fold(
      bool use_inactive,
      AOTInductorStreamHandle cuda_stream_handle = nullptr);
  void swap_constant_buffer();

  std::vector<std::string> get_call_spec();

 protected:
  AOTIModelContainerRunner(
      const std::string& model_so_path,
      size_t num_models,
      const std::string& device_str,
      const std::string& cubin_dir);

  std::unique_ptr<at::DynamicLibrary> model_so_;
  decltype(&AOTInductorModelContainerCreateWithDevice) create_func_{nullptr};
  decltype(&AOTInductorModelContainerDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumOutputs) get_num_outputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerRun) run_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumConstants) get_num_constants_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantName) get_constant_name_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantOriginalFQN)
      get_constant_original_fqn_func_{nullptr};
  decltype(&AOTInductorModelContainerGetConstantDtype) get_constant_dtype_func_{
      nullptr};
  decltype(&AOTInductorModelContainerUpdateConstantBuffer)
      update_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateInactiveConstantBuffer)
      update_inactive_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerRunConstantFolding) run_const_fold_func_{
      nullptr};
  decltype(&AOTInductorModelContainerSwapConstantBuffer)
      swap_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerGetCallSpec) get_call_spec_func_{nullptr};

  AOTInductorModelContainerHandle container_handle_ = nullptr;

  // TODO: need an OSS proxy executor implementation. For now,
  // proxy_executor_handle_ will always be nullptr.
  AOTIProxyExecutorHandle proxy_executor_handle_ = nullptr;
};

} // namespace torch::inductor
#endif
