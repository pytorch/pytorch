#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>

// Forward declare DynamicLibrary
namespace at {
struct DynamicLibrary;
}

namespace torch::inductor {
using ConstantMap = std::unordered_map<std::string, at::Tensor*>;

class TORCH_API AOTIModelRunner {
 public:
  AOTIModelRunner() = delete;
  AOTIModelRunner(const AOTIModelRunner& other) = delete;
  AOTIModelRunner(AOTIModelRunner&& other) = delete;
  AOTIModelRunner& operator=(const AOTIModelRunner& other) = delete;
  AOTIModelRunner& operator=(AOTIModelRunner&& other) = delete;

  void update_constants(const ConstantMap& const_map);
  void update_constants_map(const ConstantMap& const_map);

 protected:
  std::vector<at::Tensor> run(std::vector<at::Tensor> inputs);

  AOTIModelRunner(const char* model_path_, const ConstantMap& const_map);

  ~AOTIModelRunner();

  std::unique_ptr<at::DynamicLibrary> model_so_;
  decltype(&AOTInductorModelCreate) create_func_{nullptr};
  decltype(&AOTInductorModelDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelGetNumOutputs) get_num_outputs_func_{nullptr};
  decltype(&AOTInductorModelRun) run_func_{nullptr};
  decltype(&AOTInductorModelUpdateConstants) update_constants_func_{nullptr};
  decltype(&AOTInductorModelUpdateConstantsMap) update_constants_map_func_{
      nullptr};
  AOTInductorModelHandle model_handle_ = nullptr;
};

class TORCH_API AOTIModelRunnerCpu : public AOTIModelRunner {
 public:
  AOTIModelRunnerCpu(const char* model_path, const ConstantMap& const_map_)
      : AOTIModelRunner(model_path, const_map_) {}

  std::vector<at::Tensor> run(std::vector<at::Tensor> inputs) {
    return AOTIModelRunner::run(inputs);
  }
};

} // namespace torch::inductor
#endif
