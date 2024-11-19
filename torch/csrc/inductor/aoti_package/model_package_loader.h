#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
class TORCH_API AOTIModelPackageLoader {
 public:
  AOTIModelPackageLoader(const std::string& model_package_path);
  AOTIModelPackageLoader(
      const std::string& model_package_path,
      const std::string& model_name);

  AOTIModelContainerRunner* get_runner();
  std::unordered_map<std::string, std::string> get_metadata();
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs);
  std::vector<std::string> get_call_spec();

 private:
  std::unique_ptr<AOTIModelContainerRunner> runner_;
  std::unordered_map<std::string, std::string> metadata_;

  void load_metadata(const std::string& cpp_filename);
};

} // namespace torch::inductor
#endif
