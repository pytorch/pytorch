#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
class TORCH_API AOTIModelPackageLoader {
 public:
  AOTIModelPackageLoader(
      const std::string& model_package_path,
      const std::string& model_name = "model",
      const bool run_single_threaded = false,
      const size_t num_runners = 1,
      const c10::DeviceIndex device_index = -1);
  ~AOTIModelPackageLoader();

  AOTIModelContainerRunner* get_runner();
  std::unordered_map<std::string, std::string> get_metadata();

  std::vector<at::Tensor> run(
      const std::vector<at::Tensor>& inputs,
      void* stream_handle = nullptr);

  // boxed_run will steal the ownership of the input tensors
  std::vector<at::Tensor> boxed_run(
      std::vector<at::Tensor>&& inputs,
      void* stream_handle = nullptr);

  std::vector<std::string> get_call_spec();
  void load_constants(
      std::unordered_map<std::string, at::Tensor>& constants_map,
      bool use_inactive,
      bool check_full_update,
      bool user_managed = false);
  std::vector<std::string> get_constant_fqns();

  void update_constant_buffer(
      std::unordered_map<std::string, at::Tensor>& tensor_map,
      bool use_inactive,
      bool validate_full_updates,
      bool user_managed = false);

 private:
  std::string temp_dir_;
  std::unique_ptr<AOTIModelContainerRunner> runner_;
  std::unordered_map<std::string, std::string> metadata_;

  void load_metadata(const std::string& cpp_filename);
};

} // namespace torch::inductor
#endif
