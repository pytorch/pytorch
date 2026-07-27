#pragma once

#include <memory>

#include <torch/torch.h>

namespace torch::inductor {

class AOTIModelContainerRunner;

} // namespace torch::inductor

namespace torch::aot_inductor {

class MyAOTIClass : public torch::CustomClassHolder {
 public:
  explicit MyAOTIClass(
      const std::string& model_path,
      const std::string& device = "cuda");

  ~MyAOTIClass() {}

  MyAOTIClass(const MyAOTIClass&) = delete;
  MyAOTIClass& operator=(const MyAOTIClass&) = delete;
  MyAOTIClass& operator=(MyAOTIClass&&) = delete;

  const std::string& lib_path() const {
    return lib_path_;
  }

  const std::string& device() const {
    return device_;
  }

  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs);

 private:
  const std::string lib_path_;

  const std::string device_;

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner_;
};

} // namespace torch::aot_inductor
