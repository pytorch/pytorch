#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>

#include <ATen/ATen.h>

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace torch {
namespace optim {

struct AdamOptions {
  /* implicit */ AdamOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.999;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(bool, amsgrad) = false;
};

class Adam : public Optimizer {
 public:
  Adam(std::shared_ptr<nn::Module> model, const AdamOptions& options);

  template <typename ModuleType>
  Adam(nn::ModuleHolder<ModuleType> module_holder, const AdamOptions& options)
      : Adam(module_holder.get(), options) {}

  void step() override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(step_buffer_),
       CEREAL_NVP(exp_avg_buffer_),
       CEREAL_NVP(exp_avg_sq_buffer_),
       CEREAL_NVP(max_exp_avg_sq_buffer_));
  }

  const AdamOptions& options() const noexcept;

 private:
  friend class cereal::access;
  Adam() : options_(0) {}

  AdamOptions options_;

  std::unordered_map<std::string, int> step_buffer_;
  std::unordered_map<std::string, at::Tensor> exp_avg_buffer_;
  std::unordered_map<std::string, at::Tensor> exp_avg_sq_buffer_;
  std::unordered_map<std::string, at::Tensor> max_exp_avg_sq_buffer_;
};

} // namespace optim
} // namespace torch
