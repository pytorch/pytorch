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

struct SGDOptions {
  /* implicit */ SGDOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(double, dampening) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, nesterov) = false;
};

class SGD : public Optimizer {
 public:
  SGD(std::shared_ptr<nn::Module> model, const SGDOptions& options);

  template <typename ModuleType>
  SGD(nn::ModuleHolder<ModuleType> module_holder, const SGDOptions& options)
      : SGD(module_holder.get(), options) {}

  void step() override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(momentum_buffers_));
  }

  const SGDOptions& options() const noexcept;

 private:
  friend class cereal::access;
  SGD() : options_(0) {}

  SGDOptions options_;
  std::unordered_map<std::string, at::Tensor> momentum_buffers_;
};
} // namespace optim
} // namespace torch
