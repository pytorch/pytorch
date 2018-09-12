#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/serialize/base.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace optim {

struct RMSpropOptions {
  RMSpropOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, alpha) = 0.99;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(bool, centered) = false;
};

class RMSprop : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit RMSprop(
      ParameterContainer&& parameters,
      const RMSpropOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {}

  void step() override;

  RMSpropOptions options;

  void save(serialize::Writer& writer) const override;
  void load(serialize::Reader& reader) override;

 private:
  RMSprop() : options(0) {}

  template <typename Self, typename Serializer>
  static void serialize(Self& self, Serializer& serializer) {
    serializer(
        "square_average_buffers",
        self.square_average_buffers_,
        /*is_buffer=*/true);
    serializer("momentum_buffers", self.momentum_buffers_, /*is_buffer=*/true);
    serializer(
        "grad_average_buffers", self.grad_average_buffers_, /*is_buffer=*/true);
  }

  std::vector<Tensor> square_average_buffers_;
  std::vector<Tensor> momentum_buffers_;
  std::vector<Tensor> grad_average_buffers_;
};
} // namespace optim
} // namespace torch
