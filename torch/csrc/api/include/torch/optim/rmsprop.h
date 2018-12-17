#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/types.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

struct TORCH_API RMSpropOptions {
  RMSpropOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, alpha) = 0.99;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(bool, centered) = false;
};

class TORCH_API RMSprop : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit RMSprop(
      ParameterContainer&& parameters,
      const RMSpropOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {}

  void step() override;

  RMSpropOptions options;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

  std::vector<Tensor> square_average_buffers;
  std::vector<Tensor> momentum_buffers;
  std::vector<Tensor> grad_average_buffers;

 private:
  RMSprop() : options(0) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    TORCH_OPTIM_SERIALIZE(square_average_buffers);
    TORCH_OPTIM_SERIALIZE(momentum_buffers);
    TORCH_OPTIM_SERIALIZE(grad_average_buffers);
  }
};
} // namespace optim
} // namespace torch
