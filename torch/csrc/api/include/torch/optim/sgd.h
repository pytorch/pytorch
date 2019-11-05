#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/types.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

struct TORCH_API SGDOptions {
  /* implicit */ SGDOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(double, dampening) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, nesterov) = false;
};

class TORCH_API SGD : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit SGD(ParameterContainer&& parameters, const SGDOptions& options_)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options_) {}

  void step() override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;
  int64_t iteration() const;

  SGDOptions options;

  std::vector<Tensor> momentum_buffers;

 private:
  SGD() : options(0) {}

  /// Counts how often `step()` is called, for dampening.
  int64_t iteration_{0};
};
} // namespace optim
} // namespace torch
