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

/// The implementation of SGD with Momentum/Nesterov subtly differs from
/// Sutskever et. al. and implementations in some other frameworks.
/// Considering the specific case of Momentum, the update can be written as
///
/// .. math::
///           v = \rho * v + g \\
///           p = p - lr * v
/// where p, g, v and :math:`\rho` denote the parameters, gradient,
/// velocity, and momentum respectively.
/// This is in contrast to Sutskever et. al. and
/// other frameworks which employ an update of the form
///
/// .. math::
///      v = \rho * v + lr * g \\
///      p = p - v
///
/// The Nesterov version is analogously modified.
class TORCH_API SGD : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit SGD(ParameterContainer&& parameters, const SGDOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {}

  void step() override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

  SGDOptions options;

  std::vector<Tensor> momentum_buffers;

 private:
  SGD() : options(0) {}

  /// Counts how often `step()` is called, for dampening.
  size_t iteration_{0};
};
} // namespace optim
} // namespace torch
