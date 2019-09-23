#pragma once

#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/types.h>

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

struct TORCH_API AdagradOptions {
  AdagradOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
};

class TORCH_API Adagrad : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit Adagrad(
      ParameterContainer&& parameters,
      const AdagradOptions& options_)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options_) {}

  void step() override;

  AdagradOptions options;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

  std::vector<Tensor> sum_buffers;
  std::vector<int64_t> step_buffers;

 private:
  Adagrad() : options(0) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE(sum_buffers);
    _TORCH_OPTIM_SERIALIZE(step_buffers);
  }
};
} // namespace optim
} // namespace torch
