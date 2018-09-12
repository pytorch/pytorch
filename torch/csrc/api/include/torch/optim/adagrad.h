#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/serialize/base.h>
#include <torch/tensor.h>

#include <utility>
#include <vector>

namespace torch {
namespace optim {

struct AdagradOptions {
  AdagradOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
};

class Adagrad : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit Adagrad(
      ParameterContainer&& parameters,
      const AdagradOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {}

  void step() override;

  AdagradOptions options;

  void save(serialize::Writer& writer) const override;
  void load(serialize::Reader& reader) override;

 private:
  Adagrad() : options(0) {}

  template <typename Self, typename Serializer>
  static void serialize(Self& self, Serializer& serializer) {
    optim::detail::serialize(serializer, "step", self.step_);
    serializer("sum", self.sum_, /*is_buffer=*/true);
  }

  std::vector<Tensor> sum_;
  std::vector<int64_t> step_;
};
} // namespace optim
} // namespace torch
