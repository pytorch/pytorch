#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/serialize/base.h>

#include <utility>
#include <vector>

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
  template <typename ParameterContainer>
  explicit Adam(ParameterContainer&& parameters, const AdamOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {}

  void step() override;

  void save(serialize::Writer& writer) const override;
  void load(serialize::Reader& reader) override;

  AdamOptions options;

 private:
  Adam() : options(0) {}

  template <typename Self, typename Serializer>
  static void serialize(Self& self, Serializer& serializer) {
    optim::detail::serialize(serializer, "step_buffers", self.step_buffers_);
    serializer(
        "exp_average_buffers", self.exp_average_buffers_, /*is_buffer=*/true);
    serializer(
        "exp_average_sq_buffers",
        self.exp_average_sq_buffers_,
        /*is_buffer=*/true);
    serializer(
        "max_exp_average_sq_buffers",
        self.max_exp_average_sq_buffers_,
        /*is_buffer=*/true);
  }

  std::vector<int64_t> step_buffers_;
  std::vector<Tensor> exp_average_buffers_;
  std::vector<Tensor> exp_average_sq_buffers_;
  std::vector<Tensor> max_exp_average_sq_buffers_;
};
} // namespace optim
} // namespace torch
