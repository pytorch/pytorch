#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/serialize/base.h>

#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace optim {

struct LBFGSOptions {
  LBFGSOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(int64_t, max_iter) = 20;
  TORCH_ARG(int64_t, max_eval) = 25;
  TORCH_ARG(float, tolerance_grad) = 1e-5;
  TORCH_ARG(float, tolerance_change) = 1e-9;
  TORCH_ARG(size_t, history_size) = 100;
};

class LBFGS : public LossClosureOptimizer {
 public:
  template <typename ParameterContainer>
  explicit LBFGS(ParameterContainer&& parameters, const LBFGSOptions& options)
      : LossClosureOptimizer(std::forward<ParameterContainer>(parameters)),
        options(options),
        ro(options.history_size_),
        al(options.history_size_) {}

  torch::Tensor step(LossClosure closure) override;

  LBFGSOptions options;

  void save(serialize::Writer& writer) const override;
  void load(serialize::Reader& reader) override;

 private:
  LBFGS() : options(0) {}

  Tensor gather_flat_grad();
  void add_grad(const torch::Tensor& step_size, const Tensor& update);

  template <typename Self, typename Serializer>
  static void serialize(Self& self, Serializer& serializer) {
    serializer("d", self.d, /*is_buffer=*/true);
    serializer("t", self.t, /*is_buffer=*/true);
    serializer("H_diag", self.H_diag, /*is_buffer=*/true);
    serializer("prev_flat_grad", self.prev_flat_grad, /*is_buffer=*/true);
    serializer("prev_loss", self.prev_loss, /*is_buffer=*/true);
  }

  Tensor d{torch::empty({0})};
  Tensor H_diag{torch::empty({0})};
  Tensor prev_flat_grad{torch::empty({0})};
  Tensor t{torch::zeros(1)};
  Tensor prev_loss{torch::zeros(1)};
  std::vector<Tensor> ro;
  std::vector<Tensor> al;
  std::deque<Tensor> old_dirs;
  std::deque<Tensor> old_stps;
  int64_t func_evals{0};
  int64_t state_n_iter{0};
};
} // namespace optim
} // namespace torch
