#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>

#include <ATen/ATen.h>

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>

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

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(d));
    ar(CEREAL_NVP(t));
    ar(CEREAL_NVP(H_diag));
    ar(CEREAL_NVP(prev_flat_grad));
    ar(CEREAL_NVP(prev_loss));
    ar(CEREAL_NVP(old_dirs));
    ar(CEREAL_NVP(old_stps));
  }

 private:
  friend class cereal::access;
  LBFGS() : options(0) {}

  Tensor gather_flat_grad();
  void add_grad(const torch::Scalar& step_size, const Tensor& update);

  Tensor d{torch::empty({0})};
  Tensor H_diag{torch::empty({0})};
  Tensor prev_flat_grad{torch::empty({0})};
  torch::Scalar t{0};
  torch::Scalar prev_loss{0};
  std::vector<Tensor> ro;
  std::vector<Tensor> al;
  std::deque<Tensor> old_dirs;
  std::deque<Tensor> old_stps;
  int64_t func_evals{0};
  int64_t state_n_iter{0};
};

} // namespace optim
} // namespace torch
