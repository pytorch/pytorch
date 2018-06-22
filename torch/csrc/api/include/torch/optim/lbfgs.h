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
  TORCH_ARG(int, max_iter) = 20;
  TORCH_ARG(int, max_eval) = 25;
  TORCH_ARG(float, tolerance_grad) = 1e-5;
  TORCH_ARG(float, tolerance_change) = 1e-9;
  TORCH_ARG(int, history_size) = 100;
};

class LBFGS : public LossClosureOptimizer {
 public:
  LBFGS(std::shared_ptr<nn::Module> model, const LBFGSOptions& options);

  template <typename ModuleType>
  LBFGS(nn::ModuleHolder<ModuleType> module_holder, const LBFGSOptions& options)
      : LBFGS(module_holder.get(), options) {}

  at::Scalar step(LossClosure closure) override;

  const LBFGSOptions& options() const noexcept;

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
  LBFGS() : options_(0) {}

  at::Tensor gather_flat_grad();
  void add_grad(const at::Scalar& step_size, const at::Tensor& update);

  LBFGSOptions options_;

  at::Tensor d;
  at::Tensor H_diag;
  at::Tensor prev_flat_grad;
  at::Scalar t;
  at::Scalar prev_loss;
  std::vector<at::Tensor> ro;
  std::vector<at::Tensor> al;
  std::deque<at::Tensor> old_dirs;
  std::deque<at::Tensor> old_stps;
  int64_t func_evals;
  int64_t state_n_iter;
};

} // namespace optim
} // namespace torch
