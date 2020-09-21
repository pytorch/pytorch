#pragma once

#include <torch/torch.h>

struct DeepAndWide : torch::nn::Module {
  DeepAndWide(int num_features = 50) {
    mu_ = register_parameter("mu_", torch::randn({1, num_features}));
    sigma_ = register_parameter("sigma_", torch::randn({1, num_features}));
    fc_w_ = register_parameter("fc_w_", torch::randn({1, num_features + 1}));
    fc_b_ = register_parameter("fc_b_", torch::randn({1}));
  }

  torch::Tensor forward(
      torch::Tensor ad_emb_packed,
      torch::Tensor user_emb,
      torch::Tensor wide) {
    auto wide_offset = wide + mu_;
    auto wide_normalized = wide_offset * sigma_;
    auto wide_noNaN = wide_normalized;
    // Placeholder for ReplaceNaN
    auto wide_preproc = torch::clamp(wide_noNaN, -10.0, 10.0);

    auto user_emb_t = torch::transpose(user_emb, 1, 2);
    auto dp_unflatten = torch::bmm(ad_emb_packed, user_emb_t);
    auto dp = torch::flatten(dp_unflatten, 1);
    auto input = torch::cat({dp, wide_preproc}, 1);
    auto fc1 = torch::nn::functional::linear(input, fc_w_, fc_b_);
    auto pred = torch::sigmoid(fc1);
    return pred;
  }
  torch::Tensor mu_, sigma_, fc_w_, fc_b_;
};

torch::jit::Module getDeepAndWideSciptModel(int num_features = 50);

torch::jit::Module getTrivialScriptModel();
