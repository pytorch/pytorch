#pragma once

#include <ATen/CPUFunctions.h>
#include <ATen/NativeFunctions.h>
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

// Implementation using native functions and pre-allocated tensors.
// It could be used as a "speed of light" for static runtime.
struct DeepAndWideFast : torch::nn::Module {
  DeepAndWideFast(int num_features = 50) {
    mu_ = register_parameter("mu_", torch::randn({1, num_features}));
    sigma_ = register_parameter("sigma_", torch::randn({1, num_features}));
    fc_w_ = register_parameter("fc_w_", torch::randn({1, num_features + 1}));
    fc_b_ = register_parameter("fc_b_", torch::randn({1}));
    allocated = false;
    prealloc_tensors = {};
  }

  torch::Tensor forward(
      torch::Tensor ad_emb_packed,
      torch::Tensor user_emb,
      torch::Tensor wide) {
    torch::NoGradGuard no_grad;
    if (!allocated) {
      auto wide_offset = at::add(wide, mu_);
      auto wide_normalized = at::mul(wide_offset, sigma_);
      // Placeholder for ReplaceNaN
      auto wide_preproc = at::cpu::clamp(wide_normalized, -10.0, 10.0);

      auto user_emb_t = at::native::transpose(user_emb, 1, 2);
      auto dp_unflatten = at::cpu::bmm(ad_emb_packed, user_emb_t);
      // auto dp = at::native::flatten(dp_unflatten, 1);
      auto dp = dp_unflatten.view({dp_unflatten.size(0), 1});
      auto input = at::cpu::cat({dp, wide_preproc}, 1);

      // fc1 = torch::nn::functional::linear(input, fc_w_, fc_b_);
      fc_w_t_ = torch::t(fc_w_);
      auto fc1 = torch::addmm(fc_b_, input, fc_w_t_);

      auto pred = at::cpu::sigmoid(fc1);

      prealloc_tensors = {
          wide_offset,
          wide_normalized,
          wide_preproc,
          user_emb_t,
          dp_unflatten,
          dp,
          input,
          fc1,
          pred};
      allocated = true;

      return pred;
    } else {
      // Potential optimization: add and mul could be fused together (e.g. with
      // Eigen).
      at::add_out(prealloc_tensors[0], wide, mu_);
      at::mul_out(prealloc_tensors[1], prealloc_tensors[0], sigma_);

      at::native::clip_out(
          prealloc_tensors[1], -10.0, 10.0, prealloc_tensors[2]);

      // Potential optimization: original tensor could be pre-transposed.
      // prealloc_tensors[3] = at::native::transpose(user_emb, 1, 2);
      if (prealloc_tensors[3].data_ptr() != user_emb.data_ptr()) {
        auto sizes = user_emb.sizes();
        auto strides = user_emb.strides();
        prealloc_tensors[3].set_(
            user_emb.storage(),
            0,
            {sizes[0], sizes[2], sizes[1]},
            {strides[0], strides[2], strides[1]});
      }

      // Potential optimization: call ONEDNN directly.
      at::cpu::bmm_out(ad_emb_packed, prealloc_tensors[3], prealloc_tensors[4]);

      if (prealloc_tensors[5].data_ptr() != prealloc_tensors[4].data_ptr()) {
        // in unlikely case that the input tensor changed we need to
        // reinitialize the view
        prealloc_tensors[5] =
            prealloc_tensors[4].view({prealloc_tensors[4].size(0), 1});
      }

      // Potential optimization: we can replace cat with carefully constructed
      // tensor views on the output that are passed to the _out ops above.
      at::cpu::cat_outf(
          {prealloc_tensors[5], prealloc_tensors[2]}, 1, prealloc_tensors[6]);
      at::cpu::addmm_out(
          prealloc_tensors[7], fc_b_, prealloc_tensors[6], fc_w_t_, 1, 1);
      at::cpu::sigmoid_out(prealloc_tensors[7], prealloc_tensors[8]);

      return prealloc_tensors[8];
    }
  }
  torch::Tensor mu_, sigma_, fc_w_, fc_b_, fc_w_t_;
  std::vector<torch::Tensor> prealloc_tensors;
  bool allocated = false;
};

torch::jit::Module getDeepAndWideSciptModel(int num_features = 50);

torch::jit::Module getTrivialScriptModel();

torch::jit::Module getLeakyReLUScriptModel();

torch::jit::Module getLeakyReLUConstScriptModel();

torch::jit::Module getLongScriptModel();

torch::jit::Module getSignedLog1pModel();
