#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct NNUtilsTest : torch::test::SeedingFixture {};

TEST_F(NNUtilsTest, ClipGradNorm) {
  auto linear_layer = Linear(10, 10);
  float max_norm = 2;
  auto compute_norm = [linear_layer](float norm_type) -> float {
    float total_norm = 0.0;
    if (norm_type != std::numeric_limits<float>::infinity()) {
      for (const auto& p : linear_layer->parameters()) {
        total_norm +=
            p.grad().data().abs().pow(norm_type).sum().item().toFloat();
      }
      return std::pow(total_norm, 1.0 / norm_type);
    } else {
      for (const auto& p : linear_layer->parameters()) {
        auto param_max = p.grad().data().abs().max().item().toFloat();
        if (param_max > total_norm) {
          total_norm = param_max;
        }
      }
      return total_norm;
    }
  };
  auto compare_scaling =
      [linear_layer](const std::vector<torch::Tensor>& grads) -> torch::Tensor {
    std::vector<torch::Tensor> p_scale;
    for (int i = 0; i < grads.size(); i++) {
      auto param = linear_layer->parameters()[i];
      auto grad = grads[i];
      p_scale.push_back(param.grad().data().div(grad).view(-1));
    }
    auto scale = torch::cat(p_scale);
    return scale; // need to assert std is 0.
  };

  std::vector<torch::Tensor> grads = {
      torch::arange(1.0, 101).view({10, 10}),
      torch::ones(10).div(1000),
  };
  std::vector<float> norm_types = {
      0.5,
      1.5,
      2.0,
      4.0,
      std::numeric_limits<float>::infinity(),
  };
  for (auto norm_type : norm_types) {
    for (int i = 0; i < grads.size(); i++) {
      linear_layer->parameters()[i].grad() =
          grads[i].clone().view_as(linear_layer->parameters()[i].data());
    }
    auto norm_before = compute_norm(norm_type);
    auto layer_params = linear_layer->parameters();
    auto norm = utils::clip_grad_norm_(layer_params, max_norm, norm_type);
    auto norm_after = compute_norm(norm_type);
    ASSERT_FLOAT_EQ(norm, norm_before);
    ASSERT_FLOAT_EQ(norm_after, max_norm);
    ASSERT_LE(norm_after, max_norm);
    auto scaled = compare_scaling(grads);
    ASSERT_NEAR(0, scaled.std().item().toFloat(), 1e-7);
  }
  // Small gradients should be lefted unchanged
  grads = {
      torch::rand({10, 10}).div(10000),
      torch::ones(10).div(500),
  };
  for (auto norm_type : norm_types) {
    for (int i = 0; i < grads.size(); i++) {
      linear_layer->parameters()[i].grad().data().copy_(grads[i]);
    }
    auto norm_before = compute_norm(norm_type);
    auto layer_params = linear_layer->parameters();
    auto norm = utils::clip_grad_norm_(layer_params, max_norm, norm_type);
    auto norm_after = compute_norm(norm_type);
    ASSERT_FLOAT_EQ(norm, norm_before);
    ASSERT_FLOAT_EQ(norm_before, norm_after);
    ASSERT_LE(norm_after, max_norm);
    auto scaled = compare_scaling(grads);
    ASSERT_NEAR(0, scaled.std().item().toFloat(), 1e-7);
    ASSERT_EQ(scaled[0].item().toFloat(), 1);
  }
  // should accept a single tensor as input
  auto p1 = torch::randn({10, 10});
  auto p2 = torch::randn({10, 10});
  auto g = torch::arange(1., 101).view({10, 10});
  p1.grad() = g.clone();
  p2.grad() = g.clone();
  for (const auto norm_type : norm_types) {
    utils::clip_grad_norm_(p1, max_norm, norm_type);
    std::vector<torch::Tensor> params = {p2};
    utils::clip_grad_norm_(params, max_norm, norm_type);
    ASSERT_TRUE(torch::allclose(p1.grad(), p2.grad()));
  }
}
