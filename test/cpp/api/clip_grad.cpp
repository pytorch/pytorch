#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <torch/nn/utils/clip_grad.h>

using namespace torch::nn;
using namespace torch::test;

struct ClipGradTest : torch::test::SeedingFixture, torch::Tensor {};

class TestLinearModel : public torch::nn::Module {
 public:
  TestLinearModel() : l1(register_module("l1", Linear(10, 10))) {}
  Linear l1;
};

TEST_F(ClipGradTest, ClipGrad) {
  TestLinearModel m;
  auto linear_layer = m.l1;
  float max_norm = 2;
  auto compute_norm = [linear_layer](float norm_type) -> float {
    float inf = std::numeric_limits<float>::infinity();
    float total_norm = 0.0;
    if (norm_type != inf) {
      for (const auto& p : linear_layer->parameters()) {
        auto param_norm = torch::norm(p.grad(), norm_type);
        total_norm += torch::pow(param_norm, norm_type).item().toFloat();
      }
      total_norm = std::pow(total_norm, 1.0 / norm_type);
      return total_norm;
    } else {
      for (const auto& p : linear_layer->parameters()) {
        auto param_max = p.grad().abs().max().item().toFloat();
        if (param_max > total_norm) {
          total_norm = param_max;
        }
      }
      return total_norm;
    }
  };
  auto compare_scaling =
      [linear_layer](std::vector<torch::Tensor>& grads) -> torch::Tensor {
    std::vector<torch::Tensor> scaled;
    for (int i = 0; i < grads.size(); i++) {
      auto param = linear_layer->parameters()[i];
      auto grad = grads[i];
      scaled.push_back(torch::div(param.grad(), grad).view(-1));
    }
    auto scale = torch::cat(scaled);
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
          grads[i].clone().view_as(linear_layer->parameters()[i]);
    }
    auto norm_before = compute_norm(norm_type);
    auto layer_params = linear_layer->parameters();
    auto norm = clip_grad_norm_(layer_params, max_norm, norm_type);
    auto norm_after = compute_norm(norm_type);
    ASSERT_FLOAT_EQ(norm, norm_before);
    ASSERT_FLOAT_EQ(norm_after, max_norm);
    EXPECT_PRED_FORMAT2(::testing::FloatLE, norm_after, max_norm);
    auto scaled = compare_scaling(grads);
    ASSERT_NEAR(0, scaled.std().item().toFloat(), 1e-7);
  }
  // Small gradients should be lefted unchanged
  grads = {
      torch::rand({10, 10}).div(10000),
      torch::ones(10).div(1000),
  };
  for (auto norm_type : norm_types) {
    for (int i = 0; i < grads.size(); i++) {
      linear_layer->parameters()[i].grad().copy_(grads[i]);
    }
    auto norm_before = compute_norm(norm_type);
    auto layer_params = linear_layer->parameters();
    auto norm = clip_grad_norm_(layer_params, max_norm, norm_type);
    auto norm_after = compute_norm(norm_type);
    ASSERT_FLOAT_EQ(norm, norm_before);
    ASSERT_FLOAT_EQ(norm_before, norm_after);
    EXPECT_PRED_FORMAT2(::testing::FloatLE, norm_after, max_norm);
  }
  // should accept a single tensor as input
  auto p1 = torch::randn({10, 10});
  auto p2 = torch::randn({10, 10});
  auto g = torch::arange(1., 101).view({10, 10});
  p1.grad() = g.clone();
  p2.grad() = g.clone();
  for (const auto norm_type : norm_types) {
    std::vector<Tensor> params = {p1};
    clip_grad_norm_(params, max_norm, norm_type);
    params = {p2};
    clip_grad_norm_(params, max_norm, norm_type);
    ASSERT_TRUE(p1.grad().equal(p2.grad()));
  }
}
