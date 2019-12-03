#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct NNUtilsTest : torch::test::SeedingFixture {};

TEST_F(NNUtilsTest, ClipGradNorm) {
  auto l = Linear(10, 10);
  float max_norm = 2;
  auto compute_norm = [&](float norm_type) -> float {
    float total_norm = 0.0;
    if (norm_type != std::numeric_limits<float>::infinity()) {
      for (const auto& p : l->parameters()) {
        total_norm +=
            p.grad().data().abs().pow(norm_type).sum().item().toFloat();
      }
      return std::pow(total_norm, 1.0 / norm_type);
    } else {
      for (const auto& p : l->parameters()) {
        auto param_max = p.grad().data().abs().max().item().toFloat();
        if (param_max > total_norm) {
          total_norm = param_max;
        }
      }
      return total_norm;
    }
  };
  auto compare_scaling =
      [&](const std::vector<torch::Tensor>& grads) -> torch::Tensor {
    std::vector<torch::Tensor> p_scale;
    for (int i = 0; i < grads.size(); i++) {
      auto param = l->parameters()[i];
      auto grad = grads[i];
      p_scale.push_back(param.grad().data().div(grad).view(-1));
    }
    auto scale = torch::cat(p_scale);
    return scale; // need to assert std is 0.
  };

  std::vector<torch::Tensor> grads = {
      torch::arange(1.0, 101).view({10, 10}),
      torch::ones({10}).div(1000),
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
      l->parameters()[i].grad() =
          grads[i].clone().view_as(l->parameters()[i].data());
    }
    auto norm_before = compute_norm(norm_type);
    auto norm = utils::clip_grad_norm_(l->parameters(), max_norm, norm_type);
    auto norm_after = compute_norm(norm_type);
    ASSERT_FLOAT_EQ(norm, norm_before);
    ASSERT_FLOAT_EQ(norm_after, max_norm);
    ASSERT_LE(norm_after, max_norm);
    auto scaled = compare_scaling(grads);
    ASSERT_NEAR(0, scaled.std().item().toFloat(), 1e-7);
  }
  // Small gradients should be left unchanged
  grads = {
      torch::rand({10, 10}).div(10000),
      torch::ones(10).div(500),
  };
  for (auto norm_type : norm_types) {
    for (int i = 0; i < grads.size(); i++) {
      l->parameters()[i].grad().data().copy_(grads[i]);
    }
    auto norm_before = compute_norm(norm_type);
    auto norm = utils::clip_grad_norm_(l->parameters(), max_norm, norm_type);
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
    utils::clip_grad_norm_({p2}, max_norm, norm_type);
    ASSERT_TRUE(torch::allclose(p1.grad(), p2.grad()));
  }
}

TEST_F(NNUtilsTest, ClipGradValue) {
  auto l = Linear(10, 10);
  float clip_value = 2.5;

  torch::Tensor grad_w = torch::arange(-50., 50).view({10, 10}).div_(5);
  torch::Tensor grad_b = torch::ones({10}).mul_(2);
  std::vector<std::vector<torch::Tensor>> grad_lists = {
      {grad_w, grad_b}, {grad_w, torch::Tensor()}};
  for (auto grad_list : grad_lists) {
    for (int i = 0; i < grad_list.size(); i++) {
      auto p = l->parameters()[i];
      auto g = grad_list[i];
      p.grad() = g.defined() ? g.clone().view_as(p.data()) : g;
    }

    utils::clip_grad_value_(l->parameters(), clip_value);
    for (const auto& p : l->parameters()) {
      if (p.grad().defined()) {
        ASSERT_LE(
            p.grad().data().max().item().toFloat(), clip_value);
        ASSERT_GE(
            p.grad().data().min().item().toFloat(), -clip_value);
      }
    }
  }

  // Should accept a single Tensor as input
  auto p1 = torch::randn({10, 10});
  auto p2 = torch::randn({10, 10});
  auto g = torch::arange(-50., 50).view({10, 10}).div_(5);
  p1.grad() = g.clone();
  p2.grad() = g.clone();
  utils::clip_grad_value_(p1, clip_value);
  utils::clip_grad_value_({p2}, clip_value);
  ASSERT_TRUE(torch::allclose(p1.grad(), p2.grad()));
}

TEST_F(NNUtilsTest, ConvertParameters) {
  std::vector<torch::Tensor> parameters{
    torch::arange(9, torch::kFloat32),
    torch::arange(9, torch::kFloat32).view({3, 3}),
    torch::arange(8, torch::kFloat32).view({2, 2, 2})
  };

  auto expected = torch::cat({
    torch::arange(9, torch::kFloat32),
    torch::arange(9, torch::kFloat32).view(-1),
    torch::arange(8, torch::kFloat32).view(-1)
  });
  auto vector = utils::parameters_to_vector(parameters);
  ASSERT_TRUE(vector.allclose(expected));

  std::vector<torch::Tensor> zero_parameters{
    torch::zeros({9}, torch::kFloat32),
    torch::zeros({9}, torch::kFloat32).view({3, 3}),
    torch::zeros({8}, torch::kFloat32).view({2, 2, 2})
  };

  utils::vector_to_parameters(vector, zero_parameters);
  for (int i = 0; i < zero_parameters.size(); ++i) {
    ASSERT_TRUE(zero_parameters[i].allclose(parameters[i]));
  }

  {
    auto conv1 = Conv2d(3, 10, 5);
    auto fc1 = Linear(10, 20);
    auto model = Sequential(conv1, fc1);

    auto vec = utils::parameters_to_vector(model->parameters());
    ASSERT_EQ(vec.size(0), 980);
  }
  {
    auto conv1 = Conv2d(3, 10, 5);
    auto fc1 = Linear(10, 20);
    auto model = Sequential(conv1, fc1);

    auto vec = torch::arange(0., 980);
    utils::vector_to_parameters(vec, model->parameters());

    auto sample = model->parameters()[0][0][0][0];
    ASSERT_TRUE(torch::equal(sample.data(), vec.data().slice(0, 0, 5)));
  }
}
