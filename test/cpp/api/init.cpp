#include <gtest/gtest.h>

#include <torch/nn/init.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/conv.h>

#include <test/cpp/api/init_baseline.h>
#include <test/cpp/api/support.h>

#include <functional>
#include <vector>

void check_exact_values(
    const std::vector<torch::Tensor>& parameters,
    const std::vector<std::vector<torch::Tensor>>& expected_parameters) {
  ASSERT_EQ(parameters.size(), expected_parameters.size());

  for (size_t i = 0; i < parameters.size(); i++) {
    auto layerParameters = parameters[i];
    auto expectedLayerParameters = expected_parameters[i];

    if (layerParameters.size(0) != expectedLayerParameters.size()) {
      std::cout << "layer #" << i
                << " layerParameters size: " << layerParameters.size(0)
                << " != "
                << " expectedLayerParameters size: "
                << expectedLayerParameters.size() << std::endl;
      ASSERT_TRUE(false);
    }

    for (size_t p = 0; p < layerParameters.size(0); p++) {
      auto tensor = layerParameters[p];
      auto expectedTensor = expectedLayerParameters[p];

      if (!tensor.allclose(expectedTensor, /*rtol=*/1e-3, /*atol=*/5e-4)) {
        std::cout << "layer " << i << ": " << tensor << " != " << expectedTensor
                  << " (parameter " << p << ")" << std::endl;
        ASSERT_TRUE(false);
      }
    }
  }
}

void check_initializer_against_baseline(
    std::function<void(torch::Tensor)> initializer,
    std::vector<std::vector<torch::Tensor>> expected) {
  torch::manual_seed(0);

  auto layer1 = torch::nn::Linear(7, 15);
  initializer(layer1->weight);
  layer1->to(torch::kFloat64);

  auto layer2 = torch::nn::Linear(15, 15);
  initializer(layer2->weight);
  layer2->to(torch::kFloat64);

  auto layer3 = torch::nn::Linear(15, 2);
  initializer(layer3->weight);
  layer3->to(torch::kFloat64);

  auto parameters = std::vector<torch::Tensor>{
      layer1->weight,
      layer2->weight,
      layer3->weight,
  };

  check_exact_values(parameters, expected);
}

TEST(InitTest, ProducesPyTorchValues_XavierUniform) {
  auto expected = expected_parameters::Xavier_Uniform();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::xavier_uniform_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_XavierNormal) {
  auto expected = expected_parameters::Xavier_Normal();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::xavier_normal_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_KaimingNormal) {
  auto expected = expected_parameters::Kaiming_Normal();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::kaiming_normal_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_KaimingUniform) {
  auto expected = expected_parameters::Kaiming_Uniform();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::kaiming_uniform_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, CanInitializeTensorThatRequiresGrad) {
  auto tensor = torch::empty({3, 4}, torch::requires_grad());
  ASSERT_THROWS_WITH(
      tensor.fill_(1),
      "a leaf Variable that requires grad "
      "has been used in an in-place operation");
  ASSERT_EQ(torch::nn::init::ones_(tensor).sum().item<int32_t>(), 12);
}

TEST(InitTest, CalculateGainWithTanh) {
  double gain =
      torch::nn::init::calculate_gain(torch::kTanh);
  ASSERT_DOUBLE_EQ(gain, 5.0 / 3.0);
}

TEST(InitTest, CalculateGainWithRelu) {
  double gain =
      torch::nn::init::calculate_gain(torch::kReLU);
  ASSERT_DOUBLE_EQ(gain, std::sqrt(2.0));
}

TEST(InitTest, CalculateGainWithLeakyRelu) {
  double gain =
      torch::nn::init::calculate_gain(torch::kLeakyReLU);
  ASSERT_DOUBLE_EQ(gain, std::sqrt(2.0 / (1 + pow(0.01, 2))));
}

TEST(InitTest, CanInitializeCnnWithOrthogonal) {
  torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(3, 2, 3).stride(2));
  torch::nn::init::orthogonal_(conv_layer->named_parameters()["weight"]);
}