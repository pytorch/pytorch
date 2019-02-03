#include <gtest/gtest.h>

#include <torch/nn/modules/linear.h>
#include <torch/nn/init.h>

#include <test/cpp/api/init_baseline.h>
#include <test/cpp/api/support.h>

#include <vector>


void check_exact_values(
    std::vector<torch::Tensor> parameters,
    std::vector<std::vector<torch::Tensor>> expected_parameters) {

    ASSERT_EQ(parameters.size(), expected_parameters.size());

    for(size_t i = 0; i < parameters.size(); i++) {

      auto layerParameters = parameters[i];
      auto expectedLayerParameters = expected_parameters[i];

      ASSERT_EQ(layerParameters.size(0), expectedLayerParameters.size());

      for(size_t p = 0; p < layerParameters.size(0); p++) {
        auto tensor = layerParameters[p];
        auto expectedTensor = expectedLayerParameters[p];

        ASSERT_TRUE(tensor.allclose(expectedTensor, /*rtol=*/1e-3, /*atol=*/5e-4));
      }
    }
}

void check_initializer_against_baseline(std::function<void (torch::Tensor)> initializer, 
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

  auto parameters = std::vector<torch::Tensor> {
    layer1->weight,
    layer2->weight,
    layer3->weight,
  };

  check_exact_values(parameters, expected);
}

TEST(InitTest, ProducesPyTorchValues_XavierUniform) {
  auto expected = expected_parameters::Xavier_Uniform();
  auto initializer = [] (torch::Tensor tensor) -> void { torch::nn::init::xavier_uniform_(tensor); };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_XavierNormal) {
  auto expected = expected_parameters::Xavier_Normal();
  auto initializer = [] (torch::Tensor tensor) -> void { torch::nn::init::xavier_normal_(tensor); };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_KaimingNormal) {
  auto expected = expected_parameters::Kaiming_Normal();
  auto initializer = [] (torch::Tensor tensor) -> void { torch::nn::init::kaiming_normal_(tensor); };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_KaimingUniform) {
  auto expected = expected_parameters::Kaiming_Uniform();
  auto initializer = [] (torch::Tensor tensor) -> void { torch::nn::init::kaiming_uniform_(tensor); };
  check_initializer_against_baseline(initializer, expected);
}