#include <catch.hpp>

#include <torch/nn/module.h>
#include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/optim.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/optim_baseline.h>
#include <test/cpp/api/util.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace torch::nn;
using namespace torch::optim;

template <typename OptimizerClass, typename Options>
bool test_optimizer_xor(Options options) {
  torch::manual_seed(0);

  Sequential model(
      Linear(2, 8),
      Functional(torch::sigmoid),
      Linear(8, 1),
      Functional(torch::sigmoid));

  const int64_t kBatchSize = 4;
  const int64_t kMaximumNumberOfEpochs = 3000;

  auto optimizer = OptimizerClass(std::vector<torch::Tensor>(), options);
  optimizer.add_parameters(model->parameters());

  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    auto inputs = torch::empty({kBatchSize, 2});
    auto labels = torch::empty({kBatchSize});
    for (size_t i = 0; i < kBatchSize; i++) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].toCLong() ^ inputs[i][1].toCLong();
    }
    inputs.set_requires_grad(true);
    optimizer.zero_grad();
    auto x = model->forward(inputs);
    torch::Tensor loss = torch::binary_cross_entropy(x, labels);
    loss.backward();

    optimizer.step();

    running_loss = running_loss * 0.99 + loss.toCFloat() * 0.01;
    if (epoch > kMaximumNumberOfEpochs) {
      std::cout << "Loss is too high after epoch " << epoch << ": "
                << running_loss << std::endl;
      return false;
    }
    epoch++;
  }
  return true;
}

template <typename Parameters>
void assign_parameter(
    const Parameters& parameters,
    const char* name,
    torch::Tensor new_tensor) {
  auto parameter = parameters.at(name);
  parameter.set_requires_grad(false);
  parameter.flatten().copy_(new_tensor);
  parameter.set_requires_grad(true);
}

template <typename OptimizerClass, typename Options>
void check_exact_values(
    Options options,
    std::vector<std::vector<torch::Tensor>> expected_parameters) {
  const size_t kIterations = 1001;
  const size_t kSampleEvery = 100;

  torch::manual_seed(0);

  Sequential model(
      Linear(2, 3),
      Functional(torch::sigmoid),
      Linear(3, 1),
      Functional(torch::sigmoid));

  model->to(torch::kFloat64);

  // Use exact input values because matching random values is hard.
  auto parameters = model->parameters();
  assign_parameter(
      parameters,
      "0.weight",
      torch::tensor({-0.2109, -0.4976, -0.1413, -0.3420, -0.2524, 0.6976}));
  assign_parameter(
      parameters, "0.bias", torch::tensor({-0.1085, -0.2979, 0.6892}));
  assign_parameter(
      parameters, "2.weight", torch::tensor({-0.0508, -0.3941, -0.2843}));
  assign_parameter(parameters, "2.bias", torch::tensor({-0.0711}));

  auto optimizer = OptimizerClass(parameters, options);
  torch::Tensor input =
      torch::tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}).reshape({3, 2});

  for (size_t i = 0; i < kIterations; ++i) {
    optimizer.zero_grad();
    auto output = model->forward(input);
    auto loss = output.sum();
    loss.backward();

    optimizer.step();

    if (i % kSampleEvery == 0) {
      REQUIRE(
          expected_parameters.at(i / kSampleEvery).size() == parameters.size());
      for (size_t p = 0; p < parameters.size(); ++p) {
        REQUIRE(parameters.at(p)->defined());
        auto computed = parameters.at(p)->flatten();
        auto expected = expected_parameters.at(i / kSampleEvery).at(p);
        if (!computed.allclose(expected, /*rtol=*/1e-3, /*atol=*/5e-4)) {
          std::cout << "Iteration " << i << ": " << computed
                    << " != " << expected << " (parameter " << p << ")"
                    << std::endl;
          REQUIRE(false);
        }
      }
    }
  }
}

TEST_CASE("Optim/BasicInterface") {
  struct MyOptimizer : Optimizer {
    using Optimizer::Optimizer;
    void step() override {}
  };
  std::vector<torch::Tensor> parameters = {
      torch::ones({2, 3}), torch::zeros({2, 3}), torch::rand({2, 3})};
  {
    MyOptimizer optimizer(parameters);
    REQUIRE(optimizer.size() == parameters.size());
  }
  {
    MyOptimizer optimizer;
    REQUIRE(optimizer.size() == 0);
    optimizer.add_parameters(parameters);
    REQUIRE(optimizer.size() == parameters.size());
  }
  {
    Linear linear(3, 4);
    MyOptimizer optimizer(linear->parameters());
    REQUIRE(optimizer.size() == linear->parameters().size());
  }
}

TEST_CASE("Optim/XORConvergence/SGD") {
  REQUIRE(test_optimizer_xor<SGD>(
      SGDOptions(0.1).momentum(0.9).nesterov(true).weight_decay(1e-6)));
}

TEST_CASE("Optim/XORConvergence/Adagrad") {
  REQUIRE(test_optimizer_xor<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3)));
}

TEST_CASE("Optim/XORConvergence/RMSprop") {
  REQUIRE(test_optimizer_xor<RMSprop>(RMSpropOptions(0.1).centered(true)));
}

TEST_CASE("Optim/XORConvergence/RMSpropWithMomentum") {
  REQUIRE(test_optimizer_xor<RMSprop>(
      RMSpropOptions(0.1).momentum(0.9).weight_decay(1e-6)));
}

TEST_CASE("Optim/XORConvergence/Adam") {
  REQUIRE(test_optimizer_xor<Adam>(AdamOptions(0.1).weight_decay(1e-6)));
}

TEST_CASE("Optim/XORConvergence/AdamWithAmsgrad") {
  REQUIRE(test_optimizer_xor<Adam>(
      AdamOptions(0.1).weight_decay(1e-6).amsgrad(true)));
}

TEST_CASE("Optim/ProducesPyTorchValues/Adam") {
  check_exact_values<Adam>(AdamOptions(1.0), expected_parameters::Adam);
}

TEST_CASE("Optim/ProducesPyTorchValues/AdamWithWeightDecay") {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adam_with_weight_decay);
}

TEST_CASE("Optim/ProducesPyTorchValues/AdamWithWeightDecayAndAMSGrad") {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-6).amsgrad(true),
      expected_parameters::Adam_with_weight_decay_and_amsgrad);
}

TEST_CASE("Optim/ProducesPyTorchValues/Adagrad") {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0), expected_parameters::Adagrad);
}

TEST_CASE("Optim/ProducesPyTorchValues/AdagradWithWeightDecay") {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adagrad_with_weight_decay);
}

TEST_CASE("Optim/ProducesPyTorchValues/AdagradWithWeightDecayAndLRDecay") {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3),
      expected_parameters::Adagrad_with_weight_decay_and_lr_decay);
}

TEST_CASE("Optim/ProducesPyTorchValues/RMSprop") {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1), expected_parameters::RMSprop);
}

TEST_CASE("Optim/ProducesPyTorchValues/RMSpropWithWeightDecay") {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-2),
      expected_parameters::RMSprop_with_weight_decay);
}

TEST_CASE("Optim/ProducesPyTorchValues/RMSpropWithWeightDecayAndCentered") {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true),
      expected_parameters::RMSprop_with_weight_decay_and_centered);
}

TEST_CASE(
    "Optim/ProducesPyTorchValues/RMSpropWithWeightDecayAndCenteredAndMomentum") {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true).momentum(0.9),
      expected_parameters::RMSprop_with_weight_decay_and_centered_and_momentum);
}

TEST_CASE("Optim/ProducesPyTorchValues/SGD") {
  check_exact_values<SGD>(SGDOptions(0.1), expected_parameters::SGD);
}

TEST_CASE("Optim/ProducesPyTorchValues/SGDWithWeightDecay") {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2),
      expected_parameters::SGD_with_weight_decay);
}

TEST_CASE("Optim/ProducesPyTorchValues/SGDWithWeightDecayAndMomentum") {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2).momentum(0.9),
      expected_parameters::SGD_with_weight_decay_and_momentum);
}

TEST_CASE("Optim/ProducesPyTorchValues/SGDWithWeightDecayAndNesterovMomentum") {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-6).momentum(0.9).nesterov(true),
      expected_parameters::SGD_with_weight_decay_and_nesterov_momentum);
}

TEST_CASE("Optim/ZeroGrad") {
  torch::manual_seed(0);

  Linear model(2, 8);
  SGD optimizer(model->parameters(), 0.1);

  for (const auto& parameter : model->parameters()) {
    REQUIRE(!parameter->grad().defined());
  }

  auto output = model->forward(torch::ones({5, 2}));
  auto loss = output.sum();
  loss.backward();

  for (const auto& parameter : model->parameters()) {
    REQUIRE(parameter->grad().defined());
    REQUIRE(parameter->grad().sum().toCFloat() > 0);
  }

  optimizer.zero_grad();

  for (const auto& parameter : model->parameters()) {
    REQUIRE(parameter->grad().defined());
    REQUIRE(parameter->grad().sum().toCFloat() == 0);
  }
}

TEST_CASE("Optim/ExternalVectorOfParameters") {
  torch::manual_seed(0);

  std::vector<torch::Tensor> parameters = {
      torch::randn({2, 2}), torch::randn({3, 3}), torch::randn({4, 4})};
  std::vector<torch::Tensor> original_parameters = {
      parameters[0].clone(), parameters[1].clone(), parameters[2].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.grad() = torch::ones_like(parameter);
  }

  SGD optimizer(parameters, 1.0);

  optimizer.step();

  REQUIRE(parameters[0].allclose(original_parameters[0] - 1.0));
  REQUIRE(parameters[1].allclose(original_parameters[1] - 1.0));
  REQUIRE(parameters[2].allclose(original_parameters[2] - 1.0));
}

TEST_CASE("Optim/AddParameter/LBFGS") {
  torch::manual_seed(0);

  std::vector<torch::Tensor> parameters = {torch::randn({5, 5})};
  std::vector<torch::Tensor> original_parameters = {parameters[0].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.grad() = torch::ones_like(parameter);
  }

  LBFGS optimizer(std::vector<torch::Tensor>{}, 1.0);
  optimizer.add_parameters(parameters);

  optimizer.step([]() { return torch::tensor(1); });

  // REQUIRE this doesn't throw
}
