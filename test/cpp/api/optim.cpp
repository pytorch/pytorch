#include <catch.hpp>

#include <torch/nn/module.h>
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

bool test_optimizer_xor(
    torch::optim::Optimizer&& optimizer,
    Sequential& model) {
  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    int64_t bs = 4;
    auto inp = torch::empty({bs, 2});
    auto lab = torch::empty({bs});
    for (size_t i = 0; i < bs; i++) {
      const int64_t a = std::rand() % 2;
      const int64_t b = std::rand() % 2;
      const int64_t c = static_cast<uint64_t>(a) ^ static_cast<uint64_t>(b);
      inp[i][0] = a;
      inp[i][1] = b;
      lab[i] = c;
    }
    inp.set_requires_grad(true);
    optimizer.zero_grad();
    auto x = model.forward(inp);
    torch::Tensor loss = at::binary_cross_entropy(x, lab);
    loss.backward();

    optimizer.step();

    running_loss = running_loss * 0.99 + loss.toCFloat() * 0.01;
    if (epoch > 3000) {
      return false;
    }
    epoch++;
  }
  return true;
}

template <typename OptimizerClass, typename Options>
void check_exact_values(
    Options options,
    std::vector<std::vector<at::Tensor>> expected_parameters) {
  const size_t kIterations = 1001;
  const size_t kSampleEvery = 100;

  torch::manual_seed(0);
  Sequential model(SigmoidLinear(Linear(2, 3)), SigmoidLinear(Linear(3, 1)));
  model.to(torch::kFloat64);

  // Use exact input values because matching random values is hard.
  auto parameters = model.parameters();
  parameters.at("0.linear.weight").data().flatten() = at::tensor(
      {-0.2109, -0.4976, -0.1413, -0.3420, -0.2524, 0.6976}, torch::kFloat64);
  parameters.at("0.linear.bias").data() =
      at::tensor({-0.1085, -0.2979, 0.6892}, torch::kFloat64);
  parameters.at("1.linear.weight").data().flatten() =
      at::tensor({-0.0508, -0.3941, -0.2843}, torch::kFloat64);
  parameters.at("1.linear.bias").data() =
      at::tensor({-0.0711}, torch::kFloat64);

  auto optimizer = OptimizerClass(parameters, options);

  auto input = at::tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, torch::kFloat64)
                   .reshape({3, 2});

  for (size_t i = 0; i < kIterations; ++i) {
    optimizer.zero_grad();
    auto output = model.forward(torch::autograd::make_variable(input));
    auto loss = output.sum();
    loss.backward();

    optimizer.step();

    if (i % kSampleEvery == 0) {
      REQUIRE(
          expected_parameters.at(i / kSampleEvery).size() == parameters.size());
      for (size_t p = 0; p < parameters.size(); ++p) {
        REQUIRE(parameters.at(p)->defined());
        auto computed = parameters.at(p)->data().flatten();
        auto expected = expected_parameters.at(i / kSampleEvery).at(p);
        if (!computed.allclose(expected, /*rtol=*/1e-3, /*atol=*/1e-5)) {
          std::cout << "Iteration " << i << ": " << computed
                    << " != " << expected << " (parameter " << p << ")"
                    << std::endl;
          REQUIRE(false);
        }
      }
    }
  }
}

TEST_CASE("Optim/XORConvergence") {
  std::srand(0);
  torch::manual_seed(0);
  Sequential model(
      torch::SigmoidLinear(Linear(2, 8)), torch::SigmoidLinear(Linear(8, 1)));

  SECTION("sgd") {
    REQUIRE(test_optimizer_xor(
        optim::SGD(
            model.parameters(),
            SGDOptions(1e-1).momentum(0.9).nesterov(true).weight_decay(1e-6)),
        model));
  }

  // // Flaky
  SECTION("lbfgs") {
    auto optimizer = LBFGS(model.parameters(), LBFGSOptions(5e-2).max_iter(5));
    // REQUIRE(test_optimizer_xor(optimizer, model));
  }

  SECTION("adagrad") {
    REQUIRE(test_optimizer_xor(
        optim::Adagrad(
            model.parameters(),
            AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3)),
        model));
  }

  SECTION("rmsprop_simple") {
    REQUIRE(test_optimizer_xor(
        RMSprop(model.parameters(), RMSpropOptions(1e-1).centered(true)),
        model));
  }

  SECTION("rmsprop") {
    REQUIRE(test_optimizer_xor(
        RMSprop(
            model.parameters(),
            RMSpropOptions(1e-1).momentum(0.9).weight_decay(1e-6)),
        model));
  }

  // This test appears to be flaky, see
  // https://github.com/pytorch/pytorch/issues/7288
  SECTION("adam") {
    REQUIRE(test_optimizer_xor(
        optim::Adam(model.parameters(), AdamOptions(1.0).weight_decay(1e-6)),
        model));
  }

  SECTION("amsgrad") {
    REQUIRE(test_optimizer_xor(
        optim::Adam(
            model.parameters(),
            AdamOptions(0.1).weight_decay(1e-6).amsgrad(true)),
        model));
  }
}

TEST_CASE("Optim/ProducesPyTorchValues/Adam") {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-6), expected_parameters::Adam);
}

TEST_CASE("Optim/ProducesPyTorchValues/Adagrad") {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3),
      expected_parameters::Adagrad);
}

TEST_CASE("Optim/ProducesPyTorchValues/RMSprop") {
  check_exact_values<RMSprop>(
      RMSpropOptions(1e-1).momentum(0.9).weight_decay(1e-6),
      expected_parameters::RMSprop);
}

TEST_CASE("Optim/ProducesPyTorchValues/SGD") {
  check_exact_values<SGD>(
      SGDOptions(1e-1).momentum(0.9).weight_decay(1e-6),
      expected_parameters::SGD);
}

TEST_CASE("Optim/ZeroGrad") {
  Linear model(2, 8);
  SGD optimizer(model->parameters(), 0.1);

  for (const auto& parameter : model->parameters()) {
    REQUIRE(!parameter->grad().defined());
  }

  auto output = model->forward({torch::ones({5, 2})}).front();
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
  std::vector<Variable> parameters = {
      torch::randn({2, 2}), torch::randn({3, 3}), torch::randn({4, 4})};
  std::vector<Variable> original_parameters = {
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
