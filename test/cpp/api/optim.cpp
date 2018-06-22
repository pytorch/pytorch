#include <catch.hpp>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/optim.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/util.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>

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

// TODO: Add test for zero_grad
// TODO: Add test for passing arbitrary vector of variables
// TODO: Add hard tests that verify deterministically the output of each
// optimization algorithm against the PyTorch algorithms (i.e. compare against a
// matrix of values after 1000 iterations)

TEST_CASE("optim") {
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
