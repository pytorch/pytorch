#include <catch.hpp>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/optim.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/util.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <random>

using namespace torch::nn;
using namespace torch::optim;

bool test_optimizer_xor(
    std::shared_ptr<torch::optim::OptimizerImpl> optimizer,
    std::shared_ptr<Sequential> model) {
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
    // forward
    std::function<at::Scalar()> closure = [&] {
      optimizer->zero_grad();
      auto x = model->forward(inp);
      torch::Tensor loss = at::binary_cross_entropy(x, lab);
      loss.backward();
      return at::Scalar(loss.data());
    };

    auto loss = optim->optimizer(closure);

    running_loss = running_loss * 0.99 + loss.toFloat() * 0.01;
    if (epoch > 3000) {
      return false;
    }
    epoch++;
  }
  return true;
}

TEST_CASE("optim") {
  std::srand(0);
  torch::manual_seed(0);
  auto model = std::make_shared<Sequential>(
      torch::SigmoidLinear(Linear(2, 8)), torch::SigmoidLinear(Linear(8, 1)));

  // Flaky
  // SECTION("lbfgs") {
  //   auto optim = optim::LBFGS(model, 5e-2).max_iter(5).make();
  //   REQUIRE(test_optimizer_xor(optim, model));
  // }

  SECTION("sgd") {
    auto optimizer = torch::optim::SGD(model, 1e-1)
                         .momentum(0.9)
                         .nesterov()
                         .weight_decay(1e-6)
                         .make();
    REQUIRE(test_optimizer_xor(optimizer, model));
  }

  SECTION("adagrad") {
    auto optimizer = torch::optim::Adagrad(model, 1.0)
                         .weight_decay(1e-6)
                         .lr_decay(1e-3)
                         .make();
    REQUIRE(test_optimizer_xor(optimizer, model));
  }

  SECTION("rmsprop_simple") {
    auto optimizer = torch::optim::RMSprop(model, 1e-1).centered().make();
    REQUIRE(test_optimizer_xor(optimizer, model));
  }

  SECTION("rmsprop") {
    auto optimizer = torch::optim::RMSprop(model, 1e-1)
                         .momentum(0.9)
                         .weight_decay(1e-6)
                         .make();
    REQUIRE(test_optimizer_xor(optimizer, model));
  }

  /*
  // This test appears to be flaky, see
  https://github.com/pytorch/pytorch/issues/7288 SECTION("adam") { auto
  optimizer = optim::Adam(model, 1.0).weight_decay(1e-6).make();
  REQUIRE(test_optimizer_xor(optimizer, model));
  }
  */

  SECTION("amsgrad") {
    auto optimizer =
        torch::optim::Adam(model, 0.1).weight_decay(1e-6).amsgrad().make();
    REQUIRE(test_optimizer_xor(optimizer, model));
  }
}
