#include <catch.hpp>

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

bool test_optimizer_xor(Optimizer optim, std::shared_ptr<ContainerList> model) {
  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    int64_t bs = 4;
    auto inp = at::CPU(at::kFloat).tensor({bs, 2});
    auto lab = at::CPU(at::kFloat).tensor({bs});
    for (size_t i = 0; i < bs; i++) {
      const int64_t a = std::rand() % 2;
      const int64_t b = std::rand() % 2;
      const int64_t c = static_cast<uint64_t>(a) ^ static_cast<uint64_t>(b);
      inp[i][0] = a;
      inp[i][1] = b;
      lab[i] = c;
    }
    // forward
    auto x = Var(inp);
    auto y = Var(lab, false);
    for (auto& layer : *model)
      x = layer->forward({x})[0].sigmoid_();
    Variable loss = at::binary_cross_entropy(x, y);

    optim->zero_grad();
    backward(loss);
    optim->step();

    running_loss = running_loss * 0.99 + loss.data().sum().toCFloat() * 0.01;
    if (epoch > 3000) {
      return false;
    }
    epoch++;
  }
  return true;
}

TEST_CASE("optim") {
  ContainerList list;
  list.append(make(Linear(2, 8)));
  list.append(make(Linear(8, 1)));
  auto model = make(list);

  SECTION("sgd") {
    auto optim =
        SGD(model, 1e-1).momentum(0.9).nesterov().weight_decay(1e-6).make();
    REQUIRE(test_optimizer_xor(optim, model));
  }

  SECTION("adagrad") {
    auto optim = Adagrad(model, 1.0).weight_decay(1e-6).lr_decay(1e-3).make();
    REQUIRE(test_optimizer_xor(optim, model));
  }

  SECTION("rmsprop_simple") {
    auto optim = RMSprop(model, 1e-1).centered().make();
    REQUIRE(test_optimizer_xor(optim, model));
  }

  SECTION("rmsprop") {
    auto optim = RMSprop(model, 1e-1).momentum(0.9).weight_decay(1e-6).make();
    REQUIRE(test_optimizer_xor(optim, model));
  }

  /*
  // This test appears to be flaky, see
  https://github.com/pytorch/pytorch/issues/7288 SECTION("adam") { auto optim =
  Adam(model, 1.0).weight_decay(1e-6).make(); REQUIRE(test_optimizer_xor(optim,
  model));
  }
  */

  SECTION("amsgrad") {
    auto optim = Adam(model, 0.1).weight_decay(1e-6).amsgrad().make();
    REQUIRE(test_optimizer_xor(optim, model));
  }
}
