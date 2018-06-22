#include <catch.hpp>

#include <torch/functions.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
#include <torch/tensor.h>

#include <test/cpp/api/util.h>

using namespace torch;
using namespace torch::nn;

class TestModel : public Module {
 public:
  TestModel() {
    l1 = register_module("l1", Linear(10, 3));
    l2 = register_module("l2", Linear(3, 5));
    l3 = register_module("l3", Linear(5, 100));
  }

  std::vector<Variable> forward(std::vector<Variable> input) {
    return input;
  }

  Linear l1, l2, l3;
};

class NestedModel : public Module {
 public:
  NestedModel() {
    l1 = register_module("l1", Linear(5, 20));
    t = register_module("test", std::make_shared<TestModel>());
    param_ = register_parameter("param", torch::empty({3, 2, 21}));
  }

  std::vector<Variable> forward(std::vector<Variable> input) {
    return input;
  };

  Variable param_;
  Linear l1;
  std::shared_ptr<TestModel> t;
};

TEST_CASE("modules") {
  SECTION("conv") {
    SECTION("1d") {
      Conv1d model(Conv1dOptions(3, 2, 3).stride(2));
      auto x = torch::randn({2, 3, 5}, at::requires_grad());
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      s.backward();
      REQUIRE(y.ndimension() == 3);
      REQUIRE(s.ndimension() == 0);
      for (auto i = 0; i < 3; i++) {
        REQUIRE(y.size(i) == 2);
      }

      REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3);
    }
    SECTION("2d") {
      SECTION("even") {
        Conv2d model(Conv2dOptions(3, 2, 3).stride(2));
        auto x = torch::randn({2, 3, 5, 5}, at::requires_grad());
        auto y = model->forward({x})[0];
        Variable s = y.sum();

        s.backward();
        REQUIRE(y.ndimension() == 4);
        REQUIRE(s.ndimension() == 0);
        for (auto i = 0; i < 4; i++) {
          REQUIRE(y.size(i) == 2);
        }

        REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3);
      }

      SECTION("uneven") {
        Conv2d model(Conv2dOptions(3, 2, {3, 2}).stride({2, 2}));
        auto x = torch::randn({2, 3, 5, 4}, at::requires_grad());
        auto y = model->forward({x})[0];
        Variable s = y.sum();

        s.backward();
        REQUIRE(y.ndimension() == 4);
        REQUIRE(s.ndimension() == 0);
        for (auto i = 0; i < 4; i++) {
          REQUIRE(y.size(i) == 2);
        }

        REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 2);
      }
    }
    SECTION("3d") {
      Conv3d model(Conv3dOptions(3, 2, 3).stride(2));
      auto x = torch::randn({2, 3, 5, 5, 5}, at::requires_grad());
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      s.backward();
      REQUIRE(y.ndimension() == 5);
      REQUIRE(s.ndimension() == 0);
      for (auto i = 0; i < 5; i++) {
        REQUIRE(y.size(i) == 2);
      }

      REQUIRE(
          model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3 * 3);
    }
  }
  SECTION("linear") {
    SECTION("basic1") {
      Linear model(5, 2);
      auto x = torch::randn({10, 5}, at::requires_grad());
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      s.backward();
      REQUIRE(y.ndimension() == 2);
      REQUIRE(s.ndimension() == 0);
      REQUIRE(y.size(0) == 10);
      REQUIRE(y.size(1) == 2);

      REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
    }
  }

  SECTION("simple") {
    auto model = std::make_shared<SimpleContainer>();
    auto l1 = model->add(Linear(10, 3), "l1");
    auto l2 = model->add(Linear(3, 5), "l2");
    auto l3 = model->add(Linear(5, 100), "l3");

    auto x = torch::randn({1000, 10}, at::requires_grad());
    x = l1->forward({x})[0].clamp_min(0);
    x = l2->forward({x})[0].clamp_min(0);
    x = l3->forward({x})[0].clamp_min(0);

    x.backward();
    REQUIRE(x.ndimension() == 2);
    REQUIRE(x.size(0) == 1000);
    REQUIRE(x.size(1) == 100);
    REQUIRE(x.data().min().toCFloat() == 0);
  }

  SECTION("embedding") {
    SECTION("basic") {
      int dict_size = 10;
      Embedding model(dict_size, 2);
      // Cannot get gradients to change indices (input) - only for embedding
      // params
      auto x = torch::full({10}, dict_size - 1, torch::kInt64);
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      s.backward();
      REQUIRE(y.ndimension() == 2);
      REQUIRE(s.ndimension() == 0);
      REQUIRE(y.size(0) == 10);
      REQUIRE(y.size(1) == 2);

      REQUIRE(model->parameters()["table"].grad().numel() == 2 * dict_size);
    }

    SECTION("list") {
      Embedding model(6, 4);
      auto x = torch::full({2, 3}, 5, torch::kInt64);
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      s.backward();
      REQUIRE(y.ndimension() == 3);
      REQUIRE(y.size(0) == 2);
      REQUIRE(y.size(1) == 3);
      REQUIRE(y.size(2) == 4);
    }
  }

  SECTION("dropout") {
    Dropout dropout(0.5);
    Variable x = torch::ones(100, at::requires_grad());
    Variable y = dropout->forward({x})[0];

    y.backward();
    REQUIRE(y.ndimension() == 1);
    REQUIRE(y.size(0) == 100);
    // TODO: These two tests are flaky
    // https://github.com/pytorch/pytorch/issues/7286
    // REQUIRE(y.sum().toCFloat() < 130); // Probably
    // REQUIRE(y.sum().toCFloat() > 70); // Probably

    dropout->eval();
    y = dropout->forward({x})[0];
    REQUIRE(y.data().sum().toCFloat() == 100);
  }

  SECTION("param") {
    auto model = std::make_shared<NestedModel>();
    auto parameters = model->parameters();
    REQUIRE(parameters["param"].size(0) == 3);
    REQUIRE(parameters["param"].size(1) == 2);
    REQUIRE(parameters["param"].size(2) == 21);
    REQUIRE(parameters["l1.bias"].size(0) == 20);
    REQUIRE(parameters["l1.weight"].size(0) == 20);
    REQUIRE(parameters["l1.weight"].size(1) == 5);
    REQUIRE(parameters["test.l1.bias"].size(0) == 3);
    REQUIRE(parameters["test.l1.weight"].size(0) == 3);
    REQUIRE(parameters["test.l1.weight"].size(1) == 10);
    REQUIRE(parameters["test.l2.bias"].size(0) == 5);
    REQUIRE(parameters["test.l2.weight"].size(0) == 5);
    REQUIRE(parameters["test.l2.weight"].size(1) == 3);
    REQUIRE(parameters["test.l3.bias"].size(0) == 100);
    REQUIRE(parameters["test.l3.weight"].size(0) == 100);
    REQUIRE(parameters["test.l3.weight"].size(1) == 5);
  }

  SECTION("functional") {
    bool was_called = false;
    // clang-format off
    auto functional = Functional([&was_called](std::vector<Variable> input) {
      was_called = true;
      return input;
    });
    // clang-format on
    auto output = functional->forward({torch::ones(5, at::requires_grad())});
    REQUIRE(was_called);
    REQUIRE(output.size() == 1);
    REQUIRE(output.front().equal(torch::ones(5, at::requires_grad())));
  }
}

TEST_CASE("modules_cuda", "[cuda]") {
  SECTION("1") {
    Linear model(5, 2);
    model->cuda();
    auto x = torch::randn({10, 5}, at::device(at::kCUDA).requires_grad(true));
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    s.backward();
    REQUIRE(y.ndimension() == 2);
    REQUIRE(s.ndimension() == 0);
    REQUIRE(y.size(0) == 10);
    REQUIRE(y.size(1) == 2);

    REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
  }

  SECTION("2") {
    Linear model(5, 2);
    model->cuda();
    model->cpu();
    auto x = torch::randn({10, 5}, at::requires_grad());
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    s.backward();
    REQUIRE(y.ndimension() == 2);
    REQUIRE(s.ndimension() == 0);
    REQUIRE(y.size(0) == 10);
    REQUIRE(y.size(1) == 2);

    REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
  }
}
