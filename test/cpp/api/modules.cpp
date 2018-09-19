#include "catch_utils.hpp"

#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/util.h>

using Catch::StartsWith;

using namespace torch::nn;
using namespace torch::test;

class TestModel : public torch::nn::Module {
 public:
  TestModel()
      : l1(register_module("l1", Linear(10, 3))),
        l2(register_module("l2", Linear(3, 5))),
        l3(register_module("l3", Linear(5, 100))) {}

  Linear l1, l2, l3;
};

class NestedModel : public torch::nn::Module {
 public:
  NestedModel()
      : param_(register_parameter("param", torch::empty({3, 2, 21}))),
        l1(register_module("l1", Linear(5, 20))),
        t(register_module("test", std::make_shared<TestModel>())) {}

  torch::Tensor param_;
  Linear l1;
  std::shared_ptr<TestModel> t;
};

CATCH_TEST_CASE("modules") {
  torch::manual_seed(0);
  CATCH_SECTION("conv") {
    CATCH_SECTION("1d") {
      Conv1d model(Conv1dOptions(3, 2, 3).stride(2));
      auto x = torch::randn({2, 3, 5}, torch::requires_grad());
      auto y = model->forward(x);
      torch::Tensor s = y.sum();

      s.backward();
      CATCH_REQUIRE(y.ndimension() == 3);
      CATCH_REQUIRE(s.ndimension() == 0);
      for (auto i = 0; i < 3; i++) {
        CATCH_REQUIRE(y.size(i) == 2);
      }

      CATCH_REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3);
    }
    CATCH_SECTION("2d") {
      CATCH_SECTION("even") {
        Conv2d model(Conv2dOptions(3, 2, 3).stride(2));
        auto x = torch::randn({2, 3, 5, 5}, torch::requires_grad());
        auto y = model->forward(x);
        torch::Tensor s = y.sum();

        s.backward();
        CATCH_REQUIRE(y.ndimension() == 4);
        CATCH_REQUIRE(s.ndimension() == 0);
        for (auto i = 0; i < 4; i++) {
          CATCH_REQUIRE(y.size(i) == 2);
        }

        CATCH_REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3);
      }

      CATCH_SECTION("uneven") {
        Conv2d model(Conv2dOptions(3, 2, {3, 2}).stride({2, 2}));
        auto x = torch::randn({2, 3, 5, 4}, torch::requires_grad());
        auto y = model->forward(x);
        torch::Tensor s = y.sum();

        s.backward();
        CATCH_REQUIRE(y.ndimension() == 4);
        CATCH_REQUIRE(s.ndimension() == 0);
        for (auto i = 0; i < 4; i++) {
          CATCH_REQUIRE(y.size(i) == 2);
        }

        CATCH_REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 2);
      }
    }
    CATCH_SECTION("3d") {
      Conv3d model(Conv3dOptions(3, 2, 3).stride(2));
      auto x = torch::randn({2, 3, 5, 5, 5}, torch::requires_grad());
      auto y = model->forward(x);
      torch::Tensor s = y.sum();

      s.backward();
      CATCH_REQUIRE(y.ndimension() == 5);
      CATCH_REQUIRE(s.ndimension() == 0);
      for (auto i = 0; i < 5; i++) {
        CATCH_REQUIRE(y.size(i) == 2);
      }

      CATCH_REQUIRE(
          model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3 * 3);
    }
  }
  CATCH_SECTION("linear") {
    CATCH_SECTION("basic1") {
      Linear model(5, 2);
      auto x = torch::randn({10, 5}, torch::requires_grad());
      auto y = model->forward(x);
      torch::Tensor s = y.sum();

      s.backward();
      CATCH_REQUIRE(y.ndimension() == 2);
      CATCH_REQUIRE(s.ndimension() == 0);
      CATCH_REQUIRE(y.size(0) == 10);
      CATCH_REQUIRE(y.size(1) == 2);

      CATCH_REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
    }
  }

  CATCH_SECTION("simple") {
    auto model = std::make_shared<SimpleContainer>();
    auto l1 = model->add(Linear(10, 3), "l1");
    auto l2 = model->add(Linear(3, 5), "l2");
    auto l3 = model->add(Linear(5, 100), "l3");

    auto x = torch::randn({1000, 10}, torch::requires_grad());
    x = l1->forward(x).clamp_min(0);
    x = l2->forward(x).clamp_min(0);
    x = l3->forward(x).clamp_min(0);

    x.backward();
    CATCH_REQUIRE(x.ndimension() == 2);
    CATCH_REQUIRE(x.size(0) == 1000);
    CATCH_REQUIRE(x.size(1) == 100);
    CATCH_REQUIRE(x.min().toCFloat() == 0);
  }

  CATCH_SECTION("embedding") {
    CATCH_SECTION("basic") {
      const int64_t dict_size = 10;
      Embedding model(dict_size, 2);
      CATCH_REQUIRE(model->parameters().contains("weight"));
      CATCH_REQUIRE(model->weight.ndimension() == 2);
      CATCH_REQUIRE(model->weight.size(0) == dict_size);
      CATCH_REQUIRE(model->weight.size(1) == 2);

      // Cannot get gradients to change indices (input) - only for embedding
      // params
      auto x = torch::full({10}, dict_size - 1, torch::kInt64);
      auto y = model->forward(x);
      torch::Tensor s = y.sum();

      s.backward();
      CATCH_REQUIRE(y.ndimension() == 2);
      CATCH_REQUIRE(s.ndimension() == 0);
      CATCH_REQUIRE(y.size(0) == 10);
      CATCH_REQUIRE(y.size(1) == 2);

      CATCH_REQUIRE(model->parameters()["weight"].grad().numel() == 2 * dict_size);
    }

    CATCH_SECTION("list") {
      Embedding model(6, 4);
      auto x = torch::full({2, 3}, 5, torch::kInt64);
      auto y = model->forward(x);
      torch::Tensor s = y.sum();

      s.backward();
      CATCH_REQUIRE(y.ndimension() == 3);
      CATCH_REQUIRE(y.size(0) == 2);
      CATCH_REQUIRE(y.size(1) == 3);
      CATCH_REQUIRE(y.size(2) == 4);
    }
  }

  CATCH_SECTION("dropout") {
    Dropout dropout(0.5);
    torch::Tensor x = torch::ones(100, torch::requires_grad());
    torch::Tensor y = dropout->forward(x);

    y.backward();
    CATCH_REQUIRE(y.ndimension() == 1);
    CATCH_REQUIRE(y.size(0) == 100);
    CATCH_REQUIRE(y.sum().toCFloat() < 130); // Probably
    CATCH_REQUIRE(y.sum().toCFloat() > 70); // Probably

    dropout->eval();
    y = dropout->forward(x);
    CATCH_REQUIRE(y.sum().toCFloat() == 100);
  }

  CATCH_SECTION("param") {
    auto model = std::make_shared<NestedModel>();
    auto parameters = model->parameters();
    CATCH_REQUIRE(parameters["param"].size(0) == 3);
    CATCH_REQUIRE(parameters["param"].size(1) == 2);
    CATCH_REQUIRE(parameters["param"].size(2) == 21);
    CATCH_REQUIRE(parameters["l1.bias"].size(0) == 20);
    CATCH_REQUIRE(parameters["l1.weight"].size(0) == 20);
    CATCH_REQUIRE(parameters["l1.weight"].size(1) == 5);
    CATCH_REQUIRE(parameters["test.l1.bias"].size(0) == 3);
    CATCH_REQUIRE(parameters["test.l1.weight"].size(0) == 3);
    CATCH_REQUIRE(parameters["test.l1.weight"].size(1) == 10);
    CATCH_REQUIRE(parameters["test.l2.bias"].size(0) == 5);
    CATCH_REQUIRE(parameters["test.l2.weight"].size(0) == 5);
    CATCH_REQUIRE(parameters["test.l2.weight"].size(1) == 3);
    CATCH_REQUIRE(parameters["test.l3.bias"].size(0) == 100);
    CATCH_REQUIRE(parameters["test.l3.weight"].size(0) == 100);
    CATCH_REQUIRE(parameters["test.l3.weight"].size(1) == 5);
  }

  CATCH_SECTION("functional") {
    {
      bool was_called = false;
      auto functional = Functional([&was_called](torch::Tensor input) {
        was_called = true;
        return input;
      });
      auto output = functional->forward(torch::ones(5, torch::requires_grad()));
      CATCH_REQUIRE(was_called);
      CATCH_REQUIRE(output.equal(torch::ones(5, torch::requires_grad())));

      was_called = false;
      // Use the call operator overload here.
      output = functional(torch::ones(5, torch::requires_grad()));
      CATCH_REQUIRE(was_called);
      CATCH_REQUIRE(output.equal(torch::ones(5, torch::requires_grad())));
    }
    {
      auto functional = Functional(torch::relu);
      CATCH_REQUIRE(functional(torch::ones({})).toCFloat() == 1);
      CATCH_REQUIRE(functional(torch::ones({})).toCFloat() == 1);
      CATCH_REQUIRE(functional(torch::ones({}) * -1).toCFloat() == 0);
    }
    {
      auto functional =
          Functional(torch::elu, /*alpha=*/1, /*scale=*/0, /*input_scale=*/1);
      CATCH_REQUIRE(functional(torch::ones({})).toCFloat() == 0);
    }
  }

  CATCH_SECTION("batchnorm") {
    {
      BatchNorm bn(5);

      // Is stateful by default.
      CATCH_REQUIRE(bn->options.stateful());

      CATCH_REQUIRE(bn->running_mean.defined());
      CATCH_REQUIRE(bn->running_mean.dim() == 1);
      CATCH_REQUIRE(bn->running_mean.size(0) == 5);

      CATCH_REQUIRE(bn->running_variance.defined());
      CATCH_REQUIRE(bn->running_variance.dim() == 1);
      CATCH_REQUIRE(bn->running_variance.size(0) == 5);

      // Is affine by default.
      CATCH_REQUIRE(bn->options.affine());

      CATCH_REQUIRE(bn->weight.defined());
      CATCH_REQUIRE(bn->weight.dim() == 1);
      CATCH_REQUIRE(bn->weight.size(0) == 5);

      CATCH_REQUIRE(bn->bias.defined());
      CATCH_REQUIRE(bn->bias.dim() == 1);
      CATCH_REQUIRE(bn->bias.size(0) == 5);
    }
    {
      BatchNorm bn(BatchNormOptions(5).stateful(false).affine(false));

      CATCH_REQUIRE(!bn->running_mean.defined());
      CATCH_REQUIRE(!bn->running_variance.defined());
      CATCH_REQUIRE(!bn->weight.defined());
      CATCH_REQUIRE(!bn->bias.defined());

      CATCH_REQUIRE_THROWS_WITH(
          bn->forward(torch::ones({2, 5})),
          StartsWith("Calling BatchNorm::forward is only permitted "
                     "when the 'stateful' option is true (was false). "
                     "Use BatchNorm::pure_forward instead."));
    }
    {
      BatchNorm bn(BatchNormOptions(5).affine(false));
      bn->eval();

      // Want to make sure we use the supplied values in `pure_forward` even if
      // we are stateful.
      auto input = torch::randn({2, 5});
      auto mean = torch::randn(5);
      auto variance = torch::rand(5);
      auto output = bn->pure_forward(input, mean, variance);
      auto expected =
          (input - mean) / torch::sqrt(variance + bn->options.eps());
      CATCH_REQUIRE(output.allclose(expected));
    }
  }
}

CATCH_TEST_CASE("modules_cuda", "[cuda]") {
  torch::manual_seed(0);
  CATCH_SECTION("1") {
    Linear model(5, 2);
    model->to(torch::kCUDA);
    auto x =
        torch::randn({10, 5}, torch::device(torch::kCUDA).requires_grad(true));
    auto y = model->forward(x);
    torch::Tensor s = y.sum();

    s.backward();
    CATCH_REQUIRE(y.ndimension() == 2);
    CATCH_REQUIRE(s.ndimension() == 0);
    CATCH_REQUIRE(y.size(0) == 10);
    CATCH_REQUIRE(y.size(1) == 2);

    CATCH_REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
  }

  CATCH_SECTION("2") {
    Linear model(5, 2);
    model->to(torch::kCUDA);
    model->to(torch::kCPU);
    auto x = torch::randn({10, 5}, torch::requires_grad());
    auto y = model->forward(x);
    torch::Tensor s = y.sum();

    s.backward();
    CATCH_REQUIRE(y.ndimension() == 2);
    CATCH_REQUIRE(s.ndimension() == 0);
    CATCH_REQUIRE(y.size(0) == 10);
    CATCH_REQUIRE(y.size(1) == 2);

    CATCH_REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
  }
}
