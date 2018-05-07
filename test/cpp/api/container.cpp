#include <catch.hpp>

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

class TestModel : public CloneableModule<TestModel> {
 public:
  void initialize_containers() override {
    add(make(Linear(10, 3)), "l1");
    add(make(Linear(3, 5)), "l2");
    add(make(Linear(5, 100)), "l3");
  }

  variable_list forward(variable_list input) override {
    return input;
  };
};

class NestedModel : public CloneableModule<NestedModel> {
 public:
  void initialize_containers() override {
    add(make(Linear(5, 20)), "l1");
    add(make(TestModel()), "test");
  }

  void initialize_parameters() override {
    add(Var(DefaultTensor(at::kFloat).tensor({3, 2, 21}), false), "param");
  }

  variable_list forward(variable_list input) override {
    return input;
  };
};

TEST_CASE("containers") {
  SECTION("conv") {
    SECTION("1d") {
      auto model = make(Conv1d(3, 2, 3).stride(2));
      auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5}), true);
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      backward(s);
      REQUIRE(y.ndimension() == 4);
      REQUIRE(s.ndimension() == 0);
      for (auto i = 0; i < 3; i++) {
        REQUIRE(y.size(i) == 2);
      }

      REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3);
    }

    SECTION("2d") {
      SECTION("even") {
        auto model = make(Conv2d(3, 2, 3).stride(2));
        auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 5}), true);
        auto y = model->forward({x})[0];
        Variable s = y.sum();

        backward(s);
        REQUIRE(y.ndimension() == 4);
        REQUIRE(s.ndimension() == 0);
        for (auto i = 0; i < 4; i++) {
          REQUIRE(y.size(i) == 2);
        }

        REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3);
      }

      SECTION("uneven") {
        auto model = make(Conv2d(3, 2, IntVec({3, 2})).stride(2));
        auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 4}), true);
        auto y = model->forward({x})[0];
        Variable s = y.sum();

        backward(s);
        REQUIRE(y.ndimension() == 4);
        REQUIRE(s.ndimension() == 0);
        for (auto i = 0; i < 4; i++) {
          REQUIRE(y.size(i) == 2);
        }

        REQUIRE(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 2);
      }
    }

    SECTION("3d") {
      auto model = make(Conv3d(3, 2, 3).stride(2));
      auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 5, 5}), true);
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      backward(s);
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
      auto model = make(Linear(5, 2));
      auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      backward(s);
      REQUIRE(y.ndimension() == 2);
      REQUIRE(s.ndimension() == 0);
      REQUIRE(y.size(0) == 10);
      REQUIRE(y.size(1) == 2);

      REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
    }

    SECTION("sequential") {
      auto model = make(ContainerList()
                            .append(make(Linear(10, 3)))
                            .append(make(Linear(3, 5)))
                            .append(make(Linear(5, 100))));

      auto x = Var(at::CPU(at::kFloat).randn({1000, 10}));
      for (auto layer : *model) {
        x = layer->forward({x})[0];
        x = x.clamp_min(0); // relu
      }

      backward(x);
      REQUIRE(x.ndimension() == 2);
      REQUIRE(x.size(0) == 1000);
      REQUIRE(x.size(1) == 100);
      REQUIRE(x.data().min().toCFloat() == 0);
    }

    SECTION("simple") {
      auto model = make(SimpleContainer());
      auto l1 = model->add(make(Linear(10, 3)), "l1");
      auto l2 = model->add(make(Linear(3, 5)), "l2");
      auto l3 = model->add(make(Linear(5, 100)), "l3");

      auto x = Var(at::CPU(at::kFloat).randn({1000, 10}));
      x = l1->forward({x})[0].clamp_min(0);
      x = l2->forward({x})[0].clamp_min(0);
      x = l3->forward({x})[0].clamp_min(0);

      backward(x);
      REQUIRE(x.ndimension() == 2);
      REQUIRE(x.size(0) == 1000);
      REQUIRE(x.size(1) == 100);
      REQUIRE(x.data().min().toCFloat() == 0);
    }
  }

  SECTION("clone") {
    auto model = make(TestModel());

    auto model2 = model->clone();
    auto m1param = model->parameters();
    auto m2param = model2->parameters();
    for (auto& param : m1param) {
      REQUIRE(param.second.allclose(m2param[param.first]));
      param.second.data().mul_(2);
    }
    for (auto& param : m1param) {
      REQUIRE(!param.second.allclose(m2param[param.first]));
    }
  }

  SECTION("embedding") {
    SECTION("basic") {
      int dict_size = 10;
      auto model = make(Embedding(dict_size, 2));
      // Cannot get gradients to change indices (input) - only for embedding
      // params
      auto x = Var(at::CPU(at::kLong).tensor({10}).fill_(dict_size - 1), false);
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      backward(s);
      REQUIRE(y.ndimension() == 2);
      REQUIRE(s.ndimension() == 0);
      REQUIRE(y.size(0) == 10);
      REQUIRE(y.size(1) == 2);

      REQUIRE(model->parameters()["weight"].grad().numel() == 2 * dict_size);
    }

    SECTION("list") {
      auto model = make(Embedding(6, 4));
      auto x = Var(at::CPU(at::kLong).tensor({2, 3}).fill_(5), false);
      auto y = model->forward({x})[0];
      Variable s = y.sum();

      backward(s);
      REQUIRE(y.ndimension() == 3);
      REQUIRE(y.size(0) == 2);
      REQUIRE(y.size(1) == 3);
      REQUIRE(y.size(2) == 4);
    }
  }

  SECTION("dropout") {
    auto dropout = make(Dropout(0.5));
    Variable x = Var(at::CPU(at::kFloat).ones(100));
    Variable y = dropout->forward({x})[0];

    backward(y);
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
    auto model = make(NestedModel());
    REQUIRE(model->param("param").size(0) == 3);
    REQUIRE(model->param("param").size(1) == 2);
    REQUIRE(model->param("param").size(2) == 21);
    REQUIRE(model->param("l1.bias").size(0) == 20);
    REQUIRE(model->param("l1.weight").size(0) == 20);
    REQUIRE(model->param("l1.weight").size(1) == 5);
    REQUIRE(model->param("test.l1.bias").size(0) == 3);
    REQUIRE(model->param("test.l1.weight").size(0) == 3);
    REQUIRE(model->param("test.l1.weight").size(1) == 10);
    REQUIRE(model->param("test.l2.bias").size(0) == 5);
    REQUIRE(model->param("test.l2.weight").size(0) == 5);
    REQUIRE(model->param("test.l2.weight").size(1) == 3);
    REQUIRE(model->param("test.l3.bias").size(0) == 100);
    REQUIRE(model->param("test.l3.weight").size(0) == 100);
    REQUIRE(model->param("test.l3.weight").size(1) == 5);
  }
}

TEST_CASE("containers_cuda", "[cuda]") {
  SECTION("1") {
    auto model = make(Linear(5, 2));
    model->cuda();
    auto x = Var(at::CUDA(at::kFloat).randn({10, 5}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    backward(s);
    REQUIRE(y.ndimension() == 2);
    REQUIRE(s.ndimension() == 0);
    REQUIRE(y.size(0) == 10);
    REQUIRE(y.size(1) == 2);

    REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
  }

  SECTION("2") {
    auto model = make(Linear(5, 2));
    model->cuda();
    model->cpu();
    auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    backward(s);
    REQUIRE(y.ndimension() == 2);
    REQUIRE(s.ndimension() == 0);
    REQUIRE(y.size(0) == 10);
    REQUIRE(y.size(1) == 2);

    REQUIRE(model->parameters()["weight"].grad().numel() == 2 * 5);
  }
}
