#include <catch.hpp>

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;
using Catch::StartsWith;

struct AGIUnit : nn::Module {
  variable_list forward(variable_list) {
    return {};
  }
};

namespace test {
struct AGIUnit : nn::Module {
  variable_list forward(variable_list) {
    return {};
  }
};
struct AGIUnit2 : nn::Module {
  AGIUnit2() : nn::Module("Foo") {}
  variable_list forward(variable_list) {
    return {};
  }
};
} // namespace test

bool pointer_equal(Tensor first, Tensor second) {
  return first.data<float>() == second.data<float>();
}

TEST_CASE("module/training-mode") {
  auto model = Linear(3, 4).build();
  REQUIRE(model->is_training());
  SECTION("Enable eval mode") {
    model->eval();
    REQUIRE(!model->is_training());
  }
  SECTION("Enable train mode") {
    model->train();
    REQUIRE(model->is_training());
  }
}

TEST_CASE("module/zero-grad") {
  auto model = Linear(3, 4).build();
  auto weight = Var(at::ones(at::CPU(at::kFloat), {8, 3}));
  auto loss = model->forward({weight}).front().sum();
  backward(loss);
  for (auto& parameter : model->parameters()) {
    Variable grad = parameter.second.grad();
    REQUIRE(grad.defined());
    REQUIRE(grad.sum().toCFloat() != 0);
  }
  model->zero_grad();
  for (auto& parameter : model->parameters()) {
    Variable grad = parameter.second.grad();
    REQUIRE(grad.defined());
    REQUIRE(grad.sum().toCFloat() == 0);
  }
}

TEST_CASE("module/name") {
  // CHECK instead of REQUIRE because demangling may fail.
  AGIUnit agi;
  // Call it twice just to make sure there are no bugs in the lazy
  // initialization semantics.
  CHECK(agi.name() == "AGIUnit");
  CHECK(agi.name() == "AGIUnit");
  SECTION("correctly demangled") {
    CHECK(test::AGIUnit().name() == "test::AGIUnit");
    CHECK(test::AGIUnit2().name() == "Foo");
  }
}

TEST_CASE("module/conversions", "[cuda]") {
  auto model = LSTM(128, 64).layers(3).dropout(0.2).build();
  SECTION("starts as float on CPU") {
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCPU);
      REQUIRE(parameter.second.type().scalarType() == at::kFloat);
    }
  }
  SECTION("to(CUDA)") {
    model->cuda();
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCUDA);
    }
  }
  SECTION("to(CPU)") {
    model->to(at::kCPU);
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCPU);
    }
  }
  SECTION("to(Int)") {
    model->to(at::kInt);
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().scalarType() == at::kInt);
    }
  }
  SECTION("to(Double)") {
    model->to(at::kDouble);
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().scalarType() == at::kDouble);
    }
  }
  SECTION("to(CUDA(Float))") {
    model->to(at::CUDA(at::kFloat));
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCUDA);
      REQUIRE(parameter.second.type().scalarType() == at::kFloat);
    }
  }
}

TEST_CASE("module/clone") {
  SECTION(
      "a module that does not override clone() throws when clone() is called") {
    struct UnCloneable : Module {
      variable_list forward(variable_list) override {
        return {};
      }
    };
    UnCloneable module;
    REQUIRE_THROWS_WITH(
        module.clone(), StartsWith("clone() has not been implemented"));
  }

  SECTION(
      "a module that overrides clone() does not throw when clone() is called ") {
    struct Cloneable : Module {
      variable_list forward(variable_list) override {
        return {};
      }
      std::shared_ptr<Module> clone() const override {
        return nullptr;
      }
    };
    Cloneable module;
    REQUIRE_NOTHROW(module.clone());
  }

  SECTION("Cloning creates distinct parameters") {
    struct TestModel : public CloneableModule<TestModel> {
      TestModel() {
        register_module("l1", &TestModel::l1, Linear(10, 3).build());
        register_module("l2", &TestModel::l2, Linear(3, 5).build());
        register_module("l3", &TestModel::l3, Linear(5, 100).build());
      }

      void reset() override {}

      variable_list forward(variable_list input) override {
        return input;
      }

      std::shared_ptr<Linear> l1, l2, l3;
    };

    auto model = TestModel().build();

    auto model2 = model->clone();
    auto m1param = model->parameters();
    auto m2param = model2->parameters();
    for (auto& param : m1param) {
      REQUIRE(!pointer_equal(param.second, m2param[param.first]));
      REQUIRE(param.second.allclose(m2param[param.first]));
      param.second.data().mul_(2);
    }
    for (auto& param : m1param) {
      REQUIRE(!param.second.allclose(m2param[param.first]));
    }
  }

  SECTION("Cloning preserves external references") {
    struct TestModel : public CloneableModule<TestModel> {
      void reset() {
        register_parameter(
            "weight",
            &TestModel::weight,
            at::ones(at::CPU(at::kFloat), {4, 4}));
      }

      variable_list forward(variable_list input) override {
        return input;
      }

      Variable weight;
    };

    auto model = TestModel().build();
    REQUIRE(pointer_equal(model->weight, model->param("weight")));

    auto model2 = std::dynamic_pointer_cast<TestModel>(
        std::shared_ptr<Module>(model->clone()));
    REQUIRE(!pointer_equal(model2->weight, model->weight));
    REQUIRE(pointer_equal(model2->weight, model2->param("weight")));
    REQUIRE(!pointer_equal(model2->weight, model->param("weight")));
  }
}

TEST_CASE("module/parameters") {
  struct TestModule : Module {
    TestModule() {
      register_parameter(
          "a", &TestModule::a, at::zeros(at::CPU(at::kFloat), {2, 2}));
      register_parameter(
          "b", &TestModule::b, at::ones(at::CPU(at::kFloat), {2, 2}));
      register_parameter(
          "c", &TestModule::c, at::ones(at::CPU(at::kFloat), {2, 2}) * 2);
    }

    variable_list forward(variable_list) override {
      return {};
    }

    Variable a, b, c;
  };

  TestModule module;

  SECTION("has correct number of parameters") {
    REQUIRE(module.parameters().size() == 3);
  }

  SECTION("contains parameters with the correct name") {
    auto parameters = module.parameters();
    REQUIRE(parameters.count("a"));
    REQUIRE(parameters.count("b"));
    REQUIRE(parameters.count("c"));
  }
}
