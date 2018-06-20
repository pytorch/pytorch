#include <catch.hpp>

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

using Catch::StartsWith;

struct AGIUnit : nn::Module {};

namespace test {
struct AGIUnit : nn::Module {};
struct AGIUnit2 : nn::Module {
  AGIUnit2() : nn::Module("Foo") {}
};
} // namespace test

bool pointer_equal(at::Tensor first, at::Tensor second) {
  return first.data<float>() == second.data<float>();
}

TEST_CASE("module/training-mode") {
  Linear module(3, 4);
  REQUIRE(module->is_training());
  SECTION("Enable eval mode") {
    module->eval();
    REQUIRE(!module->is_training());
  }
  SECTION("Enable train mode") {
    module->train();
    REQUIRE(module->is_training());
  }
}

TEST_CASE("module/zero-grad") {
  Linear module(3, 4);
  auto weight = torch::ones({8, 3}, at::requires_grad());
  auto loss = module->forward({weight}).front().sum();
  loss.backward();
  for (auto& parameter : module->parameters()) {
    Variable grad = parameter->grad();
    REQUIRE(grad.defined());
    REQUIRE(grad.sum().toCFloat() != 0);
  }
  module->zero_grad();
  for (auto& parameter : module->parameters()) {
    Variable grad = parameter->grad();
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
  auto module = LSTM(LSTMOptions(128, 64).layers(3).dropout(0.2));
  SECTION("starts as float on CPU") {
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->type().backend() == at::kCPU);
      REQUIRE(parameter->type().scalarType() == torch::kFloat32);
    }
  }
  SECTION("to(CUDA)") {
    module->cuda();
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->type().backend() == at::kCUDA);
    }
  }
  SECTION("to(CPU)") {
    module->to(at::kCPU);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->type().backend() == at::kCPU);
    }
  }
  SECTION("to(Int)") {
    module->to(torch::kInt32);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->type().scalarType() == torch::kInt32);
    }
  }
  SECTION("to(Double)") {
    module->to(torch::kFloat64);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->type().scalarType() == torch::kFloat64);
    }
  }
  SECTION("to(CUDA(Float))") {
    module->to(at::CUDA(torch::kFloat32));
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->type().backend() == at::kCUDA);
      REQUIRE(parameter->type().scalarType() == torch::kFloat32);
    }
  }
}

TEST_CASE("module/clone") {
  SECTION(
      "a module that does not override clone() throws when clone() is called") {
    struct UnCloneable : Module {};
    UnCloneable module;
    REQUIRE_THROWS_WITH(
        module.clone(), StartsWith("clone() has not been implemented"));
  }

  SECTION(
      "a module that overrides clone() does not throw when clone() is called ") {
    struct Cloneable : Module {
      std::shared_ptr<Module> clone() const override {
        return nullptr;
      }
    };
    Cloneable module;
    REQUIRE_NOTHROW(module.clone());
  }

  SECTION("Cloning creates distinct parameters") {
    struct TestModule : public Cloneable<TestModule> {
      void reset() override {
        l1 = register_module("l1", Linear(10, 3));
        l2 = register_module("l2", Linear(3, 5));
        l3 = register_module("l3", Linear(5, 100));
      }

      Linear l1, l2, l3;
    };

    auto module = TestModule().build();

    auto module2 = module->clone();
    auto m1param = module->parameters();
    auto m2param = module2->parameters();
    for (auto& param : m1param) {
      REQUIRE(!pointer_equal(param.value, m2param[param.key]));
      REQUIRE(param->allclose(m2param[param.key]));
      param->data().mul_(2);
    }
    for (auto& param : m1param) {
      REQUIRE(!param->allclose(m2param[param.key]));
    }
  }

  SECTION("Cloning preserves external references") {
    struct TestModule : public Cloneable<TestModule> {
      void reset() override {
        weight = register_parameter("weight", torch::ones({4, 4}));
      }
      Variable weight;
    };
    auto module = TestModule().build();
    module->weight.data() += 1;
    REQUIRE(pointer_equal(module->weight, module->parameters()["weight"]));
    REQUIRE(module->weight.allclose(module->parameters()["weight"]));

    auto module2 = std::dynamic_pointer_cast<TestModule>(
        std::shared_ptr<Module>(module->clone()));
    REQUIRE(!pointer_equal(module2->weight, module->weight));
    REQUIRE(pointer_equal(module2->weight, module2->parameters()["weight"]));
    REQUIRE(module2->weight.allclose(module2->parameters()["weight"]));
    REQUIRE(module2->weight.allclose(module->weight));
    REQUIRE(!pointer_equal(module2->weight, module->parameters()["weight"]));
  }

  SECTION("Cloning copies the values of variables of submodules") {
    struct TestModule : public Cloneable<TestModule> {
      void reset() override {
        weight = register_parameter("weight", torch::ones({4, 4}));
      }

      Variable weight;
      int value = 0;
    };
    struct NestedModule : public Cloneable<NestedModule> {
      void reset() override {
        module = register_module("module", TestModule().build());
      }
      std::shared_ptr<TestModule> module;
    };

    auto a = NestedModule().build();
    a->module->weight.data() += 1;
    a->module->value = 123;

    auto b = std::static_pointer_cast<NestedModule>(a->clone());

    REQUIRE(!pointer_equal(b->module->weight, a->module->weight));
    REQUIRE(
        pointer_equal(b->module->weight, b->module->parameters()["weight"]));
    REQUIRE(b->module->parameters()["weight"].allclose(a->module->weight));
    REQUIRE(b->module->weight.allclose(a->module->weight));
    REQUIRE(b->module->value == a->module->value);
  }
}

TEST_CASE("module/parameters") {
  struct TestModule : Module {
    TestModule() {
      a = register_parameter("a", torch::zeros({2, 2}));
      b = register_parameter("b", torch::ones({2, 2}));
      c = register_parameter("c", torch::ones({2, 2}) * 2);
    }

    Variable a, b, c;
  };

  TestModule module;

  SECTION("has correct number of parameters") {
    REQUIRE(module.parameters().size() == 3);
  }

  SECTION("contains parameters with the correct name") {
    auto parameters = module.parameters();
    REQUIRE(parameters.contains("a"));
    REQUIRE(parameters.contains("b"));
    REQUIRE(parameters.contains("c"));
  }
}
