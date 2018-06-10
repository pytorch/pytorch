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

struct TestModule : public CloneableModule<TestModule> {
  void reset() override {
    weight =
        register_parameter("weight", at::ones(at::CPU(at::kFloat), {4, 4}));
  }

  Variable weight;
  int value = 0;
};

TEST_CASE("module/training-mode") {
  auto module = Linear(3, 4).build();
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
  auto module = Linear(3, 4).build();
  auto weight = Var(at::ones(at::CPU(at::kFloat), {8, 3}));
  auto loss = module->forward({weight}).front().sum();
  loss.backward();
  for (auto& parameter : module->parameters()) {
    Variable grad = parameter.second.grad();
    REQUIRE(grad.defined());
    REQUIRE(grad.sum().toCFloat() != 0);
  }
  module->zero_grad();
  for (auto& parameter : module->parameters()) {
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
  auto module = LSTM(128, 64).layers(3).dropout(0.2).build();
  SECTION("starts as float on CPU") {
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCPU);
      REQUIRE(parameter.second.type().scalarType() == at::kFloat);
    }
  }
  SECTION("to(CUDA)") {
    module->cuda();
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCUDA);
    }
  }
  SECTION("to(CPU)") {
    module->to(at::kCPU);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCPU);
    }
  }
  SECTION("to(Int)") {
    module->to(at::kInt);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter.second.type().scalarType() == at::kInt);
    }
  }
  SECTION("to(Double)") {
    module->to(at::kDouble);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter.second.type().scalarType() == at::kDouble);
    }
  }
  SECTION("to(CUDA(Float))") {
    module->to(at::CUDA(at::kFloat));
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCUDA);
      REQUIRE(parameter.second.type().scalarType() == at::kFloat);
    }
  }
}

TEST_CASE("module/clone") {
  SECTION(
      "a module that does not override clone() throws when clone() is called") {
    struct UnCloneable : Module {
      std::vector<Variable> forward(std::vector<Variable>) {
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
      std::vector<Variable> forward(std::vector<Variable>) {
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
    struct TestModule : public CloneableModule<TestModule> {
      void reset() override {
        l1 = register_module("l1", Linear(10, 3).build());
        l2 = register_module("l2", Linear(3, 5).build());
        l3 = register_module("l3", Linear(5, 100).build());
      }

      std::shared_ptr<Linear> l1, l2, l3;
    };

    auto module = TestModule().build();

    auto module2 = module->clone();
    auto m1param = module->parameters();
    auto m2param = module2->parameters();
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
      void reset() override {
        weight =
            register_parameter("weight", at::ones(at::CPU(at::kFloat), {4, 4}));
      }
      Variable weight;
    };
    auto module = TestModule().build();
    module->weight.data() += 1;
    REQUIRE(pointer_equal(module->weight, module->param("weight")));
    REQUIRE(module->weight.allclose(module->param("weight")));

    auto module2 = std::dynamic_pointer_cast<TestModule>(
        std::shared_ptr<Module>(module->clone()));
    REQUIRE(!pointer_equal(module2->weight, module->weight));
    REQUIRE(pointer_equal(module2->weight, module2->param("weight")));
    REQUIRE(module2->weight.allclose(module2->param("weight")));
    REQUIRE(module2->weight.allclose(module->weight));
    REQUIRE(!pointer_equal(module2->weight, module->param("weight")));
  }

  SECTION("Cloning copies the values of variables of submodules") {
    struct NestedModule : public CloneableModule<NestedModule> {
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
    REQUIRE(pointer_equal(b->module->weight, b->module->param("weight")));
    REQUIRE(b->module->param("weight").allclose(a->module->weight));
    REQUIRE(b->module->weight.allclose(a->module->weight));
    REQUIRE(b->module->value == a->module->value);
  }
}

TEST_CASE("module/parameters") {
  struct TestModule : Module {
    TestModule() {
      a = register_parameter("a", at::zeros(at::CPU(at::kFloat), {2, 2}));
      b = register_parameter("b", at::ones(at::CPU(at::kFloat), {2, 2}));
      c = register_parameter("c", at::ones(at::CPU(at::kFloat), {2, 2}) * 2);
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
