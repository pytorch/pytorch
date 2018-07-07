#include <catch.hpp>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/rnn.h>
#include <torch/tensor.h>
#include <torch/utils.h>

using namespace torch::nn;

using Catch::StartsWith;

struct AGIUnit : torch::nn::Module {};

namespace test {
struct AGIUnit : torch::nn::Module {};
struct AGIUnit2 : torch::nn::Module {
  AGIUnit2() : torch::nn::Module("Foo") {}
};
} // namespace test

bool pointer_equal(torch::Tensor first, torch::Tensor second) {
  return first.data().data<float>() == second.data().data<float>();
}

TEST_CASE("module/training-mode") {
  torch::manual_seed(0);
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
  torch::manual_seed(0);
  Linear module(3, 4);
  auto weight = torch::ones({8, 3}, torch::requires_grad());
  auto loss = module->forward(weight).sum();
  loss.backward();
  for (auto& parameter : module->parameters()) {
    auto grad = parameter->grad();
    REQUIRE(grad.defined());
    REQUIRE(grad.sum().toCFloat() != 0);
  }
  module->zero_grad();
  for (auto& parameter : module->parameters()) {
    auto grad = parameter->grad();
    REQUIRE(grad.defined());
    REQUIRE(grad.sum().toCFloat() == 0);
  }
}

TEST_CASE("module/zero-grad-with-undefined") {
  struct TestModule : torch::nn::Module {
    TestModule() {
      x = register_parameter("x", torch::ones(5, at::requires_grad()));
      y = register_parameter("y", torch::ones(5, at::requires_grad()));
    }
    torch::Tensor x, y;
  };

  TestModule module;
  auto z = module.x * 2;
  z.sum().backward();

  REQUIRE(module.x.grad().defined());
  REQUIRE(!module.y.grad().defined());

  module.zero_grad();

  REQUIRE(module.x.grad().defined());
  REQUIRE(!module.y.grad().defined());

  REQUIRE(module.x.grad().sum().toCFloat() == 0);
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

TEST_CASE("module/as") {
  Linear module(3, 4);
  REQUIRE(module->as<Linear>() == module.get());
  REQUIRE(module->as<LinearImpl>() == module.get());
  REQUIRE(module->as<Module>() == module.get());
  REQUIRE(module->as<AGIUnit>() == nullptr);

  std::shared_ptr<Module> raw = module.ptr();
  REQUIRE(raw->as<Linear>() == module.get());
  REQUIRE(raw->as<LinearImpl>() == module.get());
  REQUIRE(raw->as<Module>() == module.get());
  REQUIRE(raw->as<AGIUnit>() == nullptr);

  Module& raw_ref = *raw.get();
  REQUIRE(raw_ref.as<Linear>() == module.get());
  REQUIRE(raw_ref.as<LinearImpl>() == module.get());
  REQUIRE(raw_ref.as<Module>() == module.get());
  REQUIRE(raw_ref.as<AGIUnit>() == nullptr);
  if (auto* linear = raw_ref.as<Linear>()) {
    REQUIRE(linear->weight.ndimension() == 2);
  }

  AGIUnit unit;
  REQUIRE(unit.as<Linear>() == nullptr);
  REQUIRE(unit.as<LinearImpl>() == nullptr);
  REQUIRE(unit.as<AGIUnit>() == &unit);
}

TEST_CASE("module/conversions", "[cuda]") {
  torch::manual_seed(0);
  Linear module(128, 64);
  SECTION("starts as float on CPU") {
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->device() == torch::Device(torch::kCPU));
      REQUIRE(parameter->dtype() == torch::kFloat32);
    }
  }
  SECTION("to(CUDA)") {
    module->to({torch::kCUDA, 0});
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->device().type() == torch::Device::Type::CUDA);
      REQUIRE(parameter->device().index() == 0);
    }
    module->to({at::kCUDA, 1});
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->device().type() == torch::Device::Type::CUDA);
      REQUIRE(parameter->device().index() == 1);
    }
  }
  SECTION("to(CPU)") {
    module->to(torch::Device(torch::kCPU));
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->device().type() == torch::Device::Type::CPU);
    }
  }
  SECTION("to(Int32)") {
    module->to(torch::kInt32);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->dtype() == torch::kInt32);
    }
  }
  SECTION("to(Float64)") {
    module->to(torch::kFloat64);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->dtype() == torch::kFloat64);
    }
  }
  SECTION("to(CUDA, Byte)") {
    module->to(torch::Device(torch::kCUDA, 1), torch::kUInt8);
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->device().type() == torch::Device::Type::CUDA);
      REQUIRE(parameter->device().index() == 1);
    }
    for (auto& parameter : module->parameters()) {
      REQUIRE(parameter->dtype() == torch::kUInt8);
    }
  }
}

TEST_CASE("module/clone") {
  torch::manual_seed(0);
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
        buffer = register_buffer("buf", torch::ones({2, 2}));
      }

      Linear l1, l2, l3;
      torch::Tensor buffer;
    };

    auto module = TestModule().build();

    auto module2 = module->clone();
    auto params1 = module->parameters();
    auto params2 = module2->parameters();
    REQUIRE(params1.size() == 6);
    REQUIRE(params2.size() == 6);
    for (auto& param : params1) {
      REQUIRE(!pointer_equal(param.value, params2[param.key]));
      REQUIRE(param->allclose(params2[param.key]));
      param->data().mul_(2);
    }
    for (auto& param : params1) {
      REQUIRE(!param->allclose(params2[param.key]));
    }

    auto buffers1 = module->buffers();
    auto buffers2 = module2->buffers();
    REQUIRE(buffers1.size() == 1);
    REQUIRE(buffers2.size() == 1);
    for (auto& buffer : buffers1) {
      REQUIRE(!pointer_equal(buffer.value, buffers2[buffer.key]));
      REQUIRE(buffer->allclose(buffers2[buffer.key]));
      buffer->data().mul_(2);
    }
    for (auto& buffer : buffers1) {
      REQUIRE(!buffer->allclose(buffers2[buffer.key]));
    }
  }

  SECTION("Cloning preserves external references") {
    struct TestModule : public Cloneable<TestModule> {
      void reset() override {
        weight = register_parameter("weight", torch::ones({4, 4}));
      }
      torch::Tensor weight;
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

      torch::Tensor weight;
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
  torch::manual_seed(0);
  struct TestModule : Module {
    TestModule() {
      a = register_parameter("a", torch::zeros({2, 2}));
      b = register_parameter("b", torch::ones({2, 2}));
      c = register_parameter("c", torch::ones({2, 2}) * 2);
    }

    torch::Tensor a, b, c;
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

TEST_CASE("module/buffers") {
  torch::manual_seed(0);
  struct TestModule : Module {
    TestModule() {
      a = register_buffer("a", torch::zeros({2, 2}));
      b = register_buffer("b", torch::ones({2, 2}));
      c = register_buffer("c", torch::ones({2, 2}) * 2);
    }

    torch::Tensor a, b, c;
  };

  TestModule module;

  SECTION("has correct number of buffers") {
    REQUIRE(module.buffers().size() == 3);
  }

  SECTION("contains buffers with the correct name") {
    auto buffers = module.buffers();
    REQUIRE(buffers.contains("a"));
    REQUIRE(buffers.contains("b"));
    REQUIRE(buffers.contains("c"));
  }
}
