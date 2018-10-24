#include <gtest/gtest.h>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/rnn.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct AGIUnit : torch::nn::Module {};

namespace test {
struct AGIUnit : torch::nn::Module {};
struct AGIUnit2 : torch::nn::Module {
  AGIUnit2() : torch::nn::Module("Foo") {}
};
} // namespace test

struct ModuleTest : torch::test::SeedingFixture {};

TEST_F(ModuleTest, CanEnableAndDisableTrainingMode) {
  Linear module(3, 4);
  ASSERT_TRUE(module->is_training());

  module->eval();
  ASSERT_FALSE(module->is_training());

  module->train();
  ASSERT_TRUE(module->is_training());
}

TEST_F(ModuleTest, ZeroGrad) {
  Linear module(3, 4);
  auto weight = torch::ones({8, 3}, torch::requires_grad());
  auto loss = module->forward(weight).sum();
  loss.backward();
  for (auto& parameter : module->parameters()) {
    auto grad = parameter->grad();
    ASSERT_TRUE(grad.defined());
    ASSERT_NE(grad.sum().item<float>(), 0);
  }
  module->zero_grad();
  for (auto& parameter : module->parameters()) {
    auto grad = parameter->grad();
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.sum().item<float>(), 0);
  }
}

TEST_F(ModuleTest, ZeroGradWithUndefined) {
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

  ASSERT_TRUE(module.x.grad().defined());
  ASSERT_FALSE(module.y.grad().defined());

  module.zero_grad();

  ASSERT_TRUE(module.x.grad().defined());
  ASSERT_FALSE(module.y.grad().defined());

  ASSERT_EQ(module.x.grad().sum().item<float>(), 0);
}

TEST_F(ModuleTest, CanGetName) {
  // CHECK instead of REQUIRE because demangling may fail.
  AGIUnit agi;
  // Call it twice just to make sure there are no bugs in the lazy
  // initialization semantics.
  EXPECT_TRUE(agi.name() == "AGIUnit");
  EXPECT_TRUE(agi.name() == "AGIUnit");
  EXPECT_TRUE(test::AGIUnit().name() == "test::AGIUnit");
  EXPECT_TRUE(test::AGIUnit2().name() == "Foo");
}

TEST_F(ModuleTest, TestAsCastsModulesCorrectly) {
  Linear module(3, 4);
  ASSERT_EQ(module->as<Linear>(), module.get());
  ASSERT_EQ(module->as<LinearImpl>(), module.get());
  ASSERT_EQ(module->as<Module>(), module.get());
  ASSERT_EQ(module->as<AGIUnit>(), nullptr);

  std::shared_ptr<Module> raw = module.ptr();
  ASSERT_EQ(raw->as<Linear>(), module.get());
  ASSERT_EQ(raw->as<LinearImpl>(), module.get());
  ASSERT_EQ(raw->as<Module>(), module.get());
  ASSERT_EQ(raw->as<AGIUnit>(), nullptr);

  Module& raw_ref = *raw.get();
  ASSERT_EQ(raw_ref.as<Linear>(), module.get());
  ASSERT_EQ(raw_ref.as<LinearImpl>(), module.get());
  ASSERT_EQ(raw_ref.as<Module>(), module.get());
  ASSERT_EQ(raw_ref.as<AGIUnit>(), nullptr);
  if (auto* linear = raw_ref.as<Linear>()) {
    ASSERT_EQ(linear->weight.ndimension(), 2);
  }

  AGIUnit unit;
  ASSERT_EQ(unit.as<Linear>(), nullptr);
  ASSERT_EQ(unit.as<LinearImpl>(), nullptr);
  ASSERT_EQ(unit.as<AGIUnit>(), &unit);
}

TEST_F(ModuleTest, Conversion_MultiCUDA) {
  Linear module(128, 64);
  for (auto& parameter : module->parameters()) {
    ASSERT_EQ(parameter->device(), torch::Device(torch::kCPU));
    ASSERT_EQ(parameter->dtype(), torch::kFloat32);
  }
  {
    module->to({torch::kCUDA, 0});
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter->device().type(), torch::Device::Type::CUDA);
      ASSERT_EQ(parameter->device().index(), 0);
    }
    module->to({at::kCUDA, 1});
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter->device().type(), torch::Device::Type::CUDA);
      ASSERT_EQ(parameter->device().index(), 1);
    }
  }
  {
    module->to(torch::Device(torch::kCPU));
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter->device().type(), torch::Device::Type::CPU);
    }
  }
  {
    module->to(torch::kInt32);
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter->dtype(), torch::kInt32);
    }
  }
  {
    module->to(torch::kFloat64);
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter->dtype(), torch::kFloat64);
    }
  }
  {
    module->to(torch::Device(torch::kCUDA, 1), torch::kUInt8);
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter->device().type(), torch::Device::Type::CUDA);
      ASSERT_EQ(parameter->device().index(), 1);
    }
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter->dtype(), torch::kUInt8);
    }
  }
}

TEST_F(ModuleTest, CallingCloneOnModuleThatDoesNotOverrideCloneThrows) {
  struct UnCloneable : Module {};
  UnCloneable module;
  ASSERT_THROWS_WITH(module.clone(), "clone() has not been implemented");
}

TEST_F(ModuleTest, CallingCloneOnModuleThatDoesOverrideCloneDoesNotThrow) {
  struct Cloneable : Module {
    std::shared_ptr<Module> clone(
        c10::optional<torch::Device> device = c10::nullopt) const override {
      return nullptr;
    }
  };
  Cloneable module;
  ASSERT_NO_THROW({ module.clone(); });
}

TEST_F(ModuleTest, CloneCreatesDistinctParameters) {
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      reset();
    }
    void reset() override {
      l1 = register_module("l1", Linear(10, 3));
      l2 = register_module("l2", Linear(3, 5));
      l3 = register_module("l3", Linear(5, 100));
      buffer = register_buffer("buf", torch::ones({2, 2}));
    }

    Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    torch::Tensor buffer;
  };

  auto module = std::make_shared<TestModule>();

  torch::NoGradGuard no_grad;

  auto module2 = module->clone();
  auto params1 = module->parameters();
  auto params2 = module2->parameters();
  ASSERT_EQ(params1.size(), 6);
  ASSERT_EQ(params2.size(), 6);
  for (auto& param : params1) {
    ASSERT_FALSE(pointer_equal(param.value, params2[param.key]));
    ASSERT_TRUE(param->allclose(params2[param.key]));
    param->add_(2);
  }
  for (auto& param : params1) {
    ASSERT_FALSE(param->allclose(params2[param.key]));
  }

  auto buffers1 = module->buffers();
  auto buffers2 = module2->buffers();
  ASSERT_EQ(buffers1.size(), 1);
  ASSERT_EQ(buffers2.size(), 1);
  for (auto& buffer : buffers1) {
    ASSERT_FALSE(pointer_equal(buffer.value, buffers2[buffer.key]));
    ASSERT_TRUE(buffer->allclose(buffers2[buffer.key]));
    buffer->add_(2);
  }
  for (auto& buffer : buffers1) {
    ASSERT_FALSE(buffer->allclose(buffers2[buffer.key]));
  }
}

TEST_F(ModuleTest, ClonePreservesExternalReferences) {
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      reset();
    }
    void reset() override {
      weight = register_parameter("weight", torch::ones({4, 4}));
    }
    torch::Tensor weight;
  };
  auto module = std::make_shared<TestModule>();
  {
    torch::NoGradGuard no_grad;
    module->weight += 1;
  }
  ASSERT_TRUE(pointer_equal(module->weight, module->parameters()["weight"]));
  ASSERT_TRUE(module->weight.allclose(module->parameters()["weight"]));

  auto module2 = std::dynamic_pointer_cast<TestModule>(
      std::shared_ptr<Module>(module->clone()));
  ASSERT_FALSE(pointer_equal(module2->weight, module->weight));
  ASSERT_TRUE(pointer_equal(module2->weight, module2->parameters()["weight"]));
  ASSERT_TRUE(module2->weight.allclose(module2->parameters()["weight"]));
  ASSERT_TRUE(module2->weight.allclose(module->weight));
  ASSERT_FALSE(pointer_equal(module2->weight, module->parameters()["weight"]));
}

TEST_F(ModuleTest, CloneCopiesTheValuesOfVariablesOfSubmodules) {
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      reset();
    }
    void reset() override {
      weight = register_parameter("weight", torch::ones({4, 4}));
    }

    torch::Tensor weight;
    int value = 0;
  };
  struct NestedModule : public Cloneable<NestedModule> {
    NestedModule() {
      reset();
    }
    void reset() override {
      module = register_module("module", std::make_shared<TestModule>());
    }
    std::shared_ptr<TestModule> module;
  };

  auto a = std::make_shared<NestedModule>();
  {
    torch::NoGradGuard no_grad;
    a->module->weight += 1;
    a->module->value = 123;
  }

  auto b = std::dynamic_pointer_cast<NestedModule>(a->clone());

  ASSERT_FALSE(pointer_equal(b->module->weight, a->module->weight));
  ASSERT_TRUE(
      pointer_equal(b->module->weight, b->module->parameters()["weight"]));
  ASSERT_TRUE(b->module->parameters()["weight"].allclose(a->module->weight));
  ASSERT_TRUE(b->module->weight.allclose(a->module->weight));
  ASSERT_EQ(b->module->value, a->module->value);
}

TEST_F(ModuleTest, CloneToDevicePreservesTheDeviceOfParameters_CUDA) {
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      reset();
    }
    void reset() override {
      l1 = register_module("l1", Linear(10, 3));
      l2 = register_module("l2", Linear(3, 5));
      l3 = register_module("l3", Linear(5, 100));
      buffer = register_buffer("buf", torch::ones({2, 2}));
    }

    Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    torch::Tensor buffer;
  };

  TestModule m;
  torch::Device device(torch::kCUDA, 0);

  m.to(device);

  auto clone = m.clone();
  for (const auto& parameter : clone->parameters()) {
    ASSERT_EQ(parameter->device().type(), device.type());
    ASSERT_EQ(parameter->device().index(), device.index());
  }
  for (const auto& buffer : clone->buffers()) {
    ASSERT_EQ(buffer->device().type(), device.type());
    ASSERT_EQ(buffer->device().index(), device.index());
  }
}

TEST_F(ModuleTest, CloningToAParticularDevicePlacesAllParametersThere_CUDA) {
  struct TestModule : public Cloneable<TestModule> {
    TestModule() {
      reset();
    }
    void reset() override {
      l1 = register_module("l1", Linear(10, 3));
      l2 = register_module("l2", Linear(3, 5));
      l3 = register_module("l3", Linear(5, 100));
      buffer = register_buffer("buf", torch::ones({2, 2}));
    }

    Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    torch::Tensor buffer;
  };

  TestModule m;
  torch::Device device(torch::kCUDA, 1);
  // everything is on CPU here
  auto clone = m.clone(device);
  for (const auto& parameter : clone->parameters()) {
    ASSERT_EQ(parameter->device().type(), device.type());
    ASSERT_EQ(parameter->device().index(), device.index());
  }
  for (const auto& buffer : clone->buffers()) {
    ASSERT_EQ(buffer->device().type(), device.type());
    ASSERT_EQ(buffer->device().index(), device.index());
  }
}

struct ParameterTestModule : Module {
  ParameterTestModule() {
    a = register_parameter("a", torch::zeros({2, 2}));
    b = register_parameter("b", torch::ones({2, 2}));
    c = register_parameter("c", torch::ones({2, 2}) * 2);
  }

  torch::Tensor a, b, c;
};

TEST_F(ModuleTest, HasCorrectNumberOfParameters) {
  ParameterTestModule module;
  ASSERT_EQ(module.parameters().size(), 3);
}

TEST_F(ModuleTest, ContainsParametersWithTheCorrectName) {
  ParameterTestModule module;
  auto parameters = module.parameters();
  ASSERT_TRUE(parameters.contains("a"));
  ASSERT_TRUE(parameters.contains("b"));
  ASSERT_TRUE(parameters.contains("c"));
}

struct BufferTestModule : Module {
  BufferTestModule() {
    a = register_buffer("a", torch::zeros({2, 2}));
    b = register_buffer("b", torch::ones({2, 2}));
    c = register_buffer("c", torch::ones({2, 2}) * 2);
  }

  torch::Tensor a, b, c;
};

TEST_F(ModuleTest, HasCorrectNumberOfBuffers) {
  BufferTestModule module;
  ASSERT_EQ(module.buffers().size(), 3);
}

TEST_F(ModuleTest, ContainsBuffersWithTheCorrectName) {
  BufferTestModule module;
  auto buffers = module.buffers();
  ASSERT_TRUE(buffers.contains("a"));
  ASSERT_TRUE(buffers.contains("b"));
  ASSERT_TRUE(buffers.contains("c"));
}

struct AImpl : torch::nn::Module {
  AImpl() : x_(123) {}
  AImpl(int x) : x_(x) {}
  int x_;
};
TORCH_MODULE(A);

TEST_F(
    ModuleTest,
    DefaultConstructorOfModuleHolderCallsDefaultConstructorOfImpl) {
  A a;
  ASSERT_TRUE(a);
  ASSERT_FALSE(a.is_empty());
  ASSERT_EQ(a->x_, 123);
}

TEST_F(
    ModuleTest,
    ValueConstructorOfModuleHolderCallsCorrectConstructorInImpl) {
  A a(5);
  ASSERT_TRUE(a);
  ASSERT_FALSE(a.is_empty());
  ASSERT_EQ(a->x_, 5);
}

TEST_F(ModuleTest, NullptrConstructorLeavesTheModuleHolderInEmptyState) {
  A a = nullptr;
  ASSERT_FALSE(a);
  ASSERT_TRUE(a.is_empty());
  ASSERT_THROWS_WITH(a->x_, "Accessing empty ModuleHolder");
}
