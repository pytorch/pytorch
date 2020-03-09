#include <gtest/gtest.h>

#include <torch/torch.h>

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
  auto loss = module(weight).sum();
  loss.backward();
  for (auto& parameter : module->parameters()) {
    auto grad = parameter.grad();
    ASSERT_TRUE(grad.defined());
    ASSERT_NE(grad.sum().item<float>(), 0);
  }
  module->zero_grad();
  for (auto& parameter : module->parameters()) {
    auto grad = parameter.grad();
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.sum().item<float>(), 0);
  }
}

TEST_F(ModuleTest, ZeroGradWithUndefined) {
  struct TestModule : torch::nn::Module {
    TestModule() {
      x = register_parameter("x", torch::ones(5, torch::requires_grad()));
      y = register_parameter("y", torch::ones(5, torch::requires_grad()));
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

TEST_F(ModuleTest, RegisterModuleThrowsForEmptyOrDottedName) {
  struct TestModel : public torch::nn::Module {};
  ASSERT_THROWS_WITH(
      TestModel{}.register_module("name.with.dot", torch::nn::Linear(3, 4)),
      "Submodule name must not contain a dot (got 'name.with.dot')");
  ASSERT_THROWS_WITH(
      TestModel{}.register_module("", torch::nn::Linear(3, 4)),
      "Submodule name must not be empty");
}

TEST_F(ModuleTest, RegisterModuleThrowsForDuplicateModuleName) {
  struct TestModel : public torch::nn::Module {};
  TestModel model;
  model.register_module("linear", torch::nn::Linear(3, 4));
  ASSERT_THROWS_WITH(
      model.register_module("linear", torch::nn::Linear(3, 4)),
      "Submodule 'linear' already defined");
}

TEST_F(ModuleTest, ReplaceModuleThrowsForUnknownModuleName) {
  torch::nn::Module model;
  ASSERT_THROWS_WITH(
      model.replace_module("linear", torch::nn::Linear(3, 4)),
      "Submodule 'linear' is not defined");
}

TEST_F(ModuleTest, ReplaceModule) {
  struct TestModel : public torch::nn::Module {
    torch::nn::Linear l1{nullptr};
    TestModel() {
      l1 = register_module("l1", torch::nn::Linear(3, 4));
    }
  };
  auto model = std::make_shared<TestModel>();
  model->l1 = model->replace_module("l1", torch::nn::Linear(5, 6));
  ASSERT_EQ(model->named_parameters()["l1.weight"].size(0), 6);
  ASSERT_EQ(model->l1.get(), model->named_modules()["l1"]->as<Linear>());
}

TEST_F(ModuleTest, UnregisterModule) {
  struct TestModel : public torch::nn::Module {};
  TestModel model;
  ASSERT_THROWS_WITH(
      model.unregister_module("linear"),
      "No Module with name `linear` is registered");
  model.register_module("linear", torch::nn::Linear(3, 4));
  model.unregister_module("linear");
  ASSERT_TRUE(model.children().empty());
}

TEST_F(ModuleTest, RegisterParameterThrowsForEmptyOrDottedName) {
  struct TestModel : public torch::nn::Module {};
  ASSERT_THROWS_WITH(
      TestModel{}.register_parameter("name.with.dot", torch::ones(5)),
      "Parameter name must not contain a dot (got 'name.with.dot')");
  ASSERT_THROWS_WITH(
      TestModel{}.register_parameter("", torch::ones(5)),
      "Parameter name must not be empty");
}

TEST_F(ModuleTest, RegisterParameterThrowsForDuplicateModuleName) {
  struct TestModel : public torch::nn::Module {};
  TestModel model;
  model.register_parameter("p", torch::ones(5));
  ASSERT_THROWS_WITH(
      model.register_parameter("p", torch::ones(5)),
      "Parameter 'p' already defined");
}

TEST_F(ModuleTest, RegisterParameterUndefinedTensor) {
  struct TestModel : public torch::nn::Module {};
  {
    TestModel model;
    model.register_parameter("undefined_tensor", torch::Tensor(), /*requires_grad=*/false);
    ASSERT_EQ(model.parameters().size(), 0);
  }
  {
    std::stringstream buffer;
    CerrRedirect cerr_redirect(buffer.rdbuf());

    TestModel model;
    model.register_parameter("undefined_tensor", torch::Tensor());
    ASSERT_EQ(model.parameters().size(), 0);

    ASSERT_EQ(
      count_substr_occurrences(
        buffer.str(),
        "Ignoring the `requires_grad=true` function parameter"
      ),
    1);
  }
}

TEST_F(ModuleTest, RegisterBufferThrowsForEmptyOrDottedName) {
  struct TestModel : public torch::nn::Module {};
  ASSERT_THROWS_WITH(
      TestModel{}.register_buffer("name.with.dot", torch::ones(5)),
      "Buffer name must not contain a dot (got 'name.with.dot')");
  ASSERT_THROWS_WITH(
      TestModel{}.register_buffer("", torch::ones(5)),
      "Buffer name must not be empty");
}

TEST_F(ModuleTest, RegisterBufferThrowsForDuplicateModuleName) {
  struct TestModel : public torch::nn::Module {};
  TestModel model;
  model.register_buffer("p", torch::ones(5));
  ASSERT_THROWS_WITH(
      model.register_buffer("p", torch::ones(5)), "Buffer 'p' already defined");
}

TEST_F(ModuleTest, CanGetName) {
  // CHECK instead of REQUIRE because demangling may fail.
  AGIUnit agi;
  // Call it twice just to make sure there are no bugs in the lazy
  // initialization semantics.
  EXPECT_EQ(agi.name(), "AGIUnit");
  EXPECT_EQ(agi.name(), "AGIUnit");
  EXPECT_EQ(test::AGIUnit().name(), "test::AGIUnit");
  EXPECT_EQ(test::AGIUnit2().name(), "Foo");
}

TEST_F(ModuleTest, AsCastsModulesCorrectly) {
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

void test_DeviceOrDtypeConversionSkipsUndefinedTensor(
  torch::Device to_device, torch::Dtype to_dtype) {
  {
    // Case 1: Undefined tensors as parameters
    Linear module(LinearOptions(10, 20).bias(false));
    ASSERT_TRUE(module->weight.defined());
    ASSERT_FALSE(module->bias.defined());

    module->to(to_device);
    ASSERT_TRUE(module->weight.defined());
    ASSERT_EQ(module->weight.device().type(), to_device.type());
    ASSERT_FALSE(module->bias.defined());

    module->to(to_dtype);
    ASSERT_TRUE(module->weight.defined());
    ASSERT_EQ(module->weight.dtype(), to_dtype);
    ASSERT_FALSE(module->bias.defined());
  }
  {
    // Case 2: Undefined tensors as buffers
    BatchNorm1d module(BatchNorm1dOptions(5).track_running_stats(false).affine(true));
    ASSERT_TRUE(module->weight.defined());
    ASSERT_FALSE(module->running_mean.defined());

    module->to(to_device);
    ASSERT_TRUE(module->weight.defined());
    ASSERT_EQ(module->weight.device().type(), to_device.type());
    ASSERT_FALSE(module->running_mean.defined());

    module->to(to_dtype);
    ASSERT_TRUE(module->weight.defined());
    ASSERT_EQ(module->weight.dtype(), to_dtype);
    ASSERT_FALSE(module->running_mean.defined());
  }
}

TEST_F(ModuleTest, DeviceOrDtypeConversionSkipsUndefinedTensor) {
  test_DeviceOrDtypeConversionSkipsUndefinedTensor(torch::kCPU, torch::kDouble);
}

TEST_F(ModuleTest, DeviceOrDtypeConversionSkipsUndefinedTensor_CUDA) {
  test_DeviceOrDtypeConversionSkipsUndefinedTensor(torch::kCUDA, torch::kDouble);
}

TEST_F(ModuleTest, ParametersAndBuffersAccessorSkipsUndefinedTensor) {
  {
    Linear module(LinearOptions(10, 20).bias(false));

    auto params = module->parameters();
    ASSERT_EQ(params.size(), 1);
    auto named_params = module->named_parameters();
    ASSERT_EQ(named_params.size(), 1);

    ASSERT_TRUE(pointer_equal(params[0], named_params["weight"]));
    ASSERT_TRUE(pointer_equal(named_params["weight"], module->weight));
  }
  {
    BatchNorm1d module(BatchNorm1dOptions(5).track_running_stats(false).affine(false));

    auto buffers = module->buffers();
    ASSERT_EQ(buffers.size(), 0);
    auto named_buffers = module->named_buffers();
    ASSERT_EQ(named_buffers.size(), 0);
  }
  {
    BatchNorm1d module(BatchNorm1dOptions(5).track_running_stats(true).affine(false));

    auto buffers = module->buffers();
    ASSERT_EQ(buffers.size(), 3);
    auto named_buffers = module->named_buffers();
    ASSERT_EQ(named_buffers.size(), 3);

    ASSERT_TRUE(pointer_equal(buffers[0], named_buffers["running_mean"]));
    ASSERT_TRUE(pointer_equal(named_buffers["running_mean"], module->running_mean));
    ASSERT_TRUE(pointer_equal(buffers[1], named_buffers["running_var"]));
    ASSERT_TRUE(pointer_equal(named_buffers["running_var"], module->running_var));
    ASSERT_TRUE(pointer_equal(buffers[2], named_buffers["num_batches_tracked"]));
    ASSERT_TRUE(pointer_equal(named_buffers["num_batches_tracked"], module->num_batches_tracked));
  }
}

TEST_F(ModuleTest, Conversion_MultiCUDA) {
  Linear module(128, 64);
  for (auto& parameter : module->parameters()) {
    ASSERT_EQ(parameter.device(), torch::Device(torch::kCPU));
    ASSERT_EQ(parameter.dtype(), torch::kFloat32);
  }
  {
    module->to({torch::kCUDA, 0});
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter.device().type(), torch::Device::Type::CUDA);
      ASSERT_EQ(parameter.device().index(), 0);
    }
    module->to({torch::kCUDA, 1});
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter.device().type(), torch::Device::Type::CUDA);
      ASSERT_EQ(parameter.device().index(), 1);
    }
  }
  {
    module->to(torch::Device(torch::kCPU));
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter.device().type(), torch::Device::Type::CPU);
    }
  }
  {
    module->to(torch::kInt32);
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter.dtype(), torch::kInt32);
    }
  }
  {
    module->to(torch::kFloat64);
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter.dtype(), torch::kFloat64);
    }
  }
  {
    module->to(torch::Device(torch::kCUDA, 1), torch::kUInt8);
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter.device().type(), torch::Device::Type::CUDA);
      ASSERT_EQ(parameter.device().index(), 1);
    }
    for (auto& parameter : module->parameters()) {
      ASSERT_EQ(parameter.dtype(), torch::kUInt8);
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
        const torch::optional<torch::Device>& device =
            torch::nullopt) const override {
      return nullptr;
    }
  };
  Cloneable module;
  ASSERT_NO_THROW({ module.clone(); });
}

struct TestDistinctParametersModule
    : public Cloneable<TestDistinctParametersModule> {
  TestDistinctParametersModule() {
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

void testDistinctParameters(std::shared_ptr<Module> m1, std::shared_ptr<Module> m2) {
  auto params1 = m1->named_parameters();
  auto params2 = m2->named_parameters();
  ASSERT_EQ(params1.size(), 6);
  ASSERT_EQ(params2.size(), 6);
  for (auto& param : params1) {
    ASSERT_FALSE(pointer_equal(param.value(), params2[param.key()]));
    ASSERT_TRUE(param->allclose(params2[param.key()]));
    param->add_(2);
  }
  for (auto& param : params1) {
    ASSERT_FALSE(param->allclose(params2[param.key()]));
  }

  auto buffers1 = m1->named_buffers();
  auto buffers2 = m2->named_buffers();
  ASSERT_EQ(buffers1.size(), 1);
  ASSERT_EQ(buffers2.size(), 1);
  for (auto& buffer : buffers1) {
    ASSERT_FALSE(pointer_equal(buffer.value(), buffers2[buffer.key()]));
    ASSERT_TRUE(buffer->allclose(buffers2[buffer.key()]));
    buffer->add_(2);
  }
  for (auto& buffer : buffers1) {
    ASSERT_FALSE(buffer->allclose(buffers2[buffer.key()]));
  }
}

TEST_F(ModuleTest, CloneCreatesDistinctParameters) {
  auto module = std::make_shared<TestDistinctParametersModule>();
  torch::NoGradGuard no_grad;
  auto module2 = module->clone();
  testDistinctParameters(module, module2);
}

TEST_F(ModuleTest, CloneCreatesDistinctParametersExplicitDevice_CUDA) {
  auto module = std::make_shared<TestDistinctParametersModule>();
  torch::NoGradGuard no_grad;
  torch::Device device(torch::kCUDA, 0);
  module->to(device);
  auto module2 = module->clone(device);
  testDistinctParameters(module, module2);
}

TEST_F(ModuleTest, CloneCreatesDistinctParametersExplicitDevice_MultiCUDA) {
  auto module = std::make_shared<TestDistinctParametersModule>();
  torch::NoGradGuard no_grad;
  torch::Device d0(torch::kCUDA, 0);
  torch::Device d1(torch::kCUDA, 1);
  module->to(d0);
  auto module2 = module->clone(d1);

  for (auto& param : module->parameters()) {
    ASSERT_EQ(param.device(), d0);
  }

  for (auto& param : module2->parameters()) {
    ASSERT_EQ(param.device(), d1);
  }

  // need to move the module back to d0 as allclose expects two tensors on
  // the same device.
  module2->to(d0);
  testDistinctParameters(module, module2);
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
  ASSERT_TRUE(
      pointer_equal(module->weight, module->named_parameters()["weight"]));
  ASSERT_TRUE(module->weight.allclose(module->named_parameters()["weight"]));

  auto module2 = std::dynamic_pointer_cast<TestModule>(
      std::shared_ptr<Module>(module->clone()));
  ASSERT_FALSE(pointer_equal(module2->weight, module->weight));
  ASSERT_TRUE(
      pointer_equal(module2->weight, module2->named_parameters()["weight"]));
  ASSERT_TRUE(module2->weight.allclose(module2->named_parameters()["weight"]));
  ASSERT_TRUE(module2->weight.allclose(module->weight));
  ASSERT_FALSE(
      pointer_equal(module2->weight, module->named_parameters()["weight"]));
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
  ASSERT_TRUE(pointer_equal(
      b->module->weight, b->module->named_parameters()["weight"]));
  ASSERT_TRUE(
      b->module->named_parameters()["weight"].allclose(a->module->weight));
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
    ASSERT_EQ(parameter.device().type(), device.type());
    ASSERT_EQ(parameter.device().index(), device.index());
  }
  for (const auto& buffer : clone->buffers()) {
    ASSERT_EQ(buffer.device().type(), device.type());
    ASSERT_EQ(buffer.device().index(), device.index());
  }
}

TEST_F(
    ModuleTest,
    CloningToAParticularDevicePlacesAllParametersThere_MultiCUDA) {
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
    ASSERT_EQ(parameter.device().type(), device.type());
    ASSERT_EQ(parameter.device().index(), device.index());
  }
  for (const auto& buffer : clone->buffers()) {
    ASSERT_EQ(buffer.device().type(), device.type());
    ASSERT_EQ(buffer.device().index(), device.index());
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
  ASSERT_EQ(module.named_parameters().size(), 3);
}

TEST_F(ModuleTest, ContainsParametersWithTheCorrectName) {
  ParameterTestModule module;
  auto parameters = module.named_parameters();
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
  ASSERT_EQ(module.named_buffers().size(), 3);
}

TEST_F(ModuleTest, ContainsBuffersWithTheCorrectName) {
  BufferTestModule module;
  auto buffers = module.named_buffers();
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

struct TestModule : public torch::nn::Module {
  TestModule(int64_t size) {
    p1 = register_parameter("p1", torch::randn({size}));
    p2 = register_parameter("p2", torch::randn({size}));
    b1 = register_buffer("b1", torch::randn({size}));
    b2 = register_buffer("b2", torch::randn({size}));
  }

  torch::Tensor forward(torch::Tensor input) {
    return input;
  }

  torch::Tensor p1, p2, b1, b2;
};

TEST_F(ModuleTest, ModulesReturnsExpectedSubmodulesForFlatModel) {
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  std::vector<std::shared_ptr<torch::nn::Module>> modules = model->modules();
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model.ptr(), model[0], model[1], model[2]};
  ASSERT_EQ(modules.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // Assert pointer equality.
    ASSERT_EQ(modules[i].get(), expected[i].get());
  }
}

TEST_F(ModuleTest, ModulesExcludesSelfWhenIncludeSelfSetToFalse) {
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  std::vector<std::shared_ptr<torch::nn::Module>> modules =
      model->modules(/*include_self=*/false);
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model[0], model[1], model[2]};
  ASSERT_EQ(modules.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // Assert pointer equality.
    ASSERT_EQ(modules[i].get(), expected[i].get());
  }
}

TEST_F(ModuleTest, NamedModulesReturnsExpectedNamedSubmodulesForFlatModel) {
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_modules();
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model.ptr(), model[0], model[1], model[2]};
  ASSERT_EQ(modules.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // Assert pointer equality.
    ASSERT_EQ(modules[i].key(), i ? std::to_string(i - 1) : std::string());
    ASSERT_EQ(modules[i].value().get(), expected[i].get());
  }
}

TEST_F(ModuleTest, NamedModulesExcludesSelfWhenIncludeSelfSetToFalse) {
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_modules(
          /*name_prefix=*/std::string(), /*include_self=*/false);
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model[0], model[1], model[2]};
  ASSERT_EQ(modules.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // Assert pointer equality.
    ASSERT_EQ(modules[i].key(), std::to_string(i));
    ASSERT_EQ(modules[i].value().get(), expected[i].get());
  }
}

TEST_F(ModuleTest, ChildrenReturnsExpectedSubmodulesForFlatModel) {
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  std::vector<std::shared_ptr<torch::nn::Module>> modules = model->children();
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model[0], model[1], model[2]};
  ASSERT_EQ(modules.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // Assert pointer equality.
    ASSERT_EQ(modules[i].get(), expected[i].get());
  }

  // For this flat model, this should be true.
  ASSERT_EQ(modules, model->modules(/*include_self=*/false));
}

TEST_F(ModuleTest, NamedChildrenReturnsExpectedNamedSubmodulesForFlatModel) {
  torch::nn::Sequential model(TestModule(1), TestModule(2), TestModule(3));
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_children();
  std::vector<std::shared_ptr<torch::nn::Module>> expected = {
      model[0], model[1], model[2]};
  ASSERT_EQ(modules.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    // Assert pointer equality.
    ASSERT_EQ(modules[i].key(), std::to_string(i));
    ASSERT_EQ(modules[i].value().get(), expected[i].get());
  }
}

TEST_F(ModuleTest, ParametersReturnsExpectedTensorsForFlatModel) {
  TestModule module(1);
  std::vector<torch::Tensor> parameters = module.parameters();
  ASSERT_EQ(parameters.size(), 2);
  ASSERT_EQ(parameters[0].data_ptr<float>(), module.p1.data_ptr<float>());
  ASSERT_EQ(parameters[1].data_ptr<float>(), module.p2.data_ptr<float>());
}

TEST_F(ModuleTest, NamedParametersReturnsExpectedTensorsForFlatModel) {
  TestModule module(1);
  torch::OrderedDict<std::string, torch::Tensor> parameters =
      module.named_parameters();
  ASSERT_EQ(parameters.size(), 2);
  ASSERT_EQ(parameters[0].key(), "p1");
  ASSERT_EQ(parameters[0]->data_ptr<float>(), module.p1.data_ptr<float>());
  ASSERT_EQ(parameters[1].key(), "p2");
  ASSERT_EQ(parameters[1]->data_ptr<float>(), module.p2.data_ptr<float>());
}

TEST_F(ModuleTest, BuffersReturnsExpectedTensorsForFlatModel) {
  TestModule module(1);
  std::vector<torch::Tensor> buffers = module.buffers();
  ASSERT_EQ(buffers.size(), 2);
  ASSERT_EQ(buffers[0].data_ptr<float>(), module.b1.data_ptr<float>());
  ASSERT_EQ(buffers[1].data_ptr<float>(), module.b2.data_ptr<float>());
}

TEST_F(ModuleTest, NamedBuffersReturnsExpectedTensorsForFlatModel) {
  TestModule module(1);
  torch::OrderedDict<std::string, torch::Tensor> buffers =
      module.named_buffers();
  ASSERT_EQ(buffers.size(), 2);
  ASSERT_EQ(buffers[0].key(), "b1");
  ASSERT_EQ(buffers[0]->data_ptr<float>(), module.b1.data_ptr<float>());
  ASSERT_EQ(buffers[1].key(), "b2");
  ASSERT_EQ(buffers[1]->data_ptr<float>(), module.b2.data_ptr<float>());
}

struct TestContainer : torch::nn::Module {
  TestContainer(int64_t number, std::vector<TestContainer> modules = {})
      : tensor(torch::tensor(number)) {
    for (size_t i = 0; i < modules.size(); ++i) {
      register_module(
          std::to_string(i),
          std::make_shared<TestContainer>(std::move(modules[i])));
    }
  }
  torch::Tensor tensor;
};

int64_t get_test_container_item(std::shared_ptr<torch::nn::Module> module) {
  return std::dynamic_pointer_cast<TestContainer>(module)
      ->tensor.item<int64_t>();
}

std::shared_ptr<TestContainer> make_deeply_nested_test_container() {
  return std::make_shared<TestContainer>(TestContainer(
      0,
      {TestContainer(1, {TestContainer(2), TestContainer(3)}),
       TestContainer(4),
       TestContainer(
           5,
           {TestContainer(6),
            TestContainer(7, {TestContainer(8), TestContainer(9)})})}));
}

std::vector<std::pair<std::string, int64_t>>
make_key_value_pairs_for_deeply_nested_container() {
  return {{"test_prefix", 0},
          {"test_prefix.0", 1},
          {"test_prefix.0.0", 2},
          {"test_prefix.0.1", 3},
          {"test_prefix.1", 4},
          {"test_prefix.2", 5},
          {"test_prefix.2.0", 6},
          {"test_prefix.2.1", 7},
          {"test_prefix.2.1.0", 8},
          {"test_prefix.2.1.1", 9}};
}

TEST_F(ModuleTest, ModulesReturnsExpectedSubmodulesForDeepModel) {
  auto model = make_deeply_nested_test_container();
  std::vector<std::shared_ptr<torch::nn::Module>> modules = model->modules();

  ASSERT_EQ(modules.size(), 10);
  for (size_t i = 0; i < modules.size(); ++i) {
    ASSERT_EQ(get_test_container_item(modules[i]), i);
  }
}

TEST_F(ModuleTest, NamedModulesReturnsExpectedNamedSubmodulesForDeepModel) {
  auto model = make_deeply_nested_test_container();
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_modules(/*name_prefix=*/"test_prefix");
  auto expected = make_key_value_pairs_for_deeply_nested_container();

  ASSERT_EQ(modules.size(), expected.size());

  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(modules[i].key(), expected[i].first);
    ASSERT_EQ(get_test_container_item(modules[i].value()), expected[i].second);
  }
}

TEST_F(ModuleTest, ChildrensReturnsExpectedSubmodulesForDeepModel) {
  auto model = make_deeply_nested_test_container();
  std::vector<std::shared_ptr<torch::nn::Module>> modules = model->children();

  ASSERT_EQ(modules.size(), 3);
  ASSERT_EQ(get_test_container_item(modules[0]), 1);
  ASSERT_EQ(get_test_container_item(modules[1]), 4);
  ASSERT_EQ(get_test_container_item(modules[2]), 5);
}

TEST_F(ModuleTest, NamedChildrensReturnsExpectedNamedSubmodulesForDeepModel) {
  auto model = make_deeply_nested_test_container();
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> modules =
      model->named_children();

  ASSERT_EQ(modules.size(), 3);

  ASSERT_EQ(get_test_container_item(modules[0].value()), 1);
  ASSERT_EQ(modules[0].key(), "0");

  ASSERT_EQ(get_test_container_item(modules[1].value()), 4);
  ASSERT_EQ(modules[1].key(), "1");

  ASSERT_EQ(get_test_container_item(modules[2].value()), 5);
  ASSERT_EQ(modules[2].key(), "2");
}

TEST_F(ModuleTest, ModuleApplyIteratesCorreclty) {
  auto model = make_deeply_nested_test_container();
  int64_t index = 0;
  model->apply([&index](torch::nn::Module& module) {
    ASSERT_EQ(module.as<TestContainer>()->tensor.item<int64_t>(), index++);
  });
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, ConstModuleApplyIteratesCorreclty) {
  std::shared_ptr<const TestContainer> model =
      make_deeply_nested_test_container();
  int64_t index = 0;
  model->apply([&index](const torch::nn::Module& module) {
    ASSERT_EQ(module.as<TestContainer>()->tensor.item<int64_t>(), index++);
  });
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, NamedModuleApplyIteratesCorreclty) {
  auto model = make_deeply_nested_test_container();
  auto expected = make_key_value_pairs_for_deeply_nested_container();
  int64_t index = 0;
  model->apply(
      [&index, expected](const std::string& name, torch::nn::Module& module) {
        ASSERT_EQ(name, expected[index].first);
        ASSERT_EQ(
            module.as<TestContainer>()->tensor.item<int64_t>(),
            expected[index++].second);
      },
      /*name_prefix=*/"test_prefix");
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, ConstNamedModuleApplyIteratesCorreclty) {
  std::shared_ptr<const TestContainer> model =
      make_deeply_nested_test_container();
  auto expected = make_key_value_pairs_for_deeply_nested_container();
  int64_t index = 0;
  model->apply(
      [&index, &expected](
          const std::string& name, const torch::nn::Module& module) {
        ASSERT_EQ(name, expected[index].first);
        ASSERT_EQ(
            module.as<const TestContainer>()->tensor.item<int64_t>(),
            expected[index++].second);
      },
      /*name_prefix=*/"test_prefix");
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, ModulePointerApplyIteratesCorreclty) {
  auto model = make_deeply_nested_test_container();
  int64_t index = 0;
  model->apply([&index](const std::shared_ptr<torch::nn::Module>& module) {
    ASSERT_EQ(get_test_container_item(module), index++);
  });
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, NamedModulePointerApplyIteratesCorreclty) {
  auto model = make_deeply_nested_test_container();
  auto expected = make_key_value_pairs_for_deeply_nested_container();
  int64_t index = 0;
  model->apply(
      [&index, &expected](
          const std::string& name,
          const std::shared_ptr<torch::nn::Module>& module) {
        ASSERT_EQ(name, expected[index].first);
        ASSERT_EQ(get_test_container_item(module), expected[index++].second);
      },
      /*name_prefix=*/"test_prefix");
  ASSERT_EQ(index, 10);
}

TEST_F(ModuleTest, ThrowsWhenAttemptingtoGetTopLevelModuleAsSharedPtr) {
  {
    TestModule module(1);
    ASSERT_THROWS_WITH(
        module.modules(),
        "It looks like you attempted to retrieve "
        "your top-level module as a shared_ptr")
  }
  {
    TestModule module(1);
    ASSERT_NO_THROW(module.modules(/*include_self=*/false));
  }
  {
    auto module = std::make_shared<TestModule>(1);
    ASSERT_NO_THROW(module->modules());
  }
}

struct EmptyModule : torch::nn::Module {};

TEST_F(ModuleTest, PrettyPrint) {
  struct TestModule : torch::nn::Module {
    TestModule(int x, float y) : x_(x), y_(y) {}

    void pretty_print(std::ostream& stream) const override {
      stream << "TestModule(x=" << x_ << ", y=" << y_ << ")";
    }

    int x_;
    float y_;
  };


  ASSERT_EQ(c10::str(EmptyModule{}), "EmptyModule");
  ASSERT_EQ(c10::str(TestModule(1, 3.14)), "TestModule(x=1, y=3.14)");
}

struct ModuleWithNonTensorForwardImpl : torch::nn::Module {
  int64_t forward(torch::Tensor x) {
    return x.numel();
  }
};
TORCH_MODULE(ModuleWithNonTensorForward);

TEST_F(ModuleTest, CanCallForwardOnNonTensorForwardThroughPimpl) {
  ModuleWithNonTensorForward m;
  ASSERT_EQ(m(torch::ones(123)), 123);
}
