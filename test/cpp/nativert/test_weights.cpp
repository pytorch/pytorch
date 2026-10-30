#include <gtest/gtest.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/custom_class.h>
#include <torch/torch.h>
#include <memory>

#include <torch/nativert/executor/Placement.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {
class WeightsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static constexpr std::string_view source =
        R"(graph(%foo, %bar, %baz):
%o1, %o2 = aten.foo(self=%foo, target=%bar, alpha=0.1)
return(%o2, %baz)
)";
    graph = stringToGraph(source);
    placement = std::make_unique<Placement>(c10::Device(c10::DeviceType::CPU));
  }
  std::shared_ptr<Graph> graph;
  std::unique_ptr<Placement> placement;
};
TEST_F(WeightsTest, ConstructEmptyStateDict) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  // Check that weights are initialized correctly
  EXPECT_TRUE(weights.parameters().empty());
  EXPECT_TRUE(weights.buffers().empty());
  EXPECT_FALSE(weights.contains("non_existent_weight"));
}
TEST_F(WeightsTest, SetAndGetValue) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  at::Tensor tensor = at::ones({2, 2});
  weights.setValue("added_weight", tensor);
  EXPECT_TRUE(weights.contains("added_weight"));
  EXPECT_EQ(weights.at("added_weight").sizes(), tensor.sizes());
}

TEST_F(WeightsTest, ResolvesGeneratedAotiWrapperPrefix) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  const std::string packedFqn = "merge.submod_0.shared_arch.layers.2.weight";
  weights.setValue(packedFqn, at::ones({2, 2}));

  EXPECT_EQ(
      weights.resolveAotiOriginalFqn(
          "merge.submod_0._run_on_acc_0.shared_arch.layers.2.weight"),
      packedFqn);
  EXPECT_EQ(
      weights.resolveAotiOriginalFqn(
          "merge.submod_0._run_on_gpu_12.shared_arch.layers.2.weight"),
      packedFqn);
}

TEST_F(WeightsTest, ExactAotiFqnWinsOverCanonicalFallback) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  const std::string wrappedFqn = "module._run_on_acc_0.weight";
  weights.setValue("module.weight", at::ones({2, 2}));
  weights.setValue(wrappedFqn, at::zeros({2, 2}));

  EXPECT_EQ(weights.resolveAotiOriginalFqn(wrappedFqn), wrappedFqn);
}

TEST_F(WeightsTest, RejectsAmbiguousAotiWrapperFallback) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  weights.setValue("module._run_on_acc_0.weight", at::ones({2, 2}));
  weights.setValue("module._run_on_acc_1.weight", at::zeros({2, 2}));

  EXPECT_FALSE(weights.resolveAotiOriginalFqn("module.weight").has_value());
}

TEST_F(WeightsTest, DoesNotStripUserNamedWrapperComponents) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  weights.setValue("module.weight", at::ones({2, 2}));

  EXPECT_FALSE(
      weights.resolveAotiOriginalFqn("module._run_on_acc_.weight").has_value());
  EXPECT_FALSE(
      weights.resolveAotiOriginalFqn("module._run_on_gpu_custom.weight")
          .has_value());
}

TEST_F(WeightsTest, ResolvesChangedFqnByStableStorageKey) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights oldWeights(graph.get(), stateDict);
  Weights newWeights(graph.get(), stateDict);
  const std::string oldFqn = "merge.submod_0._run_on_acc_0.layer.weight";
  const std::string newFqn = "merge.rewritten.layer.weight";
  oldWeights.setValue(oldFqn, at::ones({2, 2}));
  newWeights.setValue(newFqn, at::zeros({2, 2}));
  oldWeights.setWeightStorageKeys({{oldFqn, "weight_655"}}, {});
  newWeights.setWeightStorageKeys({{newFqn, "weight_655"}}, {});

  const auto storageKey = oldWeights.getWeightStorageKey(oldFqn);
  ASSERT_TRUE(storageKey.has_value());
  EXPECT_EQ(newWeights.resolveWeightStorageKey(*storageKey), newFqn);
}

TEST_F(WeightsTest, RejectsAmbiguousStableStorageKey) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  weights.setValue("first.weight", at::ones({2, 2}));
  weights.setValue("second.weight", at::ones({2, 2}));
  weights.setWeightStorageKeys(
      {{"first.weight", "shared_weight"}, {"second.weight", "shared_weight"}},
      {});

  const auto storageKey = weights.getWeightStorageKey("first.weight");
  ASSERT_TRUE(storageKey.has_value());
  EXPECT_FALSE(weights.resolveWeightStorageKey(*storageKey).has_value());
}

TEST_F(WeightsTest, NamespacesStateAndConstantStorageKeys) {
  std::unordered_map<std::string, c10::IValue> stateDict;
  Weights weights(graph.get(), stateDict);
  weights.setValue("parameter", at::ones({2, 2}));
  weights.setValue("constant", at::zeros({2, 2}));
  weights.setWeightStorageKeys(
      {{"parameter", "tensor_0"}}, {{"constant", "tensor_0"}});

  const auto parameterKey = weights.getWeightStorageKey("parameter");
  const auto constantKey = weights.getWeightStorageKey("constant");
  ASSERT_TRUE(parameterKey.has_value());
  ASSERT_TRUE(constantKey.has_value());
  EXPECT_NE(*parameterKey, *constantKey);
  EXPECT_EQ(weights.resolveWeightStorageKey(*parameterKey), "parameter");
  EXPECT_EQ(weights.resolveWeightStorageKey(*constantKey), "constant");
}

} // namespace torch::nativert

using namespace ::testing;
struct ContainsTensorDict : torch::CustomClassHolder {
  explicit ContainsTensorDict(at::Tensor t) : t_(t) {}

  explicit ContainsTensorDict(c10::Dict<std::string, at::Tensor> dict) {
    t_ = dict.at(std::string("init_tensor"));
  }

  c10::Dict<std::string, at::Tensor> serialize() const {
    c10::Dict<std::string, at::Tensor> dict;
    dict.insert(std::string("init_tensor"), t_);
    return dict;
  }

  at::Tensor t_;
};

static auto reg =
    torch::class_<ContainsTensorDict>("testing", "ContainsTensorDict")
        .def(torch::init<at::Tensor>())
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<ContainsTensorDict>& self)
                -> c10::Dict<std::string, at::Tensor> {
              return self->serialize();
            },
            // __setstate__
            [](c10::Dict<std::string, at::Tensor> data)
                -> c10::intrusive_ptr<ContainsTensorDict> {
              return c10::make_intrusive<ContainsTensorDict>(std::move(data));
            });

TEST(CustomWeightsTest, TestCustomObjWithContainedTensor) {
  // Save
  auto customObj =
      c10::make_intrusive<ContainsTensorDict>(torch::tensor({1, 2, 3}));
  const auto bytes = torch::jit::pickle_save(c10::IValue(std::move(customObj)));

  // Load
  const auto loadedCustomObj =
      torch::jit::pickle_load_obj(std::string{bytes.begin(), bytes.end()});
  EXPECT_TRUE(loadedCustomObj.isObject());
  EXPECT_EQ(
      loadedCustomObj.to<c10::intrusive_ptr<ContainsTensorDict>>()
          ->t_[0]
          .item<int>(),
      1);
}
