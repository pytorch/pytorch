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
