#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <torch/nativert/executor/Executor.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/torch.h>

#include <torch/nativert/kernels/KernelHandlerRegistry.h>

namespace torch::nativert {

/*
 * This is a lightweight version of ModelRunner that executes a model in
 * interpreter mode given a string graph with no weights/attributes
 */
class SimpleTestModelRunner {
 public:
  SimpleTestModelRunner(
      const std::string_view source,
      const ExecutorConfig& config) {
    register_kernel_handlers();
    graph_ = stringToGraph(source);
    weights_ = std::make_shared<Weights>(graph_.get());

    executor_ = std::make_unique<Executor>(config, graph_, weights_);
  }

  std::vector<c10::IValue> run(const std::vector<c10::IValue>& inputs) const {
    return executor_->execute(inputs);
  }

  ProfileMetrics benchmarkIndividualNodes(
      const std::vector<c10::IValue>& inputs) const {
    return executor_->benchmarkIndividualNodes({inputs}, 10, 10);
  }

 private:
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<Executor> executor_;
  std::shared_ptr<Weights> weights_;
};

inline void compareIValue(
    const c10::IValue& expected,
    const c10::IValue& actual,
    bool native = false) {
  if (expected.isTensor()) {
    EXPECT_TRUE(actual.isTensor());
    EXPECT_TRUE(torch::allclose(
        expected.toTensor(),
        actual.toTensor(),
        1e-5,
        1e-8,
        /*equal_nan*/ true));
    if (!native) {
      EXPECT_TRUE(expected.toTensor().strides() == actual.toTensor().strides());
    }
  } else if (expected.isTuple()) {
    EXPECT_TRUE(actual.isTuple());
    auto expected_tuple = expected.toTupleRef().elements();
    auto actual_tuple = actual.toTupleRef().elements();
    ASSERT_TRUE(expected_tuple.size() == actual_tuple.size());
    for (size_t i = 0; i < expected_tuple.size(); i++) {
      compareIValue(expected_tuple[i], actual_tuple[i], native);
    }
  } else if (expected.isList()) {
    EXPECT_TRUE(actual.isList());
    auto expected_list = expected.toList();
    auto actual_list = actual.toList();
    ASSERT_TRUE(expected_list.size() == actual_list.size());
    for (size_t i = 0; i < expected_list.size(); i++) {
      compareIValue(expected_list[i], actual_list[i], native);
    }
  } else if (expected.isGenericDict()) {
    EXPECT_TRUE(actual.isGenericDict());
    auto expected_dict = expected.toGenericDict();
    auto actual_dict = actual.toGenericDict();
    EXPECT_TRUE(expected_dict.size() == actual_dict.size());
    for (auto& expected_kv : expected_dict) {
      auto actual_kv = actual_dict.find(expected_kv.key());
      ASSERT_FALSE(actual_kv == actual_dict.end());
      compareIValue(expected_kv.value(), actual_kv->value(), native);
    }
  } else {
    // Fall back to default comparison from IValue
    EXPECT_TRUE(expected == actual);
  }
}

void compareIValues(
    std::vector<c10::IValue> expected,
    std::vector<c10::IValue> actual,
    bool native = false) {
  ASSERT_TRUE(expected.size() == actual.size());
  for (size_t i = 0; i < expected.size(); i++) {
    compareIValue(expected[i], actual[i], native);
  }
}

inline void testStaticKernelEqualityInternal(
    const SimpleTestModelRunner& modelRunner,
    const SimpleTestModelRunner& staticModelRunner,
    const std::vector<c10::IValue>& args,
    bool native = false) {
  auto expected = modelRunner.run(args);

  auto output = staticModelRunner.run(args);
  compareIValues(expected, output, native);

  // Run again to test the static kernel when outputs IValue are cached in the
  // execution frame
  auto output2 = staticModelRunner.run(args);
  compareIValues(expected, output2, native);
}

void testStaticKernelEquality(
    const std::string_view source,
    const std::vector<c10::IValue>& args,
    bool native = false) {
  ExecutorConfig config;
  config.enableStaticCPUKernels = false;
  SimpleTestModelRunner model(source, config);

  config.enableStaticCPUKernels = true;
  SimpleTestModelRunner staticKernelModel(source, config);

  testStaticKernelEqualityInternal(model, staticKernelModel, args, native);
}

inline void testGraphABEquality(
    const std::string_view graph_a,
    const std::string_view graph_b,
    const std::vector<c10::IValue>& args,
    const ExecutorConfig& config = {},
    bool native = false) {
  SimpleTestModelRunner model_a(graph_a, config);
  auto expected = model_a.run(args);

  SimpleTestModelRunner model_b(graph_b, config);
  auto output = model_b.run(args);

  compareIValues(expected, output, native);
}

inline void testGraphABPerf(
    const std::string_view graph_a,
    const std::string_view graph_b,
    const std::vector<c10::IValue>& args,
    const ExecutorConfig& config = {}) {
  SimpleTestModelRunner model_a(graph_a, config);
  auto resultA = model_a.benchmarkIndividualNodes(args);

  SimpleTestModelRunner model_b(graph_b, config);
  auto resultB = model_b.benchmarkIndividualNodes(args);
  ASSERT_TRUE(resultA.totalTime > resultB.totalTime);
}

} // namespace torch::nativert
