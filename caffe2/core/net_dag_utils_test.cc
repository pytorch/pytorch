#include <gtest/gtest.h>
#include "caffe2/core/net_dag_utils.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

namespace {
class DummySyncOp final : public Operator<CPUContext> {
 public:
  DummySyncOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    return true;
  }
};

class DummyAsyncOp final : public Operator<CPUContext> {
 public:
  DummyAsyncOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    return true;
  }

  bool HasAsyncPart() const override {
    return true;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(DagUtilTestDummySync, DummySyncOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(DagUtilTestDummyAsync, DummyAsyncOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(DagUtilTestDummySync)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(DagUtilTestDummyAsync)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX);

class DagUtilTestContext {
 public:
  DagUtilTestContext(const std::string& spec, Workspace* ws) {
    net_def_ = std::make_shared<NetDef>();
    CAFFE_ENFORCE(TextFormat::ParseFromString(spec, net_def_.get()));
    operator_nodes_ = dag_utils::prepareOperatorNodes(net_def_, ws);
  }

  dag_utils::ExecutionChains computeChains() {
    return dag_utils::computeGroups(operator_nodes_);
  }

 private:
  std::shared_ptr<NetDef> net_def_{nullptr};
  std::vector<dag_utils::OperatorNode> operator_nodes_;
};

void PrintChains(const dag_utils::ExecutionChains& chains) {
  // NOLINTNEXTLINE(performance-for-range-copy,clang-diagnostic-range-loop-construct)
  for (const auto kv : chains) {
    std::stringstream ss;
    ss << kv.first << ": ";
    for (const auto& v : kv.second) {
      ss << v << ", ";
    }
    LOG(INFO) << ss.str();
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(DagUtilTest, Empty) {
  const auto spec = R"DOC(
    name: "test0"
    type: "async_scheduling"
    )DOC";
  Workspace ws;
  DagUtilTestContext t(spec, &ws);
  auto chains = t.computeChains();
  EXPECT_TRUE(chains.empty());
}

// 4 sync ops forming a diamond
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(DagUtilTest, AllSync) {
  const auto spec = R"DOC(
    name: "test1"
    type: "async_scheduling"
    external_input: "in"
    op {
      input: "in"
      output: "n1"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n1"
      output: "n2"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n1"
      output: "n3"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n2"
      input: "n3"
      output: "out"
      type: "DagUtilTestDummySync"
    }
    )DOC";
  Workspace ws;
  ws.CreateBlob("in");
  DagUtilTestContext t(spec, &ws);
  auto chains = t.computeChains();
  dag_utils::ExecutionChains expected{{0, {0, 1, 2, 3}}};
  EXPECT_EQ(chains, expected);
}

// 3 async ops forming an L shape
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(DagUtilTest, AllAsync) {
  const auto spec = R"DOC(
    name: "test2"
    type: "async_scheduling"
    external_input: "in0"
    external_input: "in1"
    op {
      input: "in0"
      output: "n1"
      type: "DagUtilTestDummyAsync"
    }
    op {
      input: "in1"
      output: "n2"
      type: "DagUtilTestDummyAsync"
    }
    op {
      input: "n1"
      output: "n3"
      type: "DagUtilTestDummyAsync"
    }
    )DOC";
  Workspace ws;
  ws.CreateBlob("in0");
  ws.CreateBlob("in1");
  DagUtilTestContext t(spec, &ws);
  auto chains = t.computeChains();
  dag_utils::ExecutionChains expected{{0, {0}}, {1, {1}}, {2, {2}}};
  EXPECT_EQ(chains, expected);
}

// 3 sync ops and 1 async op (#2) forming a diamond
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(DagUtilTest, Mixed0) {
  const auto spec = R"DOC(
    name: "test3"
    type: "async_scheduling"
    external_input: "in"
    op {
      input: "in"
      output: "n1"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n1"
      output: "n2"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n1"
      output: "n3"
      type: "DagUtilTestDummyAsync"
    }
    op {
      input: "n2"
      input: "n3"
      output: "out"
      type: "DagUtilTestDummySync"
    }
    )DOC";
  Workspace ws;
  ws.CreateBlob("in");
  DagUtilTestContext t(spec, &ws);
  auto chains = t.computeChains();
  dag_utils::ExecutionChains expected{{0, {0, 1}}, {2, {2}}, {3, {3}}};
  EXPECT_EQ(chains, expected);
}

// 3 sync ops and 1 async op (#2) forming a Y shape
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(DagUtilTest, Mixed1) {
  const auto spec = R"DOC(
    name: "test3"
    type: "async_scheduling"
    external_input: "in0"
    external_input: "in1"
    op {
      input: "in0"
      output: "n1"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "in1"
      output: "n2"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n1"
      input: "n2"
      output: "n3"
      type: "DagUtilTestDummyAsync"
    }
    op {
      input: "n3"
      output: "out"
      type: "DagUtilTestDummySync"
    }
    )DOC";
  Workspace ws;
  ws.CreateBlob("in0");
  ws.CreateBlob("in1");
  DagUtilTestContext t(spec, &ws);
  auto chains = t.computeChains();
  dag_utils::ExecutionChains expected{{0, {0, 1}}, {2, {2}}, {3, {3}}};
  EXPECT_EQ(chains, expected);
}
// More complicated mixed case. * means async
//  0* -> 1* -> 2
//    |
//  3 -> 4 -> 5
//  |  |
//  |    6
//   - -> 8*
//  7* -/
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(DagUtilTest, Mixed2) {
  const auto spec = R"DOC(
    name: "test4"
    type: "async_scheduling"
    external_input: "in0"
    external_input: "in1"
    external_input: "in2"
    op {
      input: "in0"
      output: "n1"
      type: "DagUtilTestDummyAsync"
    }
    op {
      input: "n1"
      output: "n2"
      type: "DagUtilTestDummyAsync"
    }
    op {
      input: "n2"
      output: "out0"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "in1"
      output: "n3"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n1"
      input: "n3"
      output: "n4"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n4"
      output: "out1"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "n3"
      output: "out2"
      type: "DagUtilTestDummySync"
    }
    op {
      input: "in2"
      output: "n7"
      type: "DagUtilTestDummyAsync"
    }
    op {
      input: "n3"
      input: "n7"
      output: "out3"
      type: "DagUtilTestDummyAsync"
    }
    )DOC";
  Workspace ws;
  ws.CreateBlob("in0");
  ws.CreateBlob("in1");
  ws.CreateBlob("in2");
  DagUtilTestContext t(spec, &ws);
  auto chains = t.computeChains();
  dag_utils::ExecutionChains expected{
      {0, {0}}, {1, {1}}, {3, {3, 6}}, {4, {4, 2, 5}}, {7, {7}}, {8, {8}}};
  EXPECT_EQ(chains, expected);
}
} // namespace caffe2
