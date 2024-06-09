#include <gtest/gtest.h>
#include "c10/util/StringUtil.h"
#include "caffe2/core/net.h"
#include "caffe2/core/net_async_scheduling.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"

#include <google/protobuf/text_format.h>

namespace caffe2 {

namespace {

// A net test dummy op that does nothing but scaffolding. Here, we
// inherit from OperatorBase because we instantiate on both CPU and
// GPU. In general, you want to only inherit from Operator<Context>.
class NetSimpleRefCountTestOp final : public Operator<CPUContext> {
 public:
  NetSimpleRefCountTestOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}
  USE_OPERATOR_FUNCTIONS(CPUContext);

  bool RunOnDevice() override {
    const int32_t& input = OperatorBase::Input<int32_t>(0);
    int32_t* output = OperatorBase::Output<int32_t>(0);
    *output = input + 1;
    return true;
  }
};

REGISTER_CPU_OPERATOR(NetSimpleRefCountTest, NetSimpleRefCountTestOp);

OPERATOR_SCHEMA(NetSimpleRefCountTest).NumInputs(1).NumOutputs(1);

TEST(NetSimpleRefCountTest, TestCorrectness) {
  Workspace ws;
  *(ws.CreateBlob("a")->GetMutable<int32_t>()) = 1;
  NetDef net_def;
  net_def.set_type("simple_refcount");
  net_def.add_op()->CopyFrom(
      CreateOperatorDef("NetSimpleRefCountTest", "", {"a"}, {"b"}));
  net_def.add_op()->CopyFrom(
      CreateOperatorDef("NetSimpleRefCountTest", "", {"b"}, {"c"}));
  net_def.add_op()->CopyFrom(
      CreateOperatorDef("NetSimpleRefCountTest", "", {"b"}, {"d"}));
  net_def.add_op()->CopyFrom(
      CreateOperatorDef("NetSimpleRefCountTest", "", {"c"}, {"e"}));
  // After execution, what should look like is:
  // a = 1
  // b = deallocated
  // c = deallocated
  // d = 3
  // e = 4
  std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  net->Run();
  // Note on ASSERT vs EXPECT: ASSERT will quit directly if condition not
  // met, which is why we guard IsType<> calls with ASSERT so that the
  // subsequent Get() calls do not product an exception.
  ASSERT_TRUE(ws.GetBlob("a")->IsType<int32_t>());
  EXPECT_EQ(ws.GetBlob("a")->Get<int32_t>(), 1);
  EXPECT_EQ(ws.GetBlob("b")->GetRaw(), nullptr);
  EXPECT_EQ(ws.GetBlob("c")->GetRaw(), nullptr);
  ASSERT_TRUE(ws.GetBlob("d")->IsType<int32_t>());
  EXPECT_EQ(ws.GetBlob("d")->Get<int32_t>(), 3);
  ASSERT_TRUE(ws.GetBlob("e")->IsType<int32_t>());
  EXPECT_EQ(ws.GetBlob("e")->Get<int32_t>(), 4);
}

} // namespace
} // namespace caffe2
