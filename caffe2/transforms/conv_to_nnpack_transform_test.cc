#include <gtest/gtest.h>
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/transforms/conv_to_nnpack_transform.h"

namespace caffe2 {

namespace {

using transform::Graph;

TEST(ConvToNNPackTest, TestSimple) {
  NetDef netdef;
  OperatorDef* op;
  op = AddOp(&netdef, "Conv", {"in"}, {"out"});
  op = AddOp(&netdef, "Relu", {"out"}, {"out"});
  op = AddOp(&netdef, "Conv", {"out"}, {"out"}); // if not CPU, won't transform
  op->mutable_device_option()->set_device_type(PROTO_CUDA);
  op = AddOp(&netdef, "Relu", {"out"}, {"out"});
  op = AddOp(&netdef, "Conv", {"out"}, {"out"});
  op->set_engine("NNPACK"); // does not need to be transformed
  op = AddOp(&netdef, "Relu", {"out"}, {"out"});
  op = AddOp(&netdef, "Conv", {"out"}, {"out"});
  op = AddOp(&netdef, "Relu", {"out"}, {"out"});

  auto t = TransformRegistry()->Create("ConvToNNPack");
  NetDef transformed_netdef = t->ApplyTo(netdef);

  int nnpack_count = 0;
  for (auto& op : transformed_netdef.op()) {
    if (op.type() == "Conv" && op.device_option().device_type() == PROTO_CPU) {
      EXPECT_EQ(op.engine(), "NNPACK");
      nnpack_count++;
    }
  }
  EXPECT_EQ(nnpack_count, 3);
  EXPECT_EQ(t->PatternMatch(Graph(netdef)).size(), 2); // should get 2 matches
}

} // namespace

} // namespace Caffe2
