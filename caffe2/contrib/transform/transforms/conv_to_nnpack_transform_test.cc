#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include "caffe2/contrib/transform/transform.h"
#include "caffe2/contrib/transform/transforms/conv_to_nnpack_transform.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

namespace {

using transform::Graph;

// Adds an operator def to a netdef.
// Returns the ptr, if you want to add anything extra (such as device_option)
OperatorDef* AddOp(
    NetDef* netdef_ptr,
    string op_type,
    std::vector<string> inputs,
    std::vector<string> outputs) {
  CHECK(netdef_ptr);
  auto& netdef = *netdef_ptr;
  auto op_ptr = netdef.add_op();
  auto& op = *op_ptr;
  op.set_type(op_type);
  for (const string& inp : inputs) {
    op.add_input(inp);
  }
  for (const string& outp : outputs) {
    op.add_output(outp);
  }
  return op_ptr;
}

TEST(ConvToNNPackTest, TestSimple) {
  NetDef netdef;
  OperatorDef* op;
  op = AddOp(&netdef, "Conv", {"out"}, {"out"});
  op->mutable_device_option()->set_device_type(CUDA);
  op = AddOp(&netdef, "Relu", {"out"}, {"out"});
  op = AddOp(&netdef, "Conv", {"in"}, {"out"}); // if not CUDA, won't transform
  op = AddOp(&netdef, "Relu", {"out"}, {"out"});
  op = AddOp(&netdef, "Conv", {"out"}, {"out"});
  op->mutable_device_option()->set_device_type(CUDA);
  op->set_engine("NNPACK"); // does not need to be transformed
  op = AddOp(&netdef, "Relu", {"out"}, {"out"});
  op = AddOp(&netdef, "Conv", {"out"}, {"out"});
  op->mutable_device_option()->set_device_type(CUDA);
  op = AddOp(&netdef, "Relu", {"out"}, {"out"});

  auto t = TransformRegistry()->Create("ConvToNNPack");
  NetDef transformed_netdef = t->ApplyTo(netdef);

  int nnpack_count = 0;
  for (auto& op : transformed_netdef.op()) {
    if (op.type() == "Conv" && op.device_option().device_type() == CUDA) {
      EXPECT_EQ(op.engine(), "NNPACK");
      nnpack_count++;
    }
  }
  EXPECT_EQ(nnpack_count, 3);
  EXPECT_EQ(t->PatternMatch(Graph(netdef)).size(), 2); // should get 2 matches
}

} // namespace

} // namespace Caffe2
