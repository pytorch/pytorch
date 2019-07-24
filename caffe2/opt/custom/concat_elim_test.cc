#include "caffe2/core/common.h"
#include "caffe2/core/test_utils.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/custom/concat_elim.h"
#include "caffe2/predictor/emulator/data_filler.h"
#include "caffe2/utils/proto_utils.h"

#include <gtest/gtest.h>

TEST(ConcatElim, BasicNet) {
  using namespace caffe2::testing;
  using namespace caffe2::emulator;
  caffe2::NetDef net;
  NetMutator(&net)
      .newOp("Concat", {"X0", "X1", "X2"}, {"concat_out", "split_info"})
      .addArgument("axis", 1)
      .addArgument("add_axis", 1)
      .newOp("BatchMatMul", {"concat_out", "concat_out"}, {"matmul_out"})
      .addArgument("trans_a", 0)
      .addArgument("trans_b", 1)
      .addArgument("broadcast", 0)
      .newOp("Flatten", {"matmul_out"}, {"flatten_out"})
      .newOp("BatchGather", {"flatten_out", "indices"}, {"out"});

  auto nn = caffe2::convertToNNModule(net);
  caffe2::opt::concatElim(&nn);
  auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
  EXPECT_EQ(optimized_net.op().size(), 1);

  std::vector<int64_t> input_dim = {30, 20};
  std::vector<std::vector<std::vector<int64_t>>> input_dims = {
      {/* X0 */ input_dim, /* X1 */ input_dim, /* X2 */ input_dim},
      {/* indices */ {3}}};
  std::vector<std::vector<std::string>> input_types = {
      {"float", "float", "float"}, {"int"}};
  auto filler = TestDataRandomFiller(net, input_dims, input_types);
  caffe2::Workspace workspace;
  filler.fillInputToWorkspace(&workspace);
  workspace.RunNetOnce(net);
  auto outBefore = getTensor(workspace, "out").Clone();
  workspace.RunNetOnce(optimized_net);
  auto outAfter = getTensor(workspace, "out_cc_bmm_bg").Clone();
  assertTensorEquals(outBefore, outAfter);
}

TEST(ConcatElim, ProdNet) {
  // Test concatElim on a realistic prod model.
  caffe2::NetDef net;
  ReadProtoFromFile("caffe2/caffe2/opt/custom/concat_elim_test_net.pb", &net);
  EXPECT_EQ(net.op().size(), 176);
  auto nn = caffe2::convertToNNModule(net);
  caffe2::opt::concatElim(&nn);
  auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
  EXPECT_EQ(optimized_net.op().size(), 173);
}

TEST(ConcatAddMulNaNClipElim, BasicNet) {
  using namespace caffe2::testing;
  using namespace caffe2::emulator;
  caffe2::NetDef net;
  NetMutator(&net)
      .newOp("Concat", {"X0", "X1", "X2"}, {"concat_out", "split_info"})
      .addArgument("axis", 1)
      .newOp("Add", {"concat_out", "add_in"}, {"add_out"})
      .addArgument("broadcast", 1)
      .newOp("Mul", {"add_out", "mul_in"}, {"mul_out"})
      .addArgument("broadcast", 1)
      .newOp("ReplaceNaN", {"mul_out"}, {"replace_out"})
      .addArgument("value", 0.0001f)
      .newOp("Clip", {"replace_out"}, {"out"});

  auto nn = caffe2::convertToNNModule(net);
  caffe2::opt::concatAddMulNaNClipElim(&nn);
  auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
  EXPECT_EQ(optimized_net.op().size(), 1);

  std::vector<int64_t> input_dim = {30, 20};
  std::vector<std::vector<std::vector<int64_t>>> input_dims = {
      {/* X0 */ input_dim, /* X1 */ input_dim, /* X2 */ input_dim},
      {/* add_in */ {60}},
      {/* mul_in */ {60}}};
  std::vector<std::vector<std::string>> input_types = {
      {"float", "float", "float"}, {"float"}, {"float"}};
  auto filler = TestDataRandomFiller(net, input_dims, input_types);
  caffe2::Workspace workspace;
  filler.fillInputToWorkspace(&workspace);
  workspace.RunNetOnce(net);
  auto outBefore = getTensor(workspace, "out").Clone();
  workspace.RemoveBlob("out");
  workspace.RunNetOnce(optimized_net);
  auto outAfter = getTensor(workspace, "out").Clone();
  assertTensorEquals(outBefore, outAfter);
}

TEST(ConcatAddMulNaNClipElim, ProdNet) {
  // Test ConcatAddMulNaNClipElim on a realistic prod model.
  caffe2::NetDef net;
  ReadProtoFromFile("caffe2/caffe2/opt/custom/test_cc_amcr_net.pb", &net);
  auto size = net.op().size();
  auto nn = caffe2::convertToNNModule(net);
  caffe2::opt::concatAddMulNaNClipElim(&nn);
  auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
  // Ensure that optimization happens: number of ops is smaller than before.
  EXPECT_LT(optimized_net.op().size(), size);
}
