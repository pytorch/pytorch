#include "caffe2/core/common.h"
#include "caffe2/core/test_utils.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/custom/concat_elim.h"
#include "caffe2/predictor/emulator/data_filler.h"
#include "caffe2/utils/proto_utils.h"

#include <gtest/gtest.h>

using namespace caffe2::testing;
using namespace caffe2::emulator;

TEST(gatherFuse8BitRowwiseQuantFloatMulLengthsSumElim, Basic) {
  using namespace caffe2;
  caffe2::NetDef net;
  NetMutator(&net)
      .newOp("Gather", {"Data0", "Idx"}, {"Gout"})
      .newOp("Fused8BitRowwiseQuantizedToFloat", {"Gout"}, {"Fout"})
      .newOp("Mul", {"Fout", "Min"}, {"Mout"})
      .addArgument("axis", 0)
      .addArgument("broadcast", 1)
      .newOp("LengthsSum", {"Mout", "Len"}, {"Out"});
  auto nn = caffe2::convertToNNModule(net);
  caffe2::opt::gatherFuse8BitRowwiseQuantFloatMulLengthsSumElim(&nn);
  auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
  EXPECT_EQ(optimized_net.op().size(), 1);

  // Fill in a float typed dummy tensor first
  Workspace workspace;
  std::vector<int64_t> lut_dims{20, 20};
  TensorFiller lut_filler(lut_dims);
  const auto dummy = "dummy";
  Blob* blob = workspace.CreateBlob(dummy);
  ::fill_with_type(lut_filler, "float", BlobGetMutableTensor(blob, CPU));
  CAFFE_ENFORCE(workspace.GetBlob(dummy)->GetRaw());
  // Run the FloatToFused8BitRowwiseQuantized operator to do proper
  // quantization
  OperatorDef op_def = CreateOperatorDef(
      "FloatToFused8BitRowwiseQuantized", "", {dummy}, {"Data0"}, {});
  workspace.RunOperatorOnce(op_def);
  auto* blob2 = workspace.GetBlob("Data0");
  CAFFE_ENFORCE(blob2->GetRaw());
  // Fill in the rest of the inputs
  blob = workspace.CreateBlob("Idx");
  auto* t = BlobGetMutableTensor(blob, CPU);
  ReinitializeTensor(t, {5}, at::dtype<int32_t>().device(CPU));
  int32_t* data = t->mutable_data<int32_t>();
  for (int i = 0; i < 5; ++i) {
    data[i] = i;
  }
  blob = workspace.CreateBlob("Min");
  t = BlobGetMutableTensor(blob, CPU);
  ReinitializeTensor(t, {5}, at::dtype<float>().device(CPU));
  float* fdata = t->mutable_data<float>();
  for (int i = 0; i < 5; ++i) {
    fdata[i] = 0.9;
  }
  blob = workspace.CreateBlob("Len");
  t = BlobGetMutableTensor(blob, CPU);
  ReinitializeTensor(t, {2}, at::dtype<int32_t>().device(CPU));
  data = t->mutable_data<int32_t>();
  data[0] = 2;
  data[1] = 3;
  workspace.RunNetOnce(net);
  auto outBefore = getTensor(workspace, "Out").Clone();
  workspace.RemoveBlob("out");
  workspace.RunNetOnce(optimized_net);
  auto outAfter = getTensor(workspace, "Out").Clone();
  assertTensorEquals(outBefore, outAfter);
}

TEST(gatherFuse8BitRowwiseQuantFloatMulLengthsSumElim, NoFuse) {
  using namespace caffe2;
  caffe2::NetDef net;
  NetMutator(&net)
      .newOp("Gather", {"Data0", "Idx"}, {"Gout"})
      .newOp("Fused8BitRowwiseQuantizedToFloat", {"Gout"}, {"Fout"})
      .newOp("Mul", {"Fout", "Min"}, {"Mout"})
      .addArgument("axis", 0)
      .addArgument("broadcast", 1)
      .newOp("LengthsSum", {"Mout", "Len"}, {"Out"})
      .newOp("Copy", {"Fout"}, {"Fout2"});
  auto nn = caffe2::convertToNNModule(net);
  caffe2::opt::gatherFuse8BitRowwiseQuantFloatMulLengthsSumElim(&nn);
  auto optimized_net = caffe2::convertToCaffe2Proto(nn, net);
  EXPECT_EQ(optimized_net.op().size(), 5);
}

TEST(ConcatElim, BasicNet) {
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
