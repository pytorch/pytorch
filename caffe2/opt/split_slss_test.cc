#include <caffe2/core/common.h>
#include <caffe2/core/test_utils.h>
#include <caffe2/core/workspace.h>
#include <caffe2/opt/onnxifi_transformer.h>
#include <caffe2/utils/proto_utils.h>

#include <gtest/gtest.h>

using namespace caffe2::testing;
using namespace caffe2;

namespace {
NetDef createTest(
    const std::string& op_type,
    Workspace* ws,
    bool has_weight,
    bool fallback) {
  NetDef net;
  std::vector<std::string> inputs{
      "Data", "Weight", "Idx", "Lengths", "Compressed"};
  if (!has_weight) {
    inputs = {"Data", "Idx", "Lengths", "Compressed"};
  }
  NetMutator(&net).newOp(op_type, inputs, {"Out"});
  auto* b = ws->CreateBlob("Compressed");
  auto* t = BlobGetMutableTensor(b, {1}, at::dtype<int32_t>());
  auto* comp = t->template mutable_data<int32_t>();
  *comp = fallback ? 0 : 1;
  return net;
}

void check(
    const NetDef& net,
    const std::string& op_type,
    bool has_weight,
    bool fallback) {
  const static std::unordered_map<string, string> slss = {
      {"SparseLengthsSum4BitRowwiseSparse", "SparseLengthsSumFused4BitRowwise"},
      {"SparseLengthsWeightedSum4BitRowwiseSparse",
       "SparseLengthsWeightedSumFused4BitRowwise"},
      {"SparseLengthsSum8BitRowwiseSparse", "SparseLengthsSumFused8BitRowwise"},
      {"SparseLengthsWeightedSum8BitRowwiseSparse",
       "SparseLengthsWeightedSumFused8BitRowwise"},
      {"SparseLengthsSum2BitRowwiseSparse", "SparseLengthsSumFused2BitRowwise"},
      {"SparseLengthsWeightedSum2BitRowwiseSparse",
       "SparseLengthsWeightedSumFused2BitRowwise"}};
  if (fallback) {
    EXPECT_EQ(net.op_size(), 1);
    EXPECT_EQ(net.op(0).type(), slss.at(op_type));
    EXPECT_EQ(net.op(0).input_size(), has_weight ? 4 : 3);
    EXPECT_EQ(net.op(0).output_size(), 1);
    EXPECT_EQ(net.op(0).input(0), "Data");
    EXPECT_EQ(net.op(0).input(has_weight ? 2 : 1), "Idx");
    EXPECT_EQ(net.op(0).input(has_weight ? 3 : 2), "Lengths");
    if (has_weight) {
      EXPECT_EQ(net.op(0).input(1), "Weight");
    }
    EXPECT_EQ(net.op(0).output(0), "Out");
  } else {
    EXPECT_EQ(net.op_size(), 2);
    EXPECT_EQ(net.op(0).type(), "SparseLengthsSumSparseLookup");
    EXPECT_EQ(net.op(0).input_size(), has_weight ? 4 : 3);
    EXPECT_EQ(net.op(0).output_size(), has_weight ? 3 : 2);
    EXPECT_EQ(net.op(0).input(0), "Idx");
    EXPECT_EQ(net.op(0).input(1), "Lengths");
    EXPECT_EQ(net.op(0).input(2), "Compressed");
    EXPECT_EQ(net.op(0).output(0), "Idx_decomp");
    EXPECT_EQ(net.op(0).output(1), "Lengths_decomp");
    if (has_weight) {
      EXPECT_EQ(net.op(0).input(3), "Weight");
      EXPECT_EQ(net.op(0).output(2), "Weight_decomp");
    }
    EXPECT_EQ(net.op(1).type(), slss.at(op_type));
    EXPECT_EQ(net.op(1).input_size(), has_weight ? 4 : 3);
    EXPECT_EQ(net.op(1).output_size(), 1);
    EXPECT_EQ(net.op(1).input(0), "Data");
    EXPECT_EQ(net.op(1).input(has_weight ? 2 : 1), "Idx_decomp");
    EXPECT_EQ(net.op(1).input(has_weight ? 3 : 2), "Lengths_decomp");
    if (has_weight) {
      EXPECT_EQ(net.op(1).input(1), "Weight_decomp");
    }
    EXPECT_EQ(net.op(1).output(0), "Out");
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(splitSparseLengthsSumSparse, sweep) {
  std::vector<bool> has_weights = {true, false};
  std::vector<bool> fallbacks = {true, false};
  std::vector<int> bits = {2, 4, 8};
  for (const auto has_weight : has_weights) {
    for (const auto bit : bits) {
      std::string op_type = "SparseLengths";
      op_type += (has_weight ? "WeightedSum" : "Sum");
      op_type += caffe2::to_string(bit);
      op_type += "BitRowwiseSparse";
      for (const auto fallback : fallbacks) {
        Workspace ws;
        auto net = createTest(op_type, &ws, has_weight, fallback);
        splitSparseLengthsSumSparse(&net, ws);
        check(net, op_type, has_weight, fallback);
      }
    }
  }
}
