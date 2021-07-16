#include "caffe2/core/common.h"
#include "caffe2/core/test_utils.h"
#include "caffe2/predictor/emulator/data_filler.h"

#include <gtest/gtest.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(DataFiller, FillNetInputTest) {
  using namespace caffe2::testing;
  using namespace caffe2::emulator;
  caffe2::NetDef net;
  NetMutator(&net)
      .newOp("Concat", {"X0", "X1", "X2"}, {"concat_out", "split_info"})
      .addArgument("axis", 1);

  std::vector<int64_t> input_dim = {30, 20};
  std::vector<std::vector<std::vector<int64_t>>> input_dims = {
      {/* X0 */ input_dim, /* X1 */ input_dim, /* X2 */ input_dim}};
  std::vector<std::vector<std::string>> input_types = {
      {"float", "float", "float"}};
  caffe2::Workspace workspace;
  EXPECT_FALSE(workspace.HasBlob("X0"));
  fillRandomNetworkInputs(net, input_dims, input_types, &workspace);
  EXPECT_TRUE(workspace.HasBlob("X0"));
  EXPECT_EQ(getTensor(workspace, "X0").sizes(), input_dim);
}
