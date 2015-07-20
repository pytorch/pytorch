#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

namespace caffe2 {

const char kBcastNet[] =
"  name: \"bcast\""
"  op {"
"    output: \"X\""
"    type: \"ConstantFill\""
"    arg {"
"      name: \"shape\""
"      ints: 10"
"    }"
"    arg {"
"      name: \"value\""
"      f: 0.0"
"    }"
"  }"
"  op {"
"    input: \"X\""
"    output: \"X\""
"    type: \"Broadcast\""
"    arg {"
"      name: \"root\""
"      i: 0"
"    }"
"  }";

TEST(MPITest, TestBroadcast) {
  NetDef net_def;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      string(kBcastNet), &net_def));
  // Let's set the network's constant fill value to be the mpi rank.
  auto* arg = net_def.mutable_op(0)->mutable_arg(1);
  CHECK_EQ(arg->name(), "value");
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  arg->set_f(rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int root = 0; root < size; ++root) {
    net_def.mutable_op(1)->mutable_arg(0)->set_i(root);
    Workspace ws;
    unique_ptr<NetBase> net(CreateNet(net_def, &ws));
    EXPECT_NE(nullptr, net.get());
    EXPECT_TRUE(net->Verify());
    EXPECT_TRUE(net->Run());
    // Let's test the value.
    auto& X = ws.GetBlob("X")->Get<Tensor<float, CPUContext> >();
    EXPECT_EQ(X.size(), 10);
    for (int i = 0; i < X.size(); ++i) {
      EXPECT_EQ(X.data()[i], root);
    }
  }
}

const char kAllreduceNet[] =
"  name: \"allreduce\""
"  op {"
"    output: \"X\""
"    type: \"ConstantFill\""
"    arg {"
"      name: \"shape\""
"      ints: 10"
"    }"
"    arg {"
"      name: \"value\""
"      f: 0.0"
"    }"
"  }"
"  op {"
"    input: \"X\""
"    output: \"X_reduced\""
"    type: \"Allreduce\""
"  }";

TEST(MPITest, TestAllreduce) {
  NetDef net_def;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      string(kAllreduceNet), &net_def));
  // Let's set the network's constant fill value to be the mpi rank.
  auto* arg = net_def.mutable_op(0)->mutable_arg(1);
  CHECK_EQ(arg->name(), "value");
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  arg->set_f(rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Workspace ws;
  unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  EXPECT_NE(nullptr, net.get());
  EXPECT_TRUE(net->Verify());
  EXPECT_TRUE(net->Run());
  // Let's test the value.
  auto& X = ws.GetBlob("X")->Get<Tensor<float, CPUContext> >();
  EXPECT_EQ(X.size(), 10);
  for (int i = 0; i < X.size(); ++i) {
    EXPECT_EQ(X.data()[i], rank);
  }
  auto& X_reduced = ws.GetBlob("X_reduced")->Get<Tensor<float, CPUContext> >();
  EXPECT_EQ(X_reduced.size(), 10);
  int expected_result = size * (size - 1) / 2;
  for (int i = 0; i < X_reduced.size(); ++i) {
    EXPECT_EQ(X_reduced.data()[i], expected_result);
  }
}

}  // namespace caffe2

DEFINE_string(caffe_test_root, "gen/", "The root of the caffe test folder.");

GTEST_API_ int main(int argc, char **argv) {
  int mpi_ret;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_ret);
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  int test_result = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_result;
}
