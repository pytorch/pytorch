#include <chrono>  // NOLINT
#include <ctime>
#include <thread>  // NOLINT

#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

namespace caffe2 {

using std::clock_t;
using std::clock;

// SleepOp basically sleeps for a given number of seconds.
class SleepOp final : public OperatorBase {
 public:
  SleepOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        ms_(OperatorBase::GetSingleArgument<int>("ms", 1000)) {
    DCHECK_GT(ms_, 0);
    DCHECK_LT(ms_, 3600 * 1000) << "Really? This long?";
  }

  bool Run() final {
    clock_t start = clock();
    std::this_thread::sleep_for(std::chrono::milliseconds(ms_));
    clock_t end = clock();
    if (OperatorBase::OutputSize()) {
      vector<clock_t>* output = OperatorBase::Output<vector<clock_t> >(0);
      output->resize(2);
      (*output)[0] = start;
      (*output)[1] = end;
    }
    return true;
  }

 private:
  int ms_;
  // We allow arbitrary inputs and at most one output so that we can
  // test scaffolding of networks. If the output is 1, it will be filled with
  // vector<clock_t> with two elements: start time and end time.
  INPUT_OUTPUT_STATS(0, INT_MAX, 0, 1);
  DISABLE_COPY_AND_ASSIGN(SleepOp);
};

namespace {
REGISTER_CPU_OPERATOR(Sleep, SleepOp)
REGISTER_CUDA_OPERATOR(Sleep, SleepOp)
}  // namespace

const char kSleepNetDefString[] =
"  name: \"sleepnet\""
"  net_type: \"parallel\""
"  num_workers: 2"
"  op {"
"    output: \"sleep1\""
"    name: \"sleep1\""
"    type: \"Sleep\""
"    arg {"
"      name: \"ms\""
"      i: 100"
"    }"
"  }"
"  op {"
"    input: \"sleep1\""
"    output: \"sleep2\""
"    name: \"sleep2\""
"    type: \"Sleep\""
"    arg {"
"      name: \"ms\""
"      i: 100"
"    }"
"  }"
"  op {"
"    output: \"sleep3\""
"    name: \"sleep3\""
"    type: \"Sleep\""
"    arg {"
"      name: \"ms\""
"      i: 150"
"    }"
"  }";


TEST(ParallelNetTest, TestParallelNetTiming) {
  NetDef net_def;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      string(kSleepNetDefString), &net_def));
  // Below is the parallel version
  Workspace ws;
  unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  EXPECT_NE(nullptr, net.get());
  EXPECT_TRUE(net->Verify());
  auto start_time = std::chrono::system_clock::now();
  EXPECT_TRUE(net->Run());
  // Inspect the time - it should be around 2000 milliseconds, since sleep3 can
  // run in parallel with sleep1 and sleep2.
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - start_time);
  int milliseconds = duration.count();
  // We should be seeing 200 ms. This adds a little slack time.
  EXPECT_GT(milliseconds, 180);
  EXPECT_LT(milliseconds, 220);
}

// For sanity check, we also test the sequential time - it should take 0.35
// seconds instead since everything has to be sequential.
TEST(SimpleNetTest, TestSimpleNetTiming) {
  NetDef net_def;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      string(kSleepNetDefString), &net_def));
  net_def.set_net_type("simple");
  Workspace ws;
  unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  EXPECT_NE(nullptr, net.get());
  EXPECT_TRUE(net->Verify());
  auto start_time = std::chrono::system_clock::now();
  EXPECT_TRUE(net->Run());
  // Inspect the time - it should be around 2000 milliseconds, since sleep3 can
  // run in parallel with sleep1 and sleep2.
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - start_time);
  int milliseconds = duration.count();
  // We should be seeing 350 ms. This adds a little slack time.
  EXPECT_GT(milliseconds, 330);
  EXPECT_LT(milliseconds, 370);
}


}  // namespace caffe2



