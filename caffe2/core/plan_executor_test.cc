#ifndef ANDROID

#include <gtest/gtest.h>
#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/plan_executor.h"

namespace caffe2 {

TEST(PlanExecutorTest, EmptyPlan) {
  PlanDef plan_def;
  Workspace ws;
  EXPECT_TRUE(ws.RunPlan(plan_def));
}

namespace {
static std::atomic<int> cancelCount{0};
static std::atomic<bool> stuckRun{false};
} // namespace

class StuckBlockingOp final : public Operator<CPUContext> {
 public:
  StuckBlockingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    // StuckBlockingOp runs and notifies ErrorOp.
    stuckRun = true;

    while (!cancelled_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return true;
  }

  void Cancel() override {
    LOG(INFO) << "cancelled StuckBlockingOp.";
    cancelCount += 1;
    cancelled_ = true;
  }

 private:
  std::atomic<bool> cancelled_{false};
};

REGISTER_CPU_OPERATOR(StuckBlocking, StuckBlockingOp);
OPERATOR_SCHEMA(StuckBlocking).NumInputs(0).NumOutputs(0);

class NoopOp final : public Operator<CPUContext> {
 public:
  NoopOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    // notify Error op we've ran.
    stuckRun = true;
    return true;
  }
};

REGISTER_CPU_OPERATOR(Noop, NoopOp);
OPERATOR_SCHEMA(Noop).NumInputs(0).NumOutputs(0);


class StuckAsyncOp final : public Operator<CPUContext> {
 public:
  StuckAsyncOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    // notify Error op we've ran.
    stuckRun = true;
    // explicitly don't call SetFinished so this gets stuck
    return true;
  }

  void CancelAsyncCallback() override {
    LOG(INFO) << "cancelled";
    cancelCount += 1;
  }

  bool HasAsyncPart() const override {
    return true;
  }
};

REGISTER_CPU_OPERATOR(StuckAsync, StuckAsyncOp);
OPERATOR_SCHEMA(StuckAsync).NumInputs(0).NumOutputs(0);

class TestError : public std::exception {
  const char* what() const noexcept override {
    return "test error";
  }
};

class ErrorOp final : public Operator<CPUContext> {
 public:
  ErrorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    // Wait for StuckAsyncOp or StuckBlockingOp to run first.
    while (!stuckRun) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw TestError();
    return true;
  }
};

REGISTER_CPU_OPERATOR(Error, ErrorOp);
OPERATOR_SCHEMA(Error).NumInputs(0).NumOutputs(0);

static std::atomic<int> blockingErrorRuns{0};
class BlockingErrorOp final : public Operator<CPUContext> {
 public:
  BlockingErrorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    // First n op executions should block and then start throwing errors.
    if (blockingErrorRuns.fetch_sub(1) >= 1) {
      LOG(INFO) << "blocking";
      while (true) {
        std::this_thread::sleep_for(std::chrono::hours(10));
      }
    } else {
      LOG(INFO) << "throwing";
      throw TestError();
    }
  }
};

REGISTER_CPU_OPERATOR(BlockingError, BlockingErrorOp);
OPERATOR_SCHEMA(BlockingError).NumInputs(0).NumOutputs(0);

PlanDef parallelErrorPlan() {
  PlanDef plan_def;

  auto* stuck_net = plan_def.add_network();
  stuck_net->set_name("stuck_net");
  stuck_net->set_type("async_scheduling");
  {
    auto* op = stuck_net->add_op();
    op->set_type("StuckAsync");
  }

  auto* error_net = plan_def.add_network();
  error_net->set_name("error_net");
  error_net->set_type("async_scheduling");
  {
    auto op = error_net->add_op();
    op->set_type("Error");
  }

  auto* execution_step = plan_def.add_execution_step();
  execution_step->set_concurrent_substeps(true);
  {
    auto* substep = execution_step->add_substep();
    substep->add_network(stuck_net->name());
  }
  {
    auto* substep = execution_step->add_substep();
    substep->add_network(error_net->name());
  }

  return plan_def;
}

PlanDef parallelErrorPlanWithCancellableStuckNet() {
  // Set a plan with two nets: one stuck net with blocking operator that never
  // returns; one error net with error op that throws.
  PlanDef plan_def;

  auto* stuck_blocking_net = plan_def.add_network();
  stuck_blocking_net->set_name("stuck_blocking_net");
  {
    auto* op = stuck_blocking_net->add_op();
    op->set_type("StuckBlocking");
  }

  auto* error_net = plan_def.add_network();
  error_net->set_name("error_net");
  {
    auto* op = error_net->add_op();
    op->set_type("Error");
  }

  auto* execution_step = plan_def.add_execution_step();
  execution_step->set_concurrent_substeps(true);
  {
    auto* substep = execution_step->add_substep();
    substep->add_network(stuck_blocking_net->name());
  }
  {
    auto* substep = execution_step->add_substep();
    substep->add_network(error_net->name());
  }

  return plan_def;
}

PlanDef reporterErrorPlanWithCancellableStuckNet() {
  // Set a plan with a concurrent net and a reporter net: one stuck net with
  // blocking operator that never returns; one reporter net with error op
  // that throws.
  PlanDef plan_def;

  auto* stuck_blocking_net = plan_def.add_network();
  stuck_blocking_net->set_name("stuck_blocking_net");
  {
    auto* op = stuck_blocking_net->add_op();
    op->set_type("StuckBlocking");
  }

  auto* error_net = plan_def.add_network();
  error_net->set_name("error_net");
  {
    auto* op = error_net->add_op();
    op->set_type("Error");
  }

  auto* execution_step = plan_def.add_execution_step();
  execution_step->set_concurrent_substeps(true);
  {
    auto* substep = execution_step->add_substep();
    substep->add_network(stuck_blocking_net->name());
  }
  {
    auto* substep = execution_step->add_substep();
    substep->set_run_every_ms(1);
    substep->add_network(error_net->name());
  }

  return plan_def;
}

struct HandleExecutorThreadExceptionsGuard {
  HandleExecutorThreadExceptionsGuard(int timeout = 60) {
    globalInit({
        "caffe2",
        "--caffe2_handle_executor_threads_exceptions=1",
        "--caffe2_plan_executor_exception_timeout=" +
            caffe2::to_string(timeout),
    });
  }

  ~HandleExecutorThreadExceptionsGuard() {
    globalInit({
        "caffe2",
    });
  }

  HandleExecutorThreadExceptionsGuard(
      const HandleExecutorThreadExceptionsGuard&) = delete;
  void operator=(const HandleExecutorThreadExceptionsGuard&) = delete;

 private:
  void globalInit(std::vector<std::string> args) {
    std::vector<char*> args_ptrs;
    for (auto& arg : args) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast,performance-inefficient-vector-operation)
      args_ptrs.push_back(const_cast<char*>(arg.data()));
    }
    char** new_argv = args_ptrs.data();
    int new_argc = args.size();
    CAFFE_ENFORCE(GlobalInit(&new_argc, &new_argv));
  }
};

TEST(PlanExecutorTest, ErrorAsyncPlan) {
  HandleExecutorThreadExceptionsGuard guard;

  cancelCount = 0;
  PlanDef plan_def = parallelErrorPlan();
  Workspace ws;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(ws.RunPlan(plan_def), TestError);
  ASSERT_EQ(cancelCount, 1);
}

// death tests not supported on mobile
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
TEST(PlanExecutorTest, BlockingErrorPlan) {
  // TSAN doesn't play nicely with death tests
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
  return;
#endif
#endif

  testing::GTEST_FLAG(death_test_style) = "threadsafe";

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_DEATH(
      [] {
        HandleExecutorThreadExceptionsGuard guard(/*timeout=*/1);

        PlanDef plan_def;

        std::string plan_def_template = R"DOC(
          network {
            name: "net"
            op {
              type: "BlockingError"
            }
          }
          execution_step {
            num_concurrent_instances: 2
            substep {
              network: "net"
            }
          }
        )DOC";

        CAFFE_ENFORCE(
            TextFormat::ParseFromString(plan_def_template, &plan_def));
        Workspace ws;
        blockingErrorRuns = 1;
        ws.RunPlan(plan_def);
        FAIL() << "shouldn't have reached this point";
      }(),
      "failed to stop concurrent workers after exception: test error");
}
#endif

TEST(PlanExecutorTest, ErrorPlanWithCancellableStuckNet) {
  HandleExecutorThreadExceptionsGuard guard;

  cancelCount = 0;
  PlanDef plan_def = parallelErrorPlanWithCancellableStuckNet();
  Workspace ws;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(ws.RunPlan(plan_def), TestError);
  ASSERT_EQ(cancelCount, 1);
}

TEST(PlanExecutorTest, ReporterErrorPlanWithCancellableStuckNet) {
  HandleExecutorThreadExceptionsGuard guard;

  cancelCount = 0;
  PlanDef plan_def = reporterErrorPlanWithCancellableStuckNet();
  Workspace ws;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(ws.RunPlan(plan_def), TestError);
  ASSERT_EQ(cancelCount, 1);
}

PlanDef shouldStopWithCancelPlan() {
  // Set a plan with a looping net with should_stop_blob set and a concurrent
  // net that throws an error. The error should cause should_stop to return
  // false and end the concurrent net.
  PlanDef plan_def;

  auto* should_stop_net = plan_def.add_network();
  {
    auto* op = should_stop_net->add_op();
    op->set_type("Noop");
  }
  should_stop_net->set_name("should_stop_net");
  should_stop_net->set_type("async_scheduling");

  auto* error_net = plan_def.add_network();
  error_net->set_name("error_net");
  {
    auto* op = error_net->add_op();
    op->set_type("Error");
  }

  auto* execution_step = plan_def.add_execution_step();
  execution_step->set_concurrent_substeps(true);
  {
    auto* substep = execution_step->add_substep();
  execution_step->set_concurrent_substeps(true);
    substep->set_name("concurrent_should_stop");
    substep->set_should_stop_blob("should_stop_blob");
    auto* substep2 = substep->add_substep();
    substep2->set_name("should_stop_net");
    substep2->add_network(should_stop_net->name());
    substep2->set_num_iter(10);
  }
  {
    auto* substep = execution_step->add_substep();
    substep->set_name("error_step");
    substep->add_network(error_net->name());
  }

  return plan_def;
}

TEST(PlanExecutorTest, ShouldStopWithCancel) {
  HandleExecutorThreadExceptionsGuard guard;

  stuckRun = false;
  PlanDef plan_def = shouldStopWithCancelPlan();
  Workspace ws;

  Blob* blob = ws.CreateBlob("should_stop_blob");
  Tensor* tensor = BlobGetMutableTensor(blob, CPU);
  const vector<int64_t>& shape{1};
  tensor->Resize(shape);
  tensor->mutable_data<bool>()[0] = false;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(ws.RunPlan(plan_def), TestError);
  ASSERT_TRUE(stuckRun);
}

} // namespace caffe2

#endif
