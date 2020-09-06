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
    // Wait for StuckAsyncOp to run first.
    while (!stuckRun) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw TestError();
    return true;
  }
};

REGISTER_CPU_OPERATOR(Error, ErrorOp);
OPERATOR_SCHEMA(Error).NumInputs(0).NumOutputs(0);

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

struct HandleExecutorThreadExceptionsGuard {
  HandleExecutorThreadExceptionsGuard() {
    globalInit({
        "caffe2",
        "--caffe2_handle_executor_threads_exceptions=1",
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
      args_ptrs.push_back(const_cast<char*>(arg.data()));
    }
    char** new_argv = args_ptrs.data();
    int new_argc = args.size();
    CAFFE_ENFORCE(GlobalInit(&new_argc, &new_argv));
  }
};

TEST(PlanExecutorTest, ErrorAsyncPlan) {
  HandleExecutorThreadExceptionsGuard guard;

  PlanDef plan_def = parallelErrorPlan();
  Workspace ws;
  ASSERT_THROW(ws.RunPlan(plan_def), TestError);
  ASSERT_EQ(cancelCount, 1);
}

} // namespace caffe2

#endif
