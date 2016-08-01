#ifndef CAFFE2_OPERATORS_COUNTER_OPS_H
#define CAFFE2_OPERATORS_COUNTER_OPS_H

#include <atomic>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace {
template <typename T>
class Counter {
 public:
  explicit Counter(T count) : count_(count) {}
  bool CountDown() {
    if (count_ > 0) {
      --count_;
      return false;
    }
    return true;
  }

  void reset(T init_count) {
    count_ = init_count;
  }

 private:
  std::atomic<T> count_;
};
}

template <typename T, class Context = CPUContext>
class CreateCounterOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CreateCounterOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        init_count_(OperatorBase::GetSingleArgument<T>("init_count", 0)) {
    CHECK_LE(0, init_count_) << "negative init_count is not permitted.";
  }

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<Counter<T>>>(0) =
        std::unique_ptr<Counter<T>>(new Counter<T>(init_count_));
    return true;
  }

 private:
  T init_count_ = 0;
};

template <typename T, class Context = CPUContext>
class ResetCounterOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ResetCounterOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        init_count_(OperatorBase::GetSingleArgument<T>("init_count", 0)) {
    CHECK_LE(0, init_count_) << "negative init_count is not permitted.";
  }

  bool RunOnDevice() override {
    auto& counterPtr = OperatorBase::Input<std::unique_ptr<Counter<T>>>(0);
    counterPtr->reset(init_count_);
    return true;
  }

 private:
  T init_count_;
};

template <typename T, class Context = CPUContext>
class CountDownOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CountDownOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& counterPtr = OperatorBase::Input<std::unique_ptr<Counter<T>>>(0);
    auto* output = Output(0);
    output->Resize(std::vector<int>{});
    *output->template mutable_data<bool>() = counterPtr->CountDown();
    return true;
  }
};
} // namespace caffe2
#endif // CAFFE2_OPERATORS_COUNTER_OPS_H_
