#ifndef CAFFE2_OPERATORS_COUNTER_OPS_H
#define CAFFE2_OPERATORS_COUNTER_OPS_H

#include <atomic>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
template <typename T>
class TORCH_API Counter {
 public:
  explicit Counter(T count) : count_(count) {}
  bool countDown() {
    if (count_-- > 0) {
      return false;
    }
    return true;
  }

  T countUp() {
    return count_++;
  }

  T retrieve() const {
    return count_.load();
  }

  T checkIfDone() const {
    return (count_.load() <= 0);
  }

  T reset(T init_count) {
    return count_.exchange(init_count);
  }

 private:
  std::atomic<T> count_;
};

// TODO(jiayq): deprecate these ops & consolidate them with IterOp/AtomicIterOp

template <typename T, class Context>
class CreateCounterOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CreateCounterOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        init_count_(this->template GetSingleArgument<T>("init_count", 0)) {
    CAFFE_ENFORCE_LE(0, init_count_, "negative init_count is not permitted.");
  }

  bool RunOnDevice() override {
    *this->template Output<std::unique_ptr<Counter<T>>>(0) =
        std::unique_ptr<Counter<T>>(new Counter<T>(init_count_));
    return true;
  }

 private:
  T init_count_ = 0;
};

template <typename T, class Context>
class ResetCounterOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit ResetCounterOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        init_count_(this->template GetSingleArgument<T>("init_count", 0)) {
    CAFFE_ENFORCE_LE(0, init_count_, "negative init_count is not permitted.");
  }

  bool RunOnDevice() override {
    auto& counterPtr = this->template Input<std::unique_ptr<Counter<T>>>(0);
    auto previous = counterPtr->reset(init_count_);
    if (OutputSize() == 1) {
      auto* output = Output(0);
      output->Resize();
      *output->template mutable_data<T>() = previous;
    }
    return true;
  }

 private:
  T init_count_;
};

// Will always use TensorCPU regardless the Context
template <typename T, class Context>
class CountDownOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CountDownOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& counterPtr = this->template Input<std::unique_ptr<Counter<T>>>(0);
    auto* output = Output(0);
    output->Resize(std::vector<int>{});
    *output->template mutable_data<bool>() = counterPtr->countDown();
    return true;
  }
};

// Will always use TensorCPU regardless the Context
template <typename T, class Context>
class CheckCounterDoneOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CheckCounterDoneOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& counterPtr = this->template Input<std::unique_ptr<Counter<T>>>(0);
    auto* output = Output(0);
    output->Resize(std::vector<int>{});
    *output->template mutable_data<bool>() = counterPtr->checkIfDone();
    return true;
  }
};

// Will always use TensorCPU regardless the Context
template <typename T, class Context>
class CountUpOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CountUpOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& counterPtr = this->template Input<std::unique_ptr<Counter<T>>>(0);
    auto* output = Output(0);
    output->Resize(std::vector<int>{});
    *output->template mutable_data<T>() = counterPtr->countUp();
    return true;
  }
};

// Will always use TensorCPU regardless the Context
template <typename T, class Context>
class RetrieveCountOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit RetrieveCountOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& counterPtr = this->template Input<std::unique_ptr<Counter<T>>>(0);
    auto* output = Output(0);
    output->Resize(std::vector<int>{});
    *output->template mutable_data<T>() = counterPtr->retrieve();
    return true;
  }
};

} // namespace caffe2
#endif // CAFFE2_OPERATORS_COUNTER_OPS_H_
