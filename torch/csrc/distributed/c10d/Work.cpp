#include <ATen/ThreadLocalState.h>

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <utility>

namespace c10d {

Work::Work(
    int rank,
    OpType opType,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputTensors)
    : rank_(rank), opType_(opType) {
  if (profilingTitle != nullptr) {
    auto recordingFunction =
        std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (recordingFunction->isActive()) {
      // Work events follow a future like pattern and can potentially be marked
      // as complete by different threads, so explicitly set as async event.
      recordingFunction->_setAsync();
      // Passing input tensor to recordFunction allows for shape information in
      // profiling output.
      std::vector<c10::IValue> inputs;
      if (inputTensors) {
        inputs.reserve(inputTensors->size());
        for (const auto& tensor : *inputTensors) {
          inputs.emplace_back(tensor);
        }
      }
      recordingFunction->before(
          profilingTitle,
          c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
      std::function<void()> end_handler = [recordingFunction]() {
        recordingFunction->end();
      };
      recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

OpType Work::retrieveOpType() const {
  return opType_;
}

Work::~Work() = default;

bool Work::isCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_;
}

bool Work::isSuccess() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !exception_;
}

std::exception_ptr Work::exception() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return exception_;
}

int Work::sourceRank() const {
  TORCH_CHECK(
      false,
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
}

std::vector<at::Tensor> Work::result() {
  TORCH_CHECK(false, "result() not implemented.");
}

void Work::synchronize() {}

bool Work::wait(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (timeout == kNoTimeout) {
    // This waits without a timeout.
    cv_.wait(lock, [&] { return completed_; });
  } else {
    // Waits for the user-provided timeout.
    cv_.wait_for(lock, timeout, [&] { return completed_; });
    if (!completed_) {
      // Throw exception if the wait operation timed out and the work was not
      // completed.
      TORCH_CHECK(false, "Operation timed out!");
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void Work::abort() {
  TORCH_CHECK(false, "Work::abort not implemented.");
}

c10::intrusive_ptr<c10::ivalue::Future> Work::getFuture() {
  TORCH_CHECK(false, "Work::getFuture not implemented.")
}

void Work::finish(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = std::move(exception);
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  lock.unlock();
  cv_.notify_all();
}

void Work::finishAndThrow(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = std::move(exception);
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

c10::optional<float> Work::getDuration() const {
  return c10::optional<float>();
}

uint64_t Work::getSequencenumber() const {
  TORCH_CHECK(false, "This Backend doesn't support getSequencenumber.");
}

class FutureWrappingWork : public Work {
 public:
  FutureWrappingWork(c10::intrusive_ptr<c10::ivalue::Future> fut)
      : Work(), _fut(std::move(fut)) {}

  ~FutureWrappingWork() override = default;

  bool isCompleted() override {
    return _fut->completed();
  }

  bool isSuccess() const override {
    return _fut->hasValue();
  }

  std::exception_ptr exception() const override {
    return _fut->exception_ptr();
  }

  int sourceRank() const override {
    TORCH_CHECK(false, "FutureWrappingWork::sourceRank() not implemented");
  }

  std::vector<at::Tensor> result() override {
    return _fut->value().toPyObjectHolder()->extractTensors();
  }

  bool wait(std::chrono::milliseconds timeout) override {
    // FIXME
    TORCH_CHECK(
        timeout == kNoTimeout,
        "FutureWrappingWork::wait() with finite timeout not implemented");
    _fut->wait();
    return true;
  }

  void abort() override {
    TORCH_CHECK(false, "FutureWrappingWork::abort() not implemented");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    return _fut;
  }

 private:
  c10::intrusive_ptr<c10::ivalue::Future> _fut;
};

c10::intrusive_ptr<Work> Work::create_from_future(
    const c10::intrusive_ptr<c10::ivalue::Future>& future) {
  return c10::make_intrusive<FutureWrappingWork>(future);
}

} // namespace c10d
