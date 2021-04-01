#include <ATen/ThreadLocalState.h>
#include <c10d/ProcessGroup.hpp>

#include <c10/util/Logging.h>

namespace c10d {

std::string opTypeToString(OpType opType) {
  switch (opType) {
    case OpType::BROADCAST:
      return "BROADCAST";
    case OpType::ALLREDUCE:
      return "ALLREDUCE";
    case OpType::ALLREDUCE_COALESCED:
      return "ALLREDUCE_COALESCED";
    case OpType::REDUCE:
      return "REDUCE";
    case OpType::ALLGATHER:
      return "ALLGATHER";
    case OpType::ALLGATHER_BASE:
      return "ALLGATHER_BASE";
    case OpType::ALLGATHER_COALESCED:
      return "ALLGATHER_COALESCED";
    case OpType::GATHER:
      return "GATHER";
    case OpType::SCATTER:
      return "SCATTER";
    case OpType::REDUCE_SCATTER:
      return "REDUCE_SCATTER";
    case OpType::ALLTOALL_BASE:
      return "ALLTOALL_BASE";
    case OpType::ALLTOALL:
      return "ALLTOALL";
    case OpType::SEND:
      return "SEND";
    case OpType::RECV:
      return "RECV";
    case OpType::RECVANYSOURCE:
      return "RECVANYSOURCE";
    case OpType::BARRIER:
      return "BARRIER";
    case OpType::UNKNOWN:
      return "UNKNOWN";
    default:
      TORCH_INTERNAL_ASSERT("Unknown op type!");
  }
  return "UNKNOWN";
}

bool isP2POp(OpType opType) {
  return opType == OpType::SEND || opType == OpType::RECV ||
      opType == OpType::RECVANYSOURCE;
}

ProcessGroup::Work::Work(
    int rank,
    OpType opType,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputTensors)
    : rank_(rank), opType_(opType) {
  if (profilingTitle != nullptr) {
    auto recordingFunction =
        std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (recordingFunction->isActive()) {
      // Passing input tensor to recordFunction allows for shape information in
      // profiling output.
      std::vector<c10::IValue> inputs;
      if (inputTensors) {
        inputs.reserve(inputTensors->size());
        for (const auto& tensor : *inputTensors) {
          inputs.push_back(tensor);
        }
      }
      recordingFunction->before(profilingTitle, inputs);
      std::function<void()> end_handler = [this, recordingFunction]() {
        recordingFunction->end();
      };
      recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

OpType ProcessGroup::Work::retrieveOpType() {
  return opType_;
}

ProcessGroup::Work::~Work() {}

bool ProcessGroup::Work::isCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_;
}

bool ProcessGroup::Work::isSuccess() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !exception_;
}

std::exception_ptr ProcessGroup::Work::exception() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return exception_;
}

int ProcessGroup::Work::sourceRank() const {
  throw std::runtime_error(
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
}

std::vector<at::Tensor> ProcessGroup::Work::result() {
  throw std::runtime_error("result() not implemented.");
}

void ProcessGroup::Work::synchronize() {}

bool ProcessGroup::Work::wait(std::chrono::milliseconds timeout) {
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
      throw std::runtime_error("Operation timed out!");
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroup::Work::abort() {
  TORCH_CHECK(false, "ProcessGroup::Work::abort not implemented.");
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroup::Work::getFuture() {
  TORCH_CHECK(false, "ProcessGroup::Work::getFuture not implemented.")
}

void ProcessGroup::Work::finish(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = exception;
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  lock.unlock();
  cv_.notify_all();
}

void ProcessGroup::Work::finishAndThrow(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = exception;
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

ProcessGroup::ProcessGroup(int rank, int size) : rank_(rank), size_(size) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
}

ProcessGroup::~ProcessGroup() {}

// This is introduced so that implementors of ProcessGroup would not need to
// have this implmentation.
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroup::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* usused */,
    std::vector<at::Tensor>& /* usused */,
    const AllgatherOptions& /* usused */) {
  throw std::runtime_error(
      "no support for allgather_coalesced in this process group");
}

} // namespace c10d
