#include <ATen/ThreadLocalState.h>
#include <c10d/ProcessGroup.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>

namespace c10d {
namespace {

c10::intrusive_ptr<ProcessGroup::Work> broadcast(at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t root_rank, int64_t root_tensor, int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->broadcast_impl(tensor_vec,
      BroadcastOptions {root_rank, root_tensor, std::chrono::milliseconds(timeout)});
}

// Added specifically for LazyTensor.
at::Tensor broadcast_(const at::Tensor& tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t root_rank, int64_t root_tensor, int64_t timeout) {
  std::vector<at::Tensor> tensors{tensor};
  auto work = process_group->broadcast_impl(tensors,
      BroadcastOptions {root_rank, root_tensor, std::chrono::milliseconds(timeout)});
  work->wait();
  return tensor;
}

TORCH_LIBRARY(c10d, m) {
  // The following ProcessGroup and Work definations are more like declarations.
  // They don't expose the details of the two classes into TorchScript.
  m.class_<ProcessGroup>("ProcessGroup")
    .def(torch::init<int64_t, int64_t>());
  m.class_<ProcessGroup::Work>("Work")
    .def(torch::init<>());
  // It's important to register the op to the CompositeExplicitAutograd key to enable
  // __torch_dispatch__.
  m.def("broadcast", dispatch(c10::DispatchKey::CompositeExplicitAutograd, broadcast));
  m.def("broadcast_", dispatch(c10::DispatchKey::CompositeExplicitAutograd, broadcast_));
}

}  // namespace

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
    case OpType::_ALLGATHER_BASE:
      return "_ALLGATHER_BASE";
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
    case OpType::_REDUCE_SCATTER_BASE:
      return "_REDUCE_SCATTER_BASE";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown op type!");
  }
  return "UNKNOWN";
}

bool isP2POp(OpType opType, bool batchP2P /*= false*/) {
  if (batchP2P) return false;
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
      recordingFunction->before(profilingTitle, inputs);
      std::function<void()> end_handler = [recordingFunction]() {
        recordingFunction->end();
      };
      recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

OpType ProcessGroup::Work::retrieveOpType() {
  return opType_;
}

ProcessGroup::Work::~Work()=default;

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
  TORCH_CHECK(false,
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
}

std::vector<at::Tensor> ProcessGroup::Work::result() {
  TORCH_CHECK(false, "result() not implemented.");
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

void ProcessGroup::Work::abort() {
  TORCH_CHECK(false, "ProcessGroup::Work::abort not implemented.");
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

ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size), dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
}

ProcessGroup::~ProcessGroup() {}

void ProcessGroup::init() {
  C10_LOG_API_USAGE_ONCE(fmt::format("c10d.process_group_{}", getBackendName()));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroup::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("c10d::broadcast", "")
      .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(at::TensorList,
          const c10::intrusive_ptr<::c10d::ProcessGroup>&, int64_t, int64_t, int64_t)>();
  // It's awakward to unbox the opts here and box them again in the custom C++ op.
  // But it's also complicated to make opts as a CustomClassHolder. Leave it as it is now.
  return op.call(tensors, c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this), opts.rootRank,
      opts.rootTensor, opts.timeout.count());
}

} // namespace c10d
