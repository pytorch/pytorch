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

// Added specifically for AOT and LazyTensor.
at::Tensor broadcast_(const at::Tensor& tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t root_rank, int64_t root_tensor, int64_t timeout) {
  std::vector<at::Tensor> tensors{tensor};
  auto work = process_group->broadcast_impl(tensors,
      BroadcastOptions {root_rank, root_tensor, std::chrono::milliseconds(timeout)});
  work->wait();
  return tensor;
}

at::Tensor allreduce_(const at::Tensor& tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t reduce_op, int64_t timeout) {
  std::vector<at::Tensor> tensors{tensor};
  auto work = process_group->allreduce_impl(tensors,
      AllreduceOptions {static_cast<ReduceOp>(reduce_op), std::chrono::milliseconds(timeout)});
  work->wait();
  return tensor;
}

std::vector<at::Tensor> allgather_(const std::vector<at::Tensor>& output_tensors, const at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t timeout) {
  std::vector<std::vector<at::Tensor>> output_tensorss{output_tensors};
  std::vector<at::Tensor> input_tensors{input_tensor};
  auto work = process_group->allgather_impl(output_tensorss, input_tensors,
      AllgatherOptions {std::chrono::milliseconds(timeout)});
  work->wait();
  return output_tensors;
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
  m.def("allreduce_", dispatch(c10::DispatchKey::CompositeExplicitAutograd, allreduce_));
  m.def("allgather_", dispatch(c10::DispatchKey::CompositeExplicitAutograd, allgather_));
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
  // static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("c10d::broadcast", "")
  //     .typed<c10::intrusive_ptr<::c10d::ProcessGroup::Work>(at::TensorList,
  //         const c10::intrusive_ptr<::c10d::ProcessGroup>&, int64_t, int64_t, int64_t)>();
  // // It's awakward to unbox the opts here and box them again in the custom C++ op.
  // // But it's also complicated to make opts as a CustomClassHolder. Leave it as it is now.
  // return op.call(tensors, c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this), opts.rootRank,
  //     opts.rootTensor, opts.timeout.count());

  // This code basically is copied from the last Lazy PR.
  // AoTAutograd has a similar problem as LazyTensor where it expects the return type to be tensor|(tensor)|[tensor].
  // Then it wraps the returns with the fx.Proxy which can then trace the op.
  // So here we use directly broadcast_ above instead.
  TORCH_CHECK(tensors.size() == 1lu, "Only one tensor input is supported for c10d::broadcast.")
  auto& tensor = tensors[0];
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("c10d::broadcast_", "")
      .typed<at::Tensor(const at::Tensor&,
          const c10::intrusive_ptr<::c10d::ProcessGroup>&, int64_t, int64_t, int64_t)>();
  op.call(tensor, c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this), opts.rootRank,
      opts.rootTensor, opts.timeout.count());
  auto future = c10::make_intrusive<at::ivalue::Future>(c10::TensorType::get(),
      std::vector<c10::Device>{tensor.device()});
  future->markCompleted(tensor);
  auto work = c10::make_intrusive<c10d::ProcessGroup::Work>(getRank(), c10d::OpType::BROADCAST,
      /*profilingTitle=*/nullptr, tensors);
  work->setFuture(std::move(future));
  work->finish();
  return work;
}


c10::intrusive_ptr<ProcessGroup::Work> ProcessGroup::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1lu, "Only one tensor input is supported for c10d::allreduce.")
  auto& tensor = tensors[0];
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("c10d::allreduce_", "")
      .typed<at::Tensor(const at::Tensor&,
          const c10::intrusive_ptr<::c10d::ProcessGroup>&, int64_t, int64_t)>();
  op.call(tensor, c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this), static_cast<uint64_t>(opts.reduceOp),
      opts.timeout.count());
  auto future = c10::make_intrusive<at::ivalue::Future>(c10::TensorType::get(),
      std::vector<c10::Device>{tensor.device()});
  future->markCompleted(tensor);
  auto work = c10::make_intrusive<c10d::ProcessGroup::Work>(getRank(), c10d::OpType::ALLREDUCE,
      /*profilingTitle=*/nullptr, tensors);
  work->setFuture(std::move(future));
  work->finish();
  return work;
}

  c10::intrusive_ptr<ProcessGroup::Work> ProcessGroup::allgather(
      std::vector<std::vector<at::Tensor>>& output_tensorss,
      std::vector<at::Tensor>& input_tensors,
      const AllgatherOptions& opts) {
  TORCH_CHECK(output_tensorss.size() == 1lu, "Only one tensor output is supported for c10d::allreduce.");
  auto& output_tensors = output_tensorss[0];
  TORCH_CHECK(input_tensors.size() == 1lu, "Only one tensor input is supported for c10d::allreduce.");
  auto& input_tensor = input_tensors[0];

  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("c10d::allgather_", "")
      .typed<std::vector<at::Tensor>(const std::vector<at::Tensor>&, const at::Tensor&,
          const c10::intrusive_ptr<::c10d::ProcessGroup>&, int64_t)>();
  op.call(output_tensors, input_tensor, c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
      opts.timeout.count());

  std::vector<c10::Device> devices;
  for (auto& tensor : output_tensors) {
    devices.push_back(tensor.device());
  }
  auto future = c10::make_intrusive<at::ivalue::Future>(c10::ListType::create(c10::TensorType::get()), devices);
  future->markCompleted(output_tensors);
  auto work = c10::make_intrusive<c10d::ProcessGroup::Work>(getRank(), c10d::OpType::ALLGATHER,
      /*profilingTitle=*/nullptr, input_tensors);
  work->setFuture(std::move(future));
  work->finish();
  return work;
}

} // namespace c10d
