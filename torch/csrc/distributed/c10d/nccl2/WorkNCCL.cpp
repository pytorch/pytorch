// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/WorkNCCL.hpp>

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <iterator>
#include <thread>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/nccl2/TracingGuard.hpp>

namespace c10d::nccl2 {

namespace {
std::atomic<uint64_t> next_completion_key{1};
} // namespace

struct WorkNCCL::Events {
  Events(ProcessGroupNCCL* comm, bool timing_enabled)
      : comm(comm),
        timing_enabled(timing_enabled),
        start(comm->getEvent(timing_enabled)),
        end(comm->getEvent(timing_enabled)) {}

  Events(const Events&) = delete;
  Events(Events&&) = delete;
  Events& operator=(const Events&) = delete;
  Events& operator=(Events&&) = delete;

  ~Events() {
    comm->returnEvent(std::move(start), timing_enabled);
    comm->returnEvent(std::move(end), timing_enabled);
  }

  ProcessGroupNCCL* comm;
  bool timing_enabled;
  std::unique_ptr<at::cuda::CUDAEvent> start;
  std::unique_ptr<at::cuda::CUDAEvent> end;
};

WorkNCCL::InputTensorShelf::InputTensorShelf(std::vector<at::Tensor> tensors)
    : tensors(std::move(tensors)) {}

void WorkNCCL::InputTensorShelf::append(InputTensorShelf& other) {
  std::scoped_lock lock(mutex, other.mutex);
  tensors.insert(
      tensors.end(),
      std::make_move_iterator(other.tensors.begin()),
      std::make_move_iterator(other.tensors.end()));
  other.tensors.clear();
}

void WorkNCCL::InputTensorShelf::clear() {
  std::lock_guard<std::mutex> lock(mutex);
  tensors.clear();
}

WorkNCCL::State::State(
    ProcessGroupNCCL* comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout)
    : comm(comm),
      reconfigure_uuid(comm->reconfigure_uuid_),
      blocking_wait(comm->blocking_wait_),
      stream(
          at::cuda::getStreamFromExternal(stream, comm->getDevice().index())),
      work_start_time(std::chrono::steady_clock::now()),
      timeout(timeout),
      completion_key(
          next_completion_key.fetch_add(1, std::memory_order_relaxed)),
      timing_enabled(comm->collectivesTimingEnabled()),
      events(std::make_shared<Events>(comm, timing_enabled)),
      duration_start_events(events),
      future_work_result(
          c10::make_intrusive<c10::ivalue::Future>(c10::AnyEnumType::get())) {}

WorkNCCL::WorkNCCL(
    ProcessGroupNCCL* comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : state_(std::make_shared<State>(comm, stream, timeout_ms)),
      input_tensors_(std::make_shared<InputTensorShelf>(inputTensors)) {}

WorkNCCL::WorkNCCL(
    ProcessGroupNCCL* comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    at::Tensor inputTensor)
    : state_(std::make_shared<State>(comm, stream, timeout_ms)),
      input_tensors_(std::make_shared<InputTensorShelf>(
          std::vector<at::Tensor>{std::move(inputTensor)})) {}

WorkNCCL::~WorkNCCL() = default;

void WorkNCCL::recordFunctionStart(std::string_view coll_name) {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }

  std::lock_guard<std::mutex> lock(input_tensors_->mutex);
  if (!input_tensors_->tensors.empty()) {
    std::vector<c10::IValue> inputs;
    inputs.reserve(input_tensors_->tensors.size());
    for (const auto& tensor : input_tensors_->tensors) {
      inputs.emplace_back(tensor);
    }
    recordFunction_->before(
        coll_name,
        c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
  } else {
    recordFunction_->before(coll_name, c10::ArrayRef<const c10::IValue>{});
  }
}

void WorkNCCL::recordStart(std::string_view coll_name) {
  recordFunctionStart(coll_name);
  state_->events->start->record(state_->stream);
}

void WorkNCCL::recordEnd() {
  state_->events->end->record(state_->stream);

  if (recordFunction_ && recordFunction_->isActive()) {
    recordFunction_->end();
  }
}

bool WorkNCCL::State::setTerminalStatus(WorkStatus terminal_status) {
  TORCH_INTERNAL_ASSERT(
      terminal_status == WorkStatus::COMPLETED ||
      terminal_status == WorkStatus::TIMEDOUT ||
      terminal_status == WorkStatus::ERROR);

  WorkResult result = WorkResult::SUCCESS;
  {
    std::lock_guard<std::mutex> lock(terminal_status_mutex);
    WorkStatus current = status();
    if (current == WorkStatus::COMPLETED || current == WorkStatus::TIMEDOUT ||
        current == WorkStatus::ERROR) {
      return false;
    }

    if (terminal_status == WorkStatus::TIMEDOUT) {
      result = WorkResult::TIMEOUT;
      work_exception = std::make_exception_ptr(
          C10_BUILD_ERROR(DistBackendError, "NCCL operation timed out"));
    } else if (terminal_status == WorkStatus::ERROR) {
      result = WorkResult::COMM_ERROR;
      work_exception = std::make_exception_ptr(
          C10_BUILD_ERROR(DistBackendError, "NCCL operation failed"));
    }
    work_status.store(terminal_status, std::memory_order_release);
  }
  future_work_result->markCompleted(c10::IValue(static_cast<uint8_t>(result)));
  return true;
}

void WorkNCCL::State::notifyCompletion() {
  // Called once per work, by the queue that popped it as COMPLETED, and only
  // for success -- a timed-out or failed work reports nothing, because a
  // consumer that read that as "finished" would lose the very fact a
  // post-mortem needs.
  if (!comm->hasCompletionHooks()) {
    return;
  }
  std::optional<float> duration;
  if (timing_enabled) {
    try {
      // cudaEventElapsedTime on two events that have already been observed to
      // complete: no synchronization, and no NCCL call, so it is legal on the
      // watchdog thread. Stock ProcessGroupNCCL's watchdog calls getDuration()
      // from its own completion hook for the same reason.
      duration = getDuration();
    } catch (const std::exception& e) {
      TC_LOG(WARNING, comm)
          << "Cannot measure collective duration: " << e.what();
    }
  }
  comm->runCompletionHooks(completion_key, duration);
}

WorkNCCL::WorkStatus WorkNCCL::status() const {
  return state_->status();
}

std::exception_ptr WorkNCCL::exception() const {
  return state_->exception();
}

WorkNCCL::WorkStatus WorkNCCL::State::status() const {
  return work_status.load(std::memory_order_acquire);
}

std::exception_ptr WorkNCCL::State::exception() const {
  std::lock_guard<std::mutex> lock(terminal_status_mutex);
  return work_exception;
}

std::chrono::milliseconds WorkNCCL::getTimeout() const {
  return state_->timeout;
}

void WorkNCCL::setChildren(std::vector<c10::intrusive_ptr<WorkNCCL>> children) {
  if (!children.empty()) {
    std::lock_guard<std::mutex> lock(state_->duration_mutex);
    state_->duration_start_events = children.front()->state_->events;
  }
  for (const auto& child : children) {
    input_tensors_->append(*child->input_tensors_);
  }
}

void WorkNCCL::setSequenceNumber(uint64_t seq) {
  state_->seq = seq;
}

void WorkNCCL::setOwnedEphemeralTimeout(std::chrono::milliseconds timeout) {
  state_->owned_ephemeral_timeout = timeout;
}

void WorkNCCL::setHostBlocking(bool host_blocking) {
  state_->host_blocking = host_blocking;
}

WorkNCCL::WorkStatus WorkNCCL::checkStatus(
    std::optional<std::chrono::milliseconds> timeout) {
  return state_->checkStatus(timeout);
}

WorkNCCL::WorkStatus WorkNCCL::State::checkStatus(
    std::optional<std::chrono::milliseconds> timeout) {
  WorkStatus current = status();
  if (current == WorkStatus::COMPLETED || current == WorkStatus::ERROR ||
      current == WorkStatus::TIMEDOUT) {
    return current;
  }

  auto comm_error = comm->getError();
  if (comm_error == ErrorType::TIMEOUT) {
    setTerminalStatus(WorkStatus::TIMEDOUT);
    return status();
  }
  if (comm_error != ErrorType::SUCCESS) {
    setTerminalStatus(WorkStatus::ERROR);
    return status();
  }

  if (current == WorkStatus::NOT_STARTED) {
    try {
      if (events->start->query()) {
        WorkStatus expected = WorkStatus::NOT_STARTED;
        work_status.compare_exchange_strong(
            expected, WorkStatus::INPROGRESS, std::memory_order_relaxed);
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, comm) << "CUDA error during start event query: "
                          << e.what();
      setTerminalStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::ERROR) {
    return status();
  }

  if (status() == WorkStatus::INPROGRESS) {
    try {
      if (events->end->query()) {
        if (setTerminalStatus(WorkStatus::COMPLETED) &&
            owned_ephemeral_timeout.count() > 0 &&
            !ephemeral_timeout_released.exchange(true)) {
          comm->releaseEphemeralTimeout(owned_ephemeral_timeout);
        }
        return status();
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, comm) << "CUDA error during end event query: " << e.what();
      setTerminalStatus(WorkStatus::ERROR);
      return status();
    }
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - work_start_time);
  auto work_timeout = timeout.value_or(this->timeout);
  if (elapsed >= work_timeout) {
    TC_LOG(ERROR, comm) << "Operation timed out after " << elapsed.count()
                        << " ms";
    setTerminalStatus(WorkStatus::TIMEDOUT);
  }
  return status();
}

bool WorkNCCL::isCompleted() {
  WorkStatus current = checkStatus();
  return current == WorkStatus::COMPLETED || current == WorkStatus::ERROR ||
      current == WorkStatus::TIMEDOUT;
}

bool WorkNCCL::isSuccess() const {
  WorkStatus s = status();
  return s != WorkStatus::ERROR && s != WorkStatus::TIMEDOUT;
}

void WorkNCCL::synchronizeInternal() {
  WorkStatus local_state = status();
  if (local_state == WorkStatus::COMPLETED) {
    input_tensors_->clear();
    return;
  }
  if (local_state == WorkStatus::ERROR || local_state == WorkStatus::TIMEDOUT) {
    return;
  }

  TracingGuard tracingGuard(
      std::string(state_->comm->getCommName()),
      state_->comm->getSize(),
      "wait",
      state_->comm->getRank(),
      state_->seq);

  // Make the current stream wait for the end event recorded on the work's
  // stream, ordering subsequent current-stream ops after this collective.
  auto current_stream =
      at::cuda::getCurrentCUDAStream(state_->comm->getDevice().index());
  state_->events->end->block(current_stream);

  // For a synchronous barrier, mirror stock ProcessGroupNCCL by host-blocking
  // the CPU thread until prior current-stream work has completed, not just
  // stream-ordering it. Callers rely on this to flush async device work before
  // proceeding (e.g. the flashinfer trtllm one-shot Lamport all_reduce clears
  // its IPC buffers on the stream, then issues a synchronous barrier before the
  // first all_reduce; a stream-order-only barrier lets the all_reduce race the
  // clear and both ranks spin forever). Skip while the stream is capturing a
  // CUDA graph: cudaStreamSynchronize is illegal during capture and the
  // captured work is replayed on-device where a host sync is meaningless.
  if (state_->host_blocking && !state_->blocking_wait &&
      !c10::cuda::isStreamCapturingMayInitCtx(current_stream)) {
    C10_CUDA_CHECK(cudaStreamSynchronize(current_stream));
  }

  // Release tensor references. The CUDA caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations complete.
  input_tensors_->clear();
}

bool WorkNCCL::wait(std::chrono::milliseconds timeout) {
  synchronize();

  auto current_stream =
      at::cuda::getCurrentCUDAStream(state_->comm->getDevice().index());
  if (timeout == kNoTimeout &&
      c10::cuda::isStreamCapturingMayInitCtx(current_stream)) {
    WorkStatus current = status();
    if (current == WorkStatus::TIMEDOUT || current == WorkStatus::ERROR) {
      std::rethrow_exception(exception());
    }
    return true;
  }

  const auto wait_timeout =
      timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout);
  if (state_->blocking_wait || wait_timeout.has_value()) {
    while (true) {
      WorkStatus current = checkStatus(wait_timeout);
      if (current == WorkStatus::COMPLETED || current == WorkStatus::TIMEDOUT ||
          current == WorkStatus::ERROR) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  WorkStatus current = checkStatus(wait_timeout);
  if (state_->blocking_wait &&
      (current == WorkStatus::TIMEDOUT || current == WorkStatus::ERROR)) {
    state_->comm->handleBlockingWaitFailure(current, state_->reconfigure_uuid);
  }
  if (state_->blocking_wait) {
    // Blocking-wait mode has no watchdog to drain completed work.
    state_->comm->workq_.garbageCollect();
  }
  if (current == WorkStatus::TIMEDOUT || current == WorkStatus::ERROR) {
    std::rethrow_exception(exception());
  }
  return true;
}

void WorkNCCL::synchronize() {
  synchronizeInternal();
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<WorkNCCL>::unsafe_reclaim_from_nonowning(this));
  }
}

std::vector<at::Tensor> WorkNCCL::result() {
  return outputs_;
}

float WorkNCCL::getDuration() const {
  return state_->getDuration();
}

float WorkNCCL::State::getDuration() {
  TORCH_CHECK(
      timing_enabled,
      "getDuration only works if timing was enabled, see ProcessGroup::_enable_collectives_timing");
  TORCH_CHECK(events->start && events->end, "getDuration requires CUDA events");
  TORCH_CHECK(
      events->end->isCreated() && events->end->query(),
      "getDuration only works after the work has completed");
  std::shared_ptr<Events> start_events;
  {
    std::lock_guard<std::mutex> lock(duration_mutex);
    start_events = duration_start_events;
  }
  return start_events->start->elapsed_time(*events->end);
}

uint64_t WorkNCCL::getSequencenumber() const {
  return state_->seq;
}

uint64_t WorkNCCL::getCompletionKey() const {
  return state_->completion_key;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkNCCL::getFuture() {
  if (future_) {
    return future_;
  }

  std::vector<c10::Device> devices;
  for (const auto& tensor : outputs_) {
    if (tensor.device().type() != c10::DeviceType::CPU) {
      devices.push_back(tensor.device());
      break;
    }
  }
  future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);

  // Order the current stream after the collective before completing the future
  // so consumers observing the future see correct results.
  synchronize();

  if (!outputs_.empty() && !devices.empty()) {
    c10::OptionalDeviceGuard guard(outputs_[0].device());
    future_->markCompleted(c10::IValue(outputs_));
  } else {
    future_->markCompleted(c10::IValue(outputs_));
  }
  return future_;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkNCCL::getFutureResult() {
  checkStatus();
  return state_->future_work_result;
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
