// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/WorkNCCL.hpp>

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <thread>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/nccl2/TracingGuard.hpp>

namespace c10d::nccl2 {

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

std::vector<at::Tensor> WorkNCCL::InputTensorShelf::copy() const {
  std::lock_guard<std::mutex> lock(mutex);
  return tensors;
}

void WorkNCCL::InputTensorShelf::clear() {
  std::lock_guard<std::mutex> lock(mutex);
  tensors.clear();
}

struct WorkNCCL::State {
  State(
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
        timing_enabled(comm->collectivesTimingEnabled()),
        events(std::make_shared<Events>(comm, timing_enabled)),
        duration_start_events(events),
        future_work_result(
            c10::make_intrusive<c10::ivalue::Future>(c10::AnyEnumType::get())) {
  }

  ProcessGroupNCCL* comm;
  int64_t reconfigure_uuid;
  bool blocking_wait;
  at::cuda::CUDAStream stream;
  std::chrono::steady_clock::time_point work_start_time;
  std::chrono::milliseconds timeout;
  std::chrono::milliseconds owned_ephemeral_timeout{0};
  std::atomic<bool> ephemeral_timeout_released{false};
  bool timing_enabled;
  uint64_t seq{0};
  std::shared_ptr<Events> events;
  std::shared_ptr<Events> duration_start_events;
  std::mutex terminal_status_mutex;
  std::atomic<WorkStatus> status{WorkStatus::NOT_STARTED};
  std::exception_ptr exception;
  c10::intrusive_ptr<c10::ivalue::Future> future_work_result;
  bool host_blocking{false};
};

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

WorkNCCL::WorkNCCL(
    TrackingTag tracking_tag,
    WorkNCCL& work,
    bool retain_input_tensors)
    : state_(work.state_),
      input_tensors_(retain_input_tensors ? work.input_tensors_ : nullptr) {}

WorkNCCL::~WorkNCCL() = default;

c10::intrusive_ptr<WorkNCCL> WorkNCCL::createTrackingWork(
    bool retain_input_tensors) {
  return c10::make_intrusive<WorkNCCL>(
      TrackingTag{}, *this, retain_input_tensors);
}

std::shared_ptr<WorkNCCL::InputTensorShelf> WorkNCCL::inputTensors() const {
  return input_tensors_;
}

void WorkNCCL::recordFunctionStart(std::string_view coll_name) {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }

  auto input_tensors = input_tensors_->copy();
  if (!input_tensors.empty()) {
    std::vector<c10::IValue> inputs;
    inputs.reserve(input_tensors.size());
    for (const auto& tensor : input_tensors) {
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

bool WorkNCCL::setTerminalStatus(WorkStatus terminal_status) {
  TORCH_INTERNAL_ASSERT(
      terminal_status == WorkStatus::COMPLETED ||
      terminal_status == WorkStatus::TIMEDOUT ||
      terminal_status == WorkStatus::ERROR);

  WorkResult result = WorkResult::SUCCESS;
  {
    std::lock_guard<std::mutex> lock(state_->terminal_status_mutex);
    WorkStatus current = status();
    if (current == WorkStatus::COMPLETED || current == WorkStatus::TIMEDOUT ||
        current == WorkStatus::ERROR) {
      return false;
    }

    if (terminal_status == WorkStatus::TIMEDOUT) {
      result = WorkResult::TIMEOUT;
      state_->exception = std::make_exception_ptr(
          C10_BUILD_ERROR(DistBackendError, "NCCL operation timed out"));
    } else if (terminal_status == WorkStatus::ERROR) {
      result = WorkResult::COMM_ERROR;
      state_->exception = std::make_exception_ptr(
          C10_BUILD_ERROR(DistBackendError, "NCCL operation failed"));
    }
    state_->status.store(terminal_status, std::memory_order_release);
  }
  state_->future_work_result->markCompleted(
      c10::IValue(static_cast<uint8_t>(result)));
  return true;
}

void WorkNCCL::notifyCompletion() {
  // Called once per work, by the queue that popped it as COMPLETED, and only
  // for success -- a timed-out or failed work reports nothing, because a
  // consumer that read that as "finished" would lose the very fact a
  // post-mortem needs.
  if (!state_->comm->hasCompletionHooks()) {
    return;
  }
  std::optional<float> duration;
  if (state_->timing_enabled) {
    try {
      // cudaEventElapsedTime on two events that have already been observed to
      // complete: no synchronization, and no NCCL call, so it is legal on the
      // watchdog thread. Stock ProcessGroupNCCL's watchdog calls getDuration()
      // from its own completion hook for the same reason.
      duration = getDuration();
    } catch (const std::exception& e) {
      TC_LOG(WARNING, state_->comm)
          << "Cannot measure collective duration: " << e.what();
    }
  }
  state_->comm->runCompletionHooks(this, duration);
}

WorkNCCL::WorkStatus WorkNCCL::status() const {
  return state_->status.load(std::memory_order_acquire);
}

std::exception_ptr WorkNCCL::exception() const {
  std::lock_guard<std::mutex> lock(state_->terminal_status_mutex);
  return state_->exception;
}

std::chrono::milliseconds WorkNCCL::getTimeout() const {
  return state_->timeout;
}

void WorkNCCL::setChildren(std::vector<c10::intrusive_ptr<WorkNCCL>> children) {
  if (!children.empty()) {
    state_->duration_start_events = children.front()->state_->events;
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
  WorkStatus current = status();
  if (current == WorkStatus::COMPLETED || current == WorkStatus::ERROR ||
      current == WorkStatus::TIMEDOUT) {
    return current;
  }

  auto comm_error = state_->comm->getError();
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
      if (state_->events->start->query()) {
        WorkStatus expected = WorkStatus::NOT_STARTED;
        state_->status.compare_exchange_strong(
            expected, WorkStatus::INPROGRESS, std::memory_order_relaxed);
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, state_->comm)
          << "CUDA error during start event query: " << e.what();
      setTerminalStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::ERROR) {
    return status();
  }

  if (status() == WorkStatus::INPROGRESS) {
    try {
      if (state_->events->end->query()) {
        if (setTerminalStatus(WorkStatus::COMPLETED) &&
            state_->owned_ephemeral_timeout.count() > 0 &&
            !state_->ephemeral_timeout_released.exchange(true)) {
          state_->comm->releaseEphemeralTimeout(
              state_->owned_ephemeral_timeout);
        }
        return status();
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, state_->comm)
          << "CUDA error during end event query: " << e.what();
      setTerminalStatus(WorkStatus::ERROR);
      return status();
    }
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - state_->work_start_time);
  auto work_timeout = timeout.value_or(state_->timeout);
  if (elapsed >= work_timeout) {
    TC_LOG(ERROR, state_->comm)
        << "Operation timed out after " << elapsed.count() << " ms";
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
  TORCH_CHECK(
      state_->timing_enabled,
      "getDuration only works if timing was enabled, see ProcessGroup::_enable_collectives_timing");
  TORCH_CHECK(
      state_->events->start && state_->events->end,
      "getDuration requires CUDA events");
  TORCH_CHECK(
      state_->events->end->isCreated() && state_->events->end->query(),
      "getDuration only works after the work has completed");
  return state_->duration_start_events->start->elapsed_time(
      *state_->events->end);
}

uint64_t WorkNCCL::getSequencenumber() const {
  return state_->seq;
}

const void* WorkNCCL::getCompletionKey() const {
  return state_.get();
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
