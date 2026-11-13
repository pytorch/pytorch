// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/WorkNCCL.hpp>

#include <ATen/core/ivalue.h>
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
std::atomic<uint64_t> nextCompletionKey{1};
} // namespace

NCCLEventPool::NCCLEventPool(
    bool cacheEnabled,
    bool timingEnabled,
    size_t maxSize)
    : event_cache_enabled_(cacheEnabled),
      max_event_pool_size_(maxSize),
      timing_enabled_(timingEnabled) {}

std::unique_ptr<at::cuda::CUDAEvent> NCCLEventPool::getEvent(
    bool timingEnabled) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);
  if (event_cache_enabled_ && timingEnabled == timing_enabled_.load() &&
      !event_pool_.empty()) {
    auto event = std::move(event_pool_.front());
    event_pool_.pop();
    return event;
  }
  return std::make_unique<at::cuda::CUDAEvent>(
      timingEnabled ? cudaEventDefault : cudaEventDisableTiming);
}

void NCCLEventPool::returnEvent(
    std::unique_ptr<at::cuda::CUDAEvent> event,
    bool timingEnabled) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);
  if (event_cache_enabled_ && timingEnabled == timing_enabled_.load() &&
      event_pool_.size() < max_event_pool_size_) {
    event_pool_.push(std::move(event));
  }
}

void NCCLEventPool::clear() {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);
  std::queue<std::unique_ptr<at::cuda::CUDAEvent>>().swap(event_pool_);
}

void NCCLEventPool::enableTiming() {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);
  if (timing_enabled_.exchange(true)) {
    return;
  }
  std::queue<std::unique_ptr<at::cuda::CUDAEvent>>().swap(event_pool_);
}

bool NCCLEventPool::timingEnabled() const {
  return timing_enabled_.load();
}

struct WorkNCCL::Events {
  Events(const std::shared_ptr<NCCLEventPool>& eventPool, bool timing_enabled)
      : eventPool(eventPool),
        timingEnabled(timing_enabled),
        start(eventPool->getEvent(timing_enabled)),
        end(eventPool->getEvent(timing_enabled)) {}

  Events(const Events&) = delete;
  Events(Events&&) = delete;
  Events& operator=(const Events&) = delete;
  Events& operator=(Events&&) = delete;

  ~Events() {
    if (auto pool = eventPool.lock()) {
      pool->returnEvent(std::move(start), timingEnabled);
      pool->returnEvent(std::move(end), timingEnabled);
    }
  }

  std::weak_ptr<NCCLEventPool> eventPool;
  bool timingEnabled;
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
      reconfigureUuid(comm->reconfigure_uuid_),
      blockingWait(comm->blocking_wait_),
      stream(
          at::cuda::getStreamFromExternal(stream, comm->getDevice().index())),
      workStartTime(std::chrono::steady_clock::now()),
      timeout(timeout),
      completionKey(nextCompletionKey.fetch_add(1, std::memory_order_relaxed)),
      timingEnabled(comm->collectivesTimingEnabled()),
      events(std::make_shared<Events>(comm->getEventPool(), timingEnabled)),
      durationStartEvents(events),
      futureWorkResult(
          c10::make_intrusive<c10::ivalue::Future>(c10::AnyEnumType::get())) {}

WorkNCCL::WorkNCCL(
    ProcessGroupNCCL* comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : state_(std::make_shared<State>(comm, stream, timeout_ms)),
      inputTensors_(std::make_shared<InputTensorShelf>(inputTensors)) {}

WorkNCCL::WorkNCCL(
    ProcessGroupNCCL* comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    at::Tensor inputTensor)
    : state_(std::make_shared<State>(comm, stream, timeout_ms)),
      inputTensors_(std::make_shared<InputTensorShelf>(
          std::vector<at::Tensor>{std::move(inputTensor)})) {}

WorkNCCL::~WorkNCCL() = default;

void WorkNCCL::recordFunctionStart(std::string_view coll_name) {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }

  std::lock_guard<std::mutex> lock(inputTensors_->mutex);
  if (!inputTensors_->tensors.empty()) {
    std::vector<c10::IValue> inputs;
    inputs.reserve(inputTensors_->tensors.size());
    for (const auto& tensor : inputTensors_->tensors) {
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
    std::lock_guard<std::mutex> lock(terminalStatusMutex);
    WorkStatus current = status();
    if (current == WorkStatus::COMPLETED || current == WorkStatus::TIMEDOUT ||
        current == WorkStatus::ERROR) {
      return false;
    }

    if (terminal_status == WorkStatus::TIMEDOUT) {
      result = WorkResult::TIMEOUT;
      workException = std::make_exception_ptr(C10_BUILD_ERROR(
          DistBackendError,
          "Watchdog caught collective operation timeout: NCCL operation "
          "timed out"));
    } else if (terminal_status == WorkStatus::ERROR) {
      result = WorkResult::COMM_ERROR;
      workException = std::make_exception_ptr(
          C10_BUILD_ERROR(DistBackendError, "NCCL operation failed"));
    }
    workStatus.store(terminal_status, std::memory_order_release);
  }
  futureWorkResult->markCompleted(c10::IValue(static_cast<uint8_t>(result)));
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
  if (timingEnabled) {
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
  comm->runCompletionHooks(completionKey, duration);
}

WorkNCCL::WorkStatus WorkNCCL::status() const {
  return state_->status();
}

std::exception_ptr WorkNCCL::exception() const {
  return state_->exception();
}

WorkNCCL::WorkStatus WorkNCCL::State::status() const {
  return workStatus.load(std::memory_order_acquire);
}

std::exception_ptr WorkNCCL::State::exception() const {
  std::lock_guard<std::mutex> lock(terminalStatusMutex);
  return workException;
}

std::chrono::milliseconds WorkNCCL::getTimeout() const {
  return state_->timeout;
}

void WorkNCCL::setChildren(std::vector<c10::intrusive_ptr<WorkNCCL>> children) {
  if (!children.empty()) {
    std::lock_guard<std::mutex> lock(state_->durationMutex);
    state_->durationStartEvents = children.front()->state_->events;
  }
  for (const auto& child : children) {
    inputTensors_->append(*child->inputTensors_);
  }
}

void WorkNCCL::setSequenceNumber(uint64_t seq) {
  state_->seq = seq;
}

void WorkNCCL::setOwnedEphemeralTimeout(std::chrono::milliseconds timeout) {
  state_->ownedEphemeralTimeout = timeout;
}

void WorkNCCL::setHostBlocking(bool host_blocking) {
  state_->hostBlocking = host_blocking;
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
        workStatus.compare_exchange_strong(
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
            ownedEphemeralTimeout.count() > 0 &&
            !ephemeralTimeoutReleased.exchange(true)) {
          comm->releaseEphemeralTimeout(ownedEphemeralTimeout);
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
      std::chrono::steady_clock::now() - workStartTime);
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
    inputTensors_->clear();
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
  if (state_->hostBlocking && !state_->blockingWait &&
      !c10::cuda::isStreamCapturingMayInitCtx(current_stream)) {
    C10_CUDA_CHECK(cudaStreamSynchronize(current_stream));
  }

  // Release tensor references. The CUDA caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations complete.
  inputTensors_->clear();
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
  if (state_->blockingWait || wait_timeout.has_value()) {
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
  if (state_->blockingWait &&
      (current == WorkStatus::TIMEDOUT || current == WorkStatus::ERROR)) {
    state_->comm->handleBlockingWaitFailure(current, state_->reconfigureUuid);
  }
  if (state_->blockingWait) {
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
      timingEnabled,
      "getDuration only works if timing was enabled, see ProcessGroup::_enable_collectives_timing");
  TORCH_CHECK(events->start && events->end, "getDuration requires CUDA events");
  TORCH_CHECK(
      events->end->isCreated() && events->end->query(),
      "getDuration only works after the work has completed");
  std::shared_ptr<Events> start_events;
  {
    std::lock_guard<std::mutex> lock(durationMutex);
    start_events = durationStartEvents;
  }
  return start_events->start->elapsed_time(*events->end);
}

uint64_t WorkNCCL::getSequencenumber() const {
  return state_->seq;
}

uint64_t WorkNCCL::getCompletionKey() const {
  return state_->completionKey;
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
  return state_->futureWorkResult;
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
