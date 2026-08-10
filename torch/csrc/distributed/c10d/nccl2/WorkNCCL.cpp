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

WorkNCCL::WorkNCCL(
    ProcessGroupNCCL* comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : inputTensors_(inputTensors),
      comm_(comm),
      reconfigure_uuid_(comm->reconfigure_uuid_),
      blocking_wait_(comm->blocking_wait_),
      stream_(
          at::cuda::getStreamFromExternal(stream, comm->getDevice().index())),
      work_start_time_(std::chrono::steady_clock::now()),
      timeout_ms_(timeout_ms),
      timing_enabled_(comm->collectivesTimingEnabled()) {
  start_event_ = comm_->getEvent(timing_enabled_);
  end_event_ = comm_->getEvent(timing_enabled_);
  future_work_result_ =
      c10::make_intrusive<c10::ivalue::Future>(c10::AnyEnumType::get());
}

WorkNCCL::WorkNCCL(
    ProcessGroupNCCL* comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    at::Tensor inputTensor)
    : inputTensor_(std::move(inputTensor)),
      comm_(comm),
      reconfigure_uuid_(comm->reconfigure_uuid_),
      blocking_wait_(comm->blocking_wait_),
      stream_(
          at::cuda::getStreamFromExternal(stream, comm->getDevice().index())),
      work_start_time_(std::chrono::steady_clock::now()),
      timeout_ms_(timeout_ms),
      timing_enabled_(comm->collectivesTimingEnabled()) {
  start_event_ = comm_->getEvent(timing_enabled_);
  end_event_ = comm_->getEvent(timing_enabled_);
  future_work_result_ =
      c10::make_intrusive<c10::ivalue::Future>(c10::AnyEnumType::get());
}

WorkNCCL::~WorkNCCL() {
  if (!comm_) {
    return;
  }
  comm_->returnEvent(std::move(start_event_), timing_enabled_);
  comm_->returnEvent(std::move(end_event_), timing_enabled_);
}

void WorkNCCL::recordFunctionStart(std::string_view coll_name) {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }

  if (!inputTensors_.empty()) {
    std::vector<c10::IValue> inputs;
    inputs.reserve(inputTensors_.size());
    for (const auto& tensor : inputTensors_) {
      inputs.emplace_back(tensor);
    }
    recordFunction_->before(
        coll_name,
        c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
  } else if (inputTensor_.defined()) {
    recordFunction_->before(
        coll_name, c10::ArrayRef<const c10::IValue>(inputTensor_));
  } else {
    recordFunction_->before(coll_name, c10::ArrayRef<const c10::IValue>{});
  }
}

void WorkNCCL::recordStart(std::string_view coll_name) {
  recordFunctionStart(coll_name);
  start_event_->record(stream_);
}

void WorkNCCL::recordEnd() {
  end_event_->record(stream_);

  if (recordFunction_ && recordFunction_->isActive()) {
    recordFunction_->end();
  }
}

bool WorkNCCL::setTerminalStatus(WorkStatus terminal_status) {
  TORCH_INTERNAL_ASSERT(
      terminal_status == WorkStatus::COMPLETED ||
      terminal_status == WorkStatus::TIMEDOUT ||
      terminal_status == WorkStatus::ERROR);

  std::lock_guard<std::mutex> lock(terminal_status_mutex_);
  WorkStatus current = status();
  if (current == WorkStatus::COMPLETED || current == WorkStatus::TIMEDOUT ||
      current == WorkStatus::ERROR) {
    return false;
  }

  WorkResult result = WorkResult::SUCCESS;
  std::exception_ptr exception;
  if (terminal_status == WorkStatus::TIMEDOUT) {
    result = WorkResult::TIMEOUT;
    exception = std::make_exception_ptr(
        C10_BUILD_ERROR(DistBackendError, "NCCL operation timed out"));
  } else if (terminal_status == WorkStatus::ERROR) {
    result = WorkResult::COMM_ERROR;
    exception = std::make_exception_ptr(
        C10_BUILD_ERROR(DistBackendError, "NCCL operation failed"));
  }
  finish(std::move(exception));
  status_.store(terminal_status, std::memory_order_release);
  future_work_result_->markCompleted(c10::IValue(static_cast<uint8_t>(result)));
  return true;
}

WorkNCCL::WorkStatus WorkNCCL::checkStatus(
    std::optional<std::chrono::milliseconds> timeout) {
  WorkStatus current = status();
  if (current == WorkStatus::COMPLETED || current == WorkStatus::ERROR ||
      current == WorkStatus::TIMEDOUT) {
    return current;
  }

  auto comm_error = comm_->getError();
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
      if (start_event_->query()) {
        WorkStatus expected = WorkStatus::NOT_STARTED;
        status_.compare_exchange_strong(
            expected, WorkStatus::INPROGRESS, std::memory_order_relaxed);
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, comm_) << "CUDA error during start event query: "
                           << e.what();
      setTerminalStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::ERROR) {
    return status();
  }

  if (status() == WorkStatus::INPROGRESS) {
    try {
      if (end_event_->query()) {
        if (setTerminalStatus(WorkStatus::COMPLETED) &&
            owned_ephemeral_timeout_.count() > 0 &&
            !ephemeral_timeout_released_.exchange(true)) {
          comm_->releaseEphemeralTimeout(owned_ephemeral_timeout_);
        }
        return status();
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, comm_) << "CUDA error during end event query: " << e.what();
      setTerminalStatus(WorkStatus::ERROR);
      return status();
    }
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - work_start_time_);
  auto work_timeout = timeout.value_or(timeout_ms_);
  if (elapsed >= work_timeout) {
    TC_LOG(ERROR, comm_) << "Operation timed out after " << elapsed.count()
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
  if (local_state == WorkStatus::COMPLETED ||
      local_state == WorkStatus::ERROR || local_state == WorkStatus::TIMEDOUT) {
    return;
  }

  TracingGuard tracingGuard(
      std::string(comm_->getCommName()),
      comm_->getSize(),
      "wait",
      comm_->getRank(),
      seq_);

  // Make the current stream wait for the end event recorded on the work's
  // stream, ordering subsequent current-stream ops after this collective.
  auto current_stream =
      at::cuda::getCurrentCUDAStream(comm_->getDevice().index());
  end_event_->block(current_stream);

  // For a synchronous barrier, mirror stock ProcessGroupNCCL by host-blocking
  // the CPU thread until prior current-stream work has completed, not just
  // stream-ordering it. Callers rely on this to flush async device work before
  // proceeding (e.g. the flashinfer trtllm one-shot Lamport all_reduce clears
  // its IPC buffers on the stream, then issues a synchronous barrier before the
  // first all_reduce; a stream-order-only barrier lets the all_reduce race the
  // clear and both ranks spin forever). Skip while the stream is capturing a
  // CUDA graph: cudaStreamSynchronize is illegal during capture and the
  // captured work is replayed on-device where a host sync is meaningless.
  if (hostBlocking_ && !blocking_wait_ &&
      !c10::cuda::isStreamCapturingMayInitCtx(current_stream)) {
    C10_CUDA_CHECK(cudaStreamSynchronize(current_stream));
  }

  // Release tensor references. The CUDA caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations complete.
  inputTensors_.clear();
  inputTensor_.reset();
}

bool WorkNCCL::wait(std::chrono::milliseconds timeout) {
  synchronize();

  auto current_stream =
      at::cuda::getCurrentCUDAStream(comm_->getDevice().index());
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
  if (blocking_wait_ || wait_timeout.has_value()) {
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
  if (blocking_wait_ &&
      (current == WorkStatus::TIMEDOUT || current == WorkStatus::ERROR)) {
    comm_->handleBlockingWaitFailure(current, reconfigure_uuid_);
  }
  if (blocking_wait_) {
    // Blocking-wait mode has no watchdog to drain completed work.
    comm_->workq_.garbageCollect();
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
      timing_enabled_,
      "getDuration only works if timing was enabled, see ProcessGroup::_enable_collectives_timing");
  TORCH_CHECK(start_event_ && end_event_, "getDuration requires CUDA events");
  // A coalesced work owns the last op of the group and holds the earlier ops as
  // children, so span from the first op's start event to this work's end event.
  const auto& start_event =
      children_.empty() ? *start_event_ : *children_.front()->start_event_;
  TORCH_CHECK(
      end_event_->isCreated() && end_event_->query(),
      "getDuration only works after the work has completed");
  return start_event.elapsed_time(*end_event_);
}

uint64_t WorkNCCL::getSequencenumber() const {
  return seq_;
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
  return future_work_result_;
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
