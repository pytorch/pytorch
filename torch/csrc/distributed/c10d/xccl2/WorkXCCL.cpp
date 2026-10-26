#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/xccl2/WorkXCCL.hpp>

#include <ATen/core/ivalue.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>

#include <thread>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/xccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/xccl2/ProcessGroupXCCL.hpp>
#include <torch/csrc/distributed/c10d/xccl2/TracingGuard.hpp>

namespace c10d::xccl2 {

std::unique_ptr<at::xpu::XPUEvent> XcclEventPool::get(bool timing_enabled) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (timing_enabled == timing_enabled_.load() && !pool_.empty()) {
    std::unique_ptr<at::xpu::XPUEvent> event = std::move(pool_.front());
    pool_.pop();
    return event;
  }

  // Create new event if pool is empty
  return std::make_unique<at::xpu::XPUEvent>(timing_enabled);
}

void XcclEventPool::put(
    std::unique_ptr<at::xpu::XPUEvent> event,
    bool timing_enabled) {
  if (!event) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);

  if (timing_enabled == timing_enabled_.load() && pool_.size() < max_size_) {
    pool_.push(std::move(event));
  }
  // Otherwise the event is destroyed with the unique_ptr.
}

void XcclEventPool::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::queue<std::unique_ptr<at::xpu::XPUEvent>>().swap(pool_);
}

void XcclEventPool::enableTiming() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (timing_enabled_.exchange(true)) {
    return;
  }
  // Pooled events were created without timing and can no longer be handed out.
  std::queue<std::unique_ptr<at::xpu::XPUEvent>>().swap(pool_);
}

bool XcclEventPool::timingEnabled() const {
  return timing_enabled_.load();
}

namespace {

XcclCommInfo makeCommInfo(ProcessGroupXCCL* comm) {
  return XcclCommInfo{
      std::string(comm->getCommName()),
      comm->getRank(),
      comm->getSize(),
      comm->getDevice()};
}

} // namespace

WorkXCCL::WorkXCCL(
    ProcessGroupXCCL* comm,
    c10::xpu::XPUStream stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : inputTensors_(inputTensors),
      comm_info_(makeCommInfo(comm)),
      event_pool_(comm->getEventPool()),
      stream_(stream),
      timeout_ms_(timeout_ms),
      create_time_(std::chrono::steady_clock::now()) {
  const auto& pool = comm->getEventPool();
  timing_enabled_ = pool->timingEnabled();
  start_event_ = pool->get(timing_enabled_);
  end_event_ = pool->get(timing_enabled_);
  // Events are recorded around the actual XCCL operations by the backend.
}

WorkXCCL::WorkXCCL(
    ProcessGroupXCCL* comm,
    c10::xpu::XPUStream stream,
    std::chrono::milliseconds timeout_ms,
    at::Tensor inputTensor)
    : inputTensor_(std::move(inputTensor)),
      comm_info_(makeCommInfo(comm)),
      event_pool_(comm->getEventPool()),
      stream_(stream),
      timeout_ms_(timeout_ms),
      create_time_(std::chrono::steady_clock::now()) {
  const auto& pool = comm->getEventPool();
  timing_enabled_ = pool->timingEnabled();
  start_event_ = pool->get(timing_enabled_);
  end_event_ = pool->get(timing_enabled_);
}

WorkXCCL::~WorkXCCL() {
  // If the backend and its pool are gone, the events are simply destroyed.
  if (auto pool = event_pool_.lock()) {
    pool->put(std::move(start_event_), timing_enabled_);
    pool->put(std::move(end_event_), timing_enabled_);
  }
}

void WorkXCCL::recordFunctionStart(std::string_view coll_name) {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }

  // Passing the input tensors to recordFunction allows for shape information in
  // the profiling output.
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

void WorkXCCL::recordStart(std::string_view coll_name) {
  recordFunctionStart(coll_name);
  start_event_->record(stream_);
}

void WorkXCCL::recordEnd() {
  end_event_->record(stream_);

  if (recordFunction_ && recordFunction_->isActive()) {
    recordFunction_->end();
  }
}

WorkXCCL::WorkStatus WorkXCCL::checkStatus(
    std::optional<std::chrono::milliseconds> timeout) {
  // The watchdog and a user calling isCompleted() can poll concurrently.
  // Serialize them so start_completed_time_ has a single writer and the first
  // terminal status is never overwritten by a racing caller.
  std::lock_guard<std::mutex> lock(check_status_mutex_);
  if (status() == WorkStatus::COMPLETED || status() == WorkStatus::ERROR ||
      status() == WorkStatus::TIMEDOUT) {
    return status();
  }

  // Step 1: if the start event has not been observed to complete yet, query it
  // and remember when it did. The work's own timeout is measured from that
  // point, not from when the work was created, so time spent queued behind
  // other collectives is not charged against it.
  if (!start_completed_time_.has_value()) {
    try {
      if (start_event_->query()) {
        start_completed_time_ = std::chrono::steady_clock::now();
        setStatus(WorkStatus::INPROGRESS);
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, &comm_info_)
          << "XPU error during start event query: " << e.what();
      setStatus(WorkStatus::ERROR);
      return status();
    }
  }

  // Step 2: once the collective has started, poll the end event.
  if (status() == WorkStatus::INPROGRESS) {
    try {
      if (end_event_->query()) {
        setStatus(WorkStatus::COMPLETED);
        return status();
      }
    } catch (const std::exception& e) {
      TC_LOG(ERROR, &comm_info_)
          << "XPU error during end event query: " << e.what();
      setStatus(WorkStatus::ERROR);
      return status();
    }
  }

  // A caller-supplied timeout (wait(timeout)) is measured from creation, as in
  // nccl2, so it still bounds a work whose start event never fires.
  auto now = std::chrono::steady_clock::now();
  std::optional<std::chrono::steady_clock::time_point> since;
  if (timeout.has_value()) {
    since = create_time_;
  } else if (start_completed_time_.has_value()) {
    since = start_completed_time_;
  }
  if (since.has_value()) {
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - *since);
    if (elapsed > timeout.value_or(timeout_ms_)) {
      TC_LOG(ERROR, &comm_info_)
          << "Operation timed out after " << elapsed.count() << " ms";
      setStatus(WorkStatus::TIMEDOUT);
    }
  }
  return status();
}

void WorkXCCL::setStatus(WorkStatus status) {
  if (status == WorkStatus::TIMEDOUT || status == WorkStatus::ERROR) {
    std::lock_guard<std::mutex> lock(mutex_);
    exception_ = std::make_exception_ptr(
        status == WorkStatus::TIMEDOUT
            ? C10_BUILD_ERROR(
                  DistBackendError,
                  "Watchdog caught collective operation timeout: XCCL "
                  "operation timed out")
            : C10_BUILD_ERROR(DistBackendError, "XCCL operation failed"));
  }
  status_.store(status, std::memory_order_release);
}

bool WorkXCCL::isCompleted() {
  // c10d semantics: a work that failed or timed out is also completed.
  WorkStatus current = checkStatus();
  return current == WorkStatus::COMPLETED || current == WorkStatus::ERROR ||
      current == WorkStatus::TIMEDOUT;
}

bool WorkXCCL::isSuccess() const {
  WorkStatus s = status();
  return s != WorkStatus::ERROR && s != WorkStatus::TIMEDOUT;
}

void WorkXCCL::synchronizeInternal() {
  WorkStatus local_state = status();
  if (local_state == WorkStatus::COMPLETED) {
    // checkStatus() no longer releases the inputs (it runs on the watchdog), so
    // a work the watchdog already retired must drop them here.
    inputTensors_.clear();
    inputTensor_.reset();
    return;
  }
  if (local_state == WorkStatus::ERROR || local_state == WorkStatus::TIMEDOUT) {
    return;
  }

  TracingGuard tracingGuard(
      comm_info_.name, comm_info_.size, "wait", comm_info_.rank, seq_);

  // Make the current stream wait for the end event recorded on the work's
  // stream, ordering subsequent current-stream ops after this collective.
  auto current_stream =
      c10::xpu::getCurrentXPUStream(comm_info_.device.index());
  end_event_->block(current_stream);

  // Release tensor references. The XPU caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations complete.
  inputTensors_.clear();
  inputTensor_.reset();
}

bool WorkXCCL::wait(std::chrono::milliseconds timeout) {
  synchronize();

  // Without a timeout this only orders the current stream after the
  // collective, and reports a failure the watchdog has already observed.
  if (timeout != kNoTimeout) {
    while (true) {
      WorkStatus current = checkStatus(timeout);
      if (current == WorkStatus::COMPLETED || current == WorkStatus::TIMEDOUT ||
          current == WorkStatus::ERROR) {
        break;
      }
      // XPUEvent has no timed wait, so poll like nccl2 does.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  WorkStatus current = status();
  if (current == WorkStatus::TIMEDOUT || current == WorkStatus::ERROR) {
    std::rethrow_exception(exception());
  }
  return true;
}

void WorkXCCL::synchronize() {
  synchronizeInternal();
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<WorkXCCL>::unsafe_reclaim_from_nonowning(this));
  }
}

std::vector<at::Tensor> WorkXCCL::result() {
  return outputs_;
}

float WorkXCCL::getDuration() const {
  TORCH_CHECK(
      timing_enabled_,
      "getDuration only works if timing was enabled, see ProcessGroup::_enable_collectives_timing");
  // XPUEvent::elapsed_time() blocks until both events complete, so require
  // completion up front rather than stalling the caller.
  TORCH_CHECK(
      end_event_->isCreated() && end_event_->query(),
      "getDuration only works after the work has completed");
  return static_cast<float>(start_event_->elapsed_time(*end_event_));
}

c10::intrusive_ptr<c10::ivalue::Future> WorkXCCL::getFuture() {
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

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
