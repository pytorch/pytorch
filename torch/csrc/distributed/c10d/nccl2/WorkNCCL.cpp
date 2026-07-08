// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/WorkNCCL.hpp>

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceGuard.h>

#include <torch/csrc/distributed/c10d/nccl2/CudaApi.hpp>
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
      stream_(stream),
      timeout_ms_(timeout_ms) {
  start_event_ = comm_->getEvent();
  end_event_ = comm_->getEvent();
}

WorkNCCL::WorkNCCL(
    ProcessGroupNCCL* comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const at::Tensor& inputTensor)
    : inputTensor_(inputTensor),
      comm_(comm),
      stream_(stream),
      timeout_ms_(timeout_ms) {
  start_event_ = comm_->getEvent();
  end_event_ = comm_->getEvent();
}

WorkNCCL::~WorkNCCL() {
  if (!comm_) {
    return;
  }
  comm_->returnEvent(start_event_);
  comm_->returnEvent(end_event_);
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

  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->eventRecord(start_event_, stream_),
      "Failed to record start event");
}

void WorkNCCL::recordEnd() {
  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->eventRecord(end_event_, stream_),
      "Failed to record end event");

  if (recordFunction_ && recordFunction_->isActive()) {
    recordFunction_->end();
  }
}

WorkNCCL::WorkStatus WorkNCCL::checkStatus() {
  if (status() == WorkStatus::COMPLETED || status() == WorkStatus::ERROR ||
      status() == WorkStatus::TIMEDOUT) {
    return status();
  }

  if (!start_completed_time_.has_value()) {
    cudaError_t start_status = comm_->getCudaApi()->eventQuery(start_event_);

    if (start_status == cudaSuccess) {
      start_completed_time_ = std::chrono::steady_clock::now();
      setStatus(WorkStatus::INPROGRESS);
    } else if (start_status != cudaErrorNotReady) {
      TC_LOG(ERROR, comm_) << "CUDA error during start event query: "
                           << comm_->getCudaApi()->getErrorString(start_status)
                           << " (" << start_status << ")";
      setStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::NOT_STARTED || status() == WorkStatus::ERROR) {
    return status();
  }

  cudaError_t end_status = comm_->getCudaApi()->eventQuery(end_event_);

  if (end_status == cudaSuccess) {
    setStatus(WorkStatus::COMPLETED);
    inputTensors_.clear();
    inputTensor_.reset();
  } else if (end_status == cudaErrorNotReady) {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_completed_time_.value());

    if (elapsed_milliseconds > timeout_ms_) {
      TC_LOG(ERROR, comm_) << "Operation timed out after "
                           << elapsed_milliseconds.count() << " ms";
      setStatus(WorkStatus::TIMEDOUT);
    }
  } else {
    TC_LOG(ERROR, comm_) << "CUDA error during end event query: "
                         << comm_->getCudaApi()->getErrorString(end_status)
                         << " (" << end_status << ")";
    setStatus(WorkStatus::ERROR);
  }
  return status();
}

bool WorkNCCL::isCompleted() {
  return checkStatus() == WorkStatus::COMPLETED;
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
      comm_->getRank());

  // Make the current stream wait for the end event recorded on the work's
  // stream, ordering subsequent current-stream ops after this collective.
  cudaStream_t current_stream =
      comm_->getCudaApi()->getCurrentCUDAStream(comm_->getDevice().index());
  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->streamWaitEvent(current_stream, end_event_, 0),
      "Failed to make stream wait for event");

  // Release tensor references. The CUDA caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations complete.
  inputTensors_.clear();
  inputTensor_.reset();
}

bool WorkNCCL::wait(std::chrono::milliseconds /*timeout*/) {
  // Unlike c10d's default wait(), this does not block the CPU: for CUDA work it
  // is sufficient (and matches upstream torchcomms) to order the current stream
  // after the collective. The timeout arg is honored by the watchdog, not here.
  synchronizeInternal();
  return true;
}

void WorkNCCL::synchronize() {
  synchronizeInternal();
}

std::vector<at::Tensor> WorkNCCL::result() {
  return outputs_;
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
  synchronizeInternal();

  if (!outputs_.empty() && !devices.empty()) {
    c10::OptionalDeviceGuard guard(outputs_[0].device());
    future_->markCompleted(c10::IValue(outputs_));
  } else {
    future_->markCompleted(c10::IValue(outputs_));
  }
  return future_;
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
