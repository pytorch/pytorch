// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/comms/nccl/TorchCommNCCL.hpp>
#include <torch/csrc/comms/nccl/TorchWorkNCCL.hpp>
#include <torch/csrc/comms/utils/Logging.hpp>
#include <torch/csrc/comms/utils/TracingGuard.hpp>

namespace torch::comms {

TorchWorkNCCL::TorchWorkNCCL(
    std::shared_ptr<TorchCommNCCL> comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : inputTensors_(inputTensors),
      comm_(std::move(comm)),
      stream_(stream),
      timeout_ms_(timeout_ms) {
  // If not in graph capture mode, create the events for start and end
  // recording
  start_event_ = comm_->getEvent();
  end_event_ = comm_->getEvent();

  // Events will be recorded around the actual NCCL operations
}

TorchWorkNCCL::TorchWorkNCCL(
    std::shared_ptr<TorchCommNCCL> comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const at::Tensor& inputTensor)
    : inputTensor_(inputTensor),
      comm_(std::move(comm)),
      stream_(stream),
      timeout_ms_(timeout_ms) {
  // If not in graph capture mode, create the events for start and end
  // recording
  start_event_ = comm_->getEvent();
  end_event_ = comm_->getEvent();

  // Events will be recorded around the actual NCCL operations
}

TorchWorkNCCL::~TorchWorkNCCL() {
  if (!comm_) {
    return;
  }
  // If not in graph capture mode, return the events to the pool
  comm_->returnEvent(start_event_);
  comm_->returnEvent(end_event_);
}

void TorchWorkNCCL::recordFunctionStart(std::string_view coll_name) {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }

  // Passing input tensor to recordFunction allows for shape information in
  // profiling output.
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

void TorchWorkNCCL::recordStart(std::string_view coll_name) {
  recordFunctionStart(coll_name);

  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->eventRecord(start_event_, stream_),
      "Failed to record start event");
}

void TorchWorkNCCL::recordEnd() {
  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->eventRecord(end_event_, stream_),
      "Failed to record end event");

  if (recordFunction_ && recordFunction_->isActive()) {
    recordFunction_->end();
  }
}

TorchWorkNCCL::WorkStatus TorchWorkNCCL::checkStatus() {
  // If already marked as completed, return COMPLETED
  if (status() == WorkStatus::COMPLETED || status() == WorkStatus::ERROR ||
      status() == WorkStatus::TIMEDOUT) {
    return status();
  }

  // Step 1: If start_completed_time_ doesn't have a value yet, query the start
  // event
  if (!start_completed_time_.has_value()) {
    cudaError_t start_status = comm_->getCudaApi()->eventQuery(start_event_);

    if (start_status == cudaSuccess) {
      // Start event has completed, store the current time
      start_completed_time_ = std::chrono::steady_clock::now();
      setStatus(WorkStatus::INPROGRESS);
    } else if (start_status != cudaErrorNotReady) {
      // Some other error occurred with the start event
      TC_LOG(ERROR, comm_.get())
          << "CUDA error during start event query: "
          << comm_->getCudaApi()->getErrorString(start_status) << " ("
          << start_status << ")";
      setStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::NOT_STARTED || status() == WorkStatus::ERROR) {
    return status();
  }

  // Step 2: If we get here, start event has completed, so query the end event
  cudaError_t end_status = comm_->getCudaApi()->eventQuery(end_event_);

  if (end_status == cudaSuccess) {
    // End event has completed, mark the work as completed
    setStatus(WorkStatus::COMPLETED);

    // Release the input tensors to keep the lifetime of the tensors short
    inputTensors_.clear();
    inputTensor_.reset();
  } else if (end_status == cudaErrorNotReady) {
    // End event has not completed yet, check for timeout
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_completed_time_.value());

    // Check if the operation has timed out
    if (elapsed_milliseconds > timeout_ms_) {
      TC_LOG(ERROR, comm_.get()) << "Operation timed out after "
                                 << elapsed_milliseconds.count() << " ms";
      setStatus(WorkStatus::TIMEDOUT);
    }
  } else {
    // Some other error occurred with the end event
    TC_LOG(ERROR, comm_.get())
        << "CUDA error during end event query: "
        << comm_->getCudaApi()->getErrorString(end_status) << " (" << end_status
        << ")";
    setStatus(WorkStatus::ERROR);
  }
  return status();
}

void TorchWorkNCCL::wait() {
  runWaitPreHooks();

  // If already completed, return immediately
  WorkStatus local_state = status();
  if (local_state == WorkStatus::COMPLETED ||
      local_state == WorkStatus::ERROR || local_state == WorkStatus::TIMEDOUT) {
    runWaitPostHooks();
    return;
  }

  TracingGuard tracingGuard(
      std::string(comm_->getCommName()),
      comm_->getSize(),
      "wait",
      comm_->getRank());

  // Get the current stream using the device from the comm object
  cudaStream_t current_stream =
      comm_->getCudaApi()->getCurrentCUDAStream(comm_->device_.index());

  // Add a dependency from the work's stream to the current stream
  // This makes the current stream wait for the end_event_ recorded on the
  // work's stream
  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->streamWaitEvent(current_stream, end_event_, 0),
      "Failed to make stream wait for event");

  // Release tensor references. The CUDA caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations
  // complete.
  inputTensors_.clear();
  inputTensor_.reset();

  runWaitPostHooks();
}
} // namespace torch::comms
