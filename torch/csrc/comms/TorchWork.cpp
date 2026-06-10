// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/TorchWork.hpp>

#include <c10/core/DeviceGuard.h> // @manual=//caffe2:c10

namespace torch::comms {

void TorchWork::markCompleted(
    c10::intrusive_ptr<c10::ivalue::Future> future_,
    std::vector<at::Tensor> outputTensors_) {
  TORCH_CHECK(
      outputTensors_.size() > 0, "At least one tensor should be present");
  // CUDA: resolve immediately. Future records a CUDA event on the current
  // stream via markCompleted(). Device guard ensures getCurrentStream()
  // returns the correct device's stream.
  const auto device = outputTensors_[0].device();
  c10::OptionalDeviceGuard guard(device);
  future_->markCompleted(c10::IValue(outputTensors_));
}

TorchWorkCompleted::TorchWorkCompleted() {
  setStatus(WorkStatus::COMPLETED);
}

void TorchWorkCompleted::wait() {
  runWaitPreHooks();
  runWaitPostHooks();
}

void TorchWorkCompleted::waitBlocking() {
  // Note: required for TorchComm::reconfigure to work with the dummy backend.
  // No need to checkStatus as the constructor sets the status to COMPLETED.
  return;
}

TorchWorkThread::TorchWorkThread(std::function<void()> fn)
    : future_(std::async(std::launch::async, [this, fn = std::move(fn)]() {
        try {
          fn();
          setStatus(WorkStatus::COMPLETED);
        } catch (...) {
          setStatus(WorkStatus::ERROR);
          throw;
        }
      })) {}

void TorchWorkThread::wait() {
  runWaitPreHooks();

  if (!future_.valid()) {
    // already waited on
    runWaitPostHooks();
    return;
  }
  future_.get();
  runWaitPostHooks();
}

} // namespace torch::comms
