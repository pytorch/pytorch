// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <queue>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <cuda_runtime.h>

#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d::nccl2 {

class ProcessGroupNCCL;

// Work object for the NCCL TorchComms backend. Ported from torchcomms'
// WorkNCCL, but rebased onto c10d::Work (upstream subclassed
// torchcomms::TorchWork). Completion is tracked with a pair of CUDA events;
// the Future/result handling that BackendWrapper::WorkWrapper used to provide
// is folded in here (see setOutputs/getFuture). The back-pointer to the owning
// backend is non-owning: the backend drains its work queue in finalize()/dtor,
// so a work never outlives its backend (upstream held a shared_ptr, which is
// incompatible with c10d's intrusive_ptr ownership of the backend).
class WorkNCCL : public c10d::Work {
 public:
  enum class WorkStatus {
    NOT_STARTED,
    INPROGRESS,
    COMPLETED,
    TIMEDOUT,
    ERROR,
  };

  WorkNCCL(
      ProcessGroupNCCL* comm,
      cudaStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);
  WorkNCCL(
      ProcessGroupNCCL* comm,
      cudaStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const at::Tensor& inputTensor);
  ~WorkNCCL() override;

  WorkNCCL(const WorkNCCL&) = delete;
  WorkNCCL(WorkNCCL&&) = delete;
  WorkNCCL& operator=(const WorkNCCL&) = delete;
  WorkNCCL& operator=(WorkNCCL&&) = delete;

  // c10d::Work overrides.
  bool isCompleted() override;
  bool isSuccess() const override;
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
  void synchronize() override;
  std::vector<at::Tensor> result() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

  std::chrono::milliseconds getTimeout() const {
    return timeout_ms_;
  }

  WorkStatus status() const {
    return status_.load(std::memory_order_relaxed);
  }

  // Output tensors for result()/getFuture(). Set by the backend after issuing.
  void setOutputs(std::vector<at::Tensor> outputs) {
    outputs_ = std::move(outputs);
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class ProcessGroupNCCL;
  friend class WorkNCCLQueue;

 private:
  void setStatus(WorkStatus status) {
    status_.store(status, std::memory_order_relaxed);
  }
  // Poll the CUDA events and advance status; used by the GC queue + watchdog.
  WorkStatus checkStatus();
  void recordFunctionStart(std::string_view coll_name);
  // Make the current stream wait on the work's end event (the c10d "wait"
  // semantics for CUDA work: order subsequent current-stream ops after this).
  void synchronizeInternal();

  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;
  std::vector<at::Tensor> outputs_;

  ProcessGroupNCCL* comm_; // non-owning; see class comment
  cudaEvent_t start_event_;
  cudaEvent_t end_event_;
  cudaStream_t stream_; // not owned by this class

  std::chrono::milliseconds timeout_ms_;

  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};
  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
  std::optional<at::RecordFunction> recordFunction_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

class WorkNCCLQueue {
 public:
  WorkNCCLQueue() = default;
  ~WorkNCCLQueue() = default;

  WorkNCCL::WorkStatus garbageCollect();
  // Finalize function can only be called from the main thread
  WorkNCCL::WorkStatus finalize();
  void enqueueWork(c10::intrusive_ptr<WorkNCCL> work, cudaStream_t stream);

 private:
  WorkNCCL::WorkStatus garbageCollectLocked();
  std::unordered_map<cudaStream_t, std::queue<c10::intrusive_ptr<WorkNCCL>>>
      stream_work_queues_;
  std::mutex work_queues_mutex_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
