// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDAStream.h>
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
      at::Tensor inputTensor);
  ~WorkNCCL() override;

  WorkNCCL(const WorkNCCL&) = delete;
  WorkNCCL(WorkNCCL&&) = delete;
  WorkNCCL& operator=(const WorkNCCL&) = delete;
  WorkNCCL& operator=(WorkNCCL&&) = delete;

  // c10d::Work overrides.
  bool isCompleted() override;
  bool isSuccess() const override;
  std::exception_ptr exception() const override;
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
  void synchronize() override;
  std::vector<at::Tensor> result() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override;
  float getDuration() const override;
  uint64_t getSequencenumber() const override;
  uint64_t getCompletionKey() const override;

  std::chrono::milliseconds getTimeout() const override;
  WorkStatus status() const;

  // Output tensors for result()/getFuture(). Set by the backend after issuing.
  void setOutputs(std::vector<at::Tensor> outputs) {
    outputs_ = std::move(outputs);
  }
  void setChildren(std::vector<c10::intrusive_ptr<WorkNCCL>> children);
  // Per-process-group collective counter of the op this work tracks; set by
  // the backend's createWork().
  void setSequenceNumber(uint64_t seq);
  void setOwnedEphemeralTimeout(std::chrono::milliseconds timeout);
  void setHostBlocking(bool host_blocking);

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class ProcessGroupNCCL;
  friend class WorkNCCLQueue;
  friend class WindowNCCL;

 private:
  struct Events;
  struct InputTensorShelf {
    explicit InputTensorShelf(std::vector<at::Tensor> tensors);
    void append(InputTensorShelf& other);
    void clear();

    std::mutex mutex;
    std::vector<at::Tensor> tensors;
  };
  struct State {
    State(
        ProcessGroupNCCL* comm,
        cudaStream_t stream,
        std::chrono::milliseconds timeout);

    WorkStatus status() const;
    std::exception_ptr exception() const;
    bool setTerminalStatus(WorkStatus status);
    WorkStatus checkStatus(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);
    void notifyCompletion();
    float getDuration();

    ProcessGroupNCCL* comm;
    int64_t reconfigureUuid;
    bool blockingWait;
    at::cuda::CUDAStream stream;
    std::chrono::steady_clock::time_point workStartTime;
    std::chrono::milliseconds timeout;
    uint64_t completionKey;
    std::chrono::milliseconds ownedEphemeralTimeout{0};
    std::atomic<bool> ephemeralTimeoutReleased{false};
    bool timingEnabled;
    uint64_t seq{0};
    std::shared_ptr<Events> events;
    std::mutex durationMutex;
    std::shared_ptr<Events> durationStartEvents;
    mutable std::mutex terminalStatusMutex;
    std::atomic<WorkStatus> workStatus{WorkStatus::NOT_STARTED};
    std::exception_ptr workException;
    c10::intrusive_ptr<c10::ivalue::Future> futureWorkResult;
    bool hostBlocking{false};
  };
  struct TrackedWork {
    std::shared_ptr<State> state;
    std::shared_ptr<InputTensorShelf> inputTensors;
  };

  // Poll the CUDA events and advance status; used by the GC queue + watchdog.
  WorkStatus checkStatus(
      std::optional<std::chrono::milliseconds> timeout = std::nullopt);
  void recordFunctionStart(std::string_view coll_name);
  // Make the current stream wait on the work's end event (the c10d "wait"
  // semantics for CUDA work: order subsequent current-stream ops after this).
  void synchronizeInternal();

  std::shared_ptr<State> state_;
  std::shared_ptr<InputTensorShelf> inputTensors_;
  std::vector<at::Tensor> outputs_;
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
  void enqueueWork(
      const c10::intrusive_ptr<WorkNCCL>& work,
      cudaStream_t stream);

 private:
  // completed collects the states retired as COMPLETED, so the caller can push
  // their completion out after dropping work_queues_mutex_.
  WorkNCCL::WorkStatus garbageCollectLocked(
      std::vector<std::shared_ptr<WorkNCCL::State>>& completed);
  std::unordered_map<cudaStream_t, std::queue<WorkNCCL::TrackedWork>>
      stream_work_queues_;
  std::queue<std::shared_ptr<WorkNCCL::InputTensorShelf>>
      completedInputTensors_;
  std::mutex work_queues_mutex_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
