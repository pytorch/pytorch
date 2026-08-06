// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_NCCL

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d::nccl2 {

class ProcessGroupNCCL;

enum class CommState {
  NORMAL,
  ERROR,
  TIMEOUT,
};

class WorkNCCLState {
 public:
  WorkNCCLState(
      size_t max_event_pool_size,
      bool timing_enabled,
      bool event_cache_enabled)
      : timing_enabled(timing_enabled),
        max_event_pool_size_(max_event_pool_size),
        event_cache_enabled_(event_cache_enabled) {}

  [[nodiscard]] std::unique_ptr<at::cuda::CUDAEvent> getEvent(
      bool timing_enabled);
  void returnEvent(
      std::unique_ptr<at::cuda::CUDAEvent> event,
      bool timing_enabled);
  void enableCollectivesTiming();
  void closeEventPool();
  std::pair<std::chrono::milliseconds, std::chrono::milliseconds>
  applyEphemeralTimeout(std::chrono::milliseconds timeout);
  void releaseEphemeralTimeout(std::chrono::milliseconds timeout);
  void addEphemeralTimeout(std::chrono::milliseconds timeout);

  std::atomic<CommState> comm_state{CommState::NORMAL};
  std::atomic<bool> timing_enabled{false};

 private:
  const size_t max_event_pool_size_;
  const bool event_cache_enabled_;
  std::queue<std::unique_ptr<at::cuda::CUDAEvent>> event_pool_;
  std::mutex event_pool_mutex_;
  bool event_pool_open_{true};
  std::mutex ephemeral_timeout_mutex_;
  std::chrono::milliseconds ephemeral_timeout_active_{0};
  std::chrono::milliseconds ephemeral_timeout_inflight_{0};
};

// Work object for the NCCL TorchComms backend. Ported from torchcomms'
// WorkNCCL, but rebased onto c10d::Work (upstream subclassed
// torchcomms::TorchWork). Completion is tracked with a pair of CUDA events;
// the Future/result handling that BackendWrapper::WorkWrapper used to provide
// is folded in here (see setOutputs/getFuture). State needed after
// process-group teardown is shared separately so Python may retain a Work
// without retaining the backend or creating a backend-to-work reference cycle.
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
      std::shared_ptr<WorkNCCLState> state,
      at::Device device,
      int rank,
      int size,
      std::string comm_name,
      cudaStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);
  WorkNCCL(
      std::shared_ptr<WorkNCCLState> state,
      at::Device device,
      int rank,
      int size,
      std::string comm_name,
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
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
  void synchronize() override;
  std::vector<at::Tensor> result() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override;
  float getDuration() const override;
  uint64_t getSequencenumber() const override;

  std::chrono::milliseconds getTimeout() const override {
    return timeout_ms_;
  }

  WorkStatus status() const {
    return status_.load(std::memory_order_acquire);
  }

  // Output tensors for result()/getFuture(). Set by the backend after issuing.
  void setOutputs(std::vector<at::Tensor> outputs) {
    outputs_ = std::move(outputs);
  }
  void setChildren(std::vector<c10::intrusive_ptr<WorkNCCL>> children) {
    children_ = std::move(children);
  }
  // Per-process-group collective counter of the op this work tracks; set by
  // the backend's createWork().
  void setSequenceNumber(uint64_t seq) {
    seq_ = seq;
  }
  void setOwnedEphemeralTimeout(std::chrono::milliseconds timeout) {
    owned_ephemeral_timeout_ = timeout;
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class ProcessGroupNCCL;
  friend class WorkNCCLQueue;
  friend class WindowNCCL;

 private:
  bool setTerminalStatus(WorkStatus status);
  // Poll the CUDA events and advance status; used by the GC queue + watchdog.
  WorkStatus checkStatus(
      std::optional<std::chrono::milliseconds> timeout = std::nullopt);
  void recordFunctionStart(std::string_view coll_name);
  // Make the current stream wait on the work's end event (the c10d "wait"
  // semantics for CUDA work: order subsequent current-stream ops after this).
  void synchronizeInternal();

  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;
  std::vector<at::Tensor> outputs_;
  std::vector<c10::intrusive_ptr<WorkNCCL>> children_;

  std::shared_ptr<WorkNCCLState> state_;
  c10::weak_intrusive_ptr<::c10d::Backend> comm_{
      c10::intrusive_ptr<::c10d::Backend>()};
  uint64_t comm_generation_{0};
  bool blocking_wait_{false};
  at::Device device_;
  int rank_;
  int comm_size_;
  std::string comm_name_;
  std::unique_ptr<at::cuda::CUDAEvent> start_event_;
  std::unique_ptr<at::cuda::CUDAEvent> end_event_;
  at::cuda::CUDAStream stream_;

  std::chrono::steady_clock::time_point work_start_time_;
  std::chrono::milliseconds timeout_ms_;
  std::chrono::milliseconds owned_ephemeral_timeout_{0};
  std::atomic<bool> ephemeral_timeout_released_{false};
  // Whether the events above were created with CUDA timing enabled, i.e.
  // whether getDuration() can be served for this work.
  bool timing_enabled_{false};
  uint64_t seq_{0};

  std::mutex terminal_status_mutex_;
  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};
  std::optional<at::RecordFunction> recordFunction_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
  c10::intrusive_ptr<c10::ivalue::Future> future_work_result_;

  // Set by the backend for a synchronous barrier: synchronizeInternal() then
  // host-blocks the CPU thread (in addition to the stream-ordered wait) to
  // mirror stock ProcessGroupNCCL, whose barrier host-blocks. See barrierImpl.
  bool hostBlocking_{false};
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
  std::queue<c10::intrusive_ptr<WorkNCCL>> completed_work_queue_;
  std::mutex work_queues_mutex_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
