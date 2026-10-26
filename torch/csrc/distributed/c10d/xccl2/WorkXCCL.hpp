#pragma once

#ifdef USE_C10D_XCCL

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/xpu/XPUStream.h>

#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d::xccl2 {

class ProcessGroupXCCL;

// Pool of XPU events owned by a ProcessGroupXCCL. Kept separate from the
// backend so a work can drop its events after the backend is destroyed (user
// code may still hold an async handle when destroy_process_group() runs)
// without extending the backend's lifetime. Works hold it weakly.
class XcclEventPool {
 public:
  explicit XcclEventPool(size_t max_size, bool timing_enabled = false)
      : max_size_(max_size), timing_enabled_(timing_enabled) {}

  [[nodiscard]] std::unique_ptr<at::xpu::XPUEvent> get(bool timing_enabled);
  void put(std::unique_ptr<at::xpu::XPUEvent> event, bool timing_enabled);
  // Release pooled events at finalize() without waiting for works that are
  // still holding theirs.
  void clear();
  // Switches the pool to timing events, for Backend::enableCollectivesTiming().
  void enableTiming();
  bool timingEnabled() const;

 private:
  std::queue<std::unique_ptr<at::xpu::XPUEvent>> pool_;
  std::mutex mutex_;
  const size_t max_size_;
  std::atomic<bool> timing_enabled_;
};

// The backend identity a work still needs once it has been issued: the logging
// prefixes and the device whose current stream wait() orders against. Copied at
// construction, for the same lifetime reason as XcclEventPool. The accessors
// exist to satisfy the TC_LOG prefix helpers, which are templated on the comm.
struct XcclCommInfo {
  std::string name;
  int rank{};
  int size{};
  at::Device device{at::kXPU, 0};

  std::string_view getCommName() const {
    return name;
  }
  int getRank() const {
    return rank;
  }
};

// Work object for the XCCL TorchComms backend. Ported from torchcomms'
// TorchWorkXCCL, but rebased onto c10d::Work (upstream subclassed
// torchcomms::TorchWork). Completion is tracked with a pair of XPU events; the
// Future/result handling that the torchcomms BackendWrapper used to provide is
// folded in here (see setOutputs/getFuture). A work keeps no pointer to its
// backend: it can outlive it, so it snapshots the identity it needs and holds
// the event pool weakly.
class WorkXCCL : public c10d::Work {
 public:
  enum class WorkStatus {
    NOT_STARTED,
    INPROGRESS,
    COMPLETED,
    TIMEDOUT,
    ERROR,
  };

  WorkXCCL(
      ProcessGroupXCCL* comm,
      c10::xpu::XPUStream stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);
  WorkXCCL(
      ProcessGroupXCCL* comm,
      c10::xpu::XPUStream stream,
      std::chrono::milliseconds timeout_ms,
      at::Tensor inputTensor);
  ~WorkXCCL() override;

  WorkXCCL(const WorkXCCL&) = delete;
  WorkXCCL(WorkXCCL&&) = delete;
  WorkXCCL& operator=(const WorkXCCL&) = delete;
  WorkXCCL& operator=(WorkXCCL&&) = delete;

  // c10d::Work overrides.
  bool isCompleted() override;
  bool isSuccess() const override;
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
  void synchronize() override;
  std::vector<at::Tensor> result() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
  float getDuration() const override;
  uint64_t getSequencenumber() const override {
    return seq_;
  }

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
  // Per-process-group collective counter of the op this work tracks; set by
  // the backend's createWork().
  void setSequenceNumber(uint64_t seq) {
    seq_ = seq;
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class ProcessGroupXCCL;
  friend class WorkXCCLQueue;

 private:
  // Also records the DistBackendError reported by exception() and wait().
  void setStatus(WorkStatus status);
  // Poll the XPU events and advance status; used by the GC queue + watchdog.
  // timeout, if set, overrides the work's own timeout (see checkStatus).
  WorkStatus checkStatus(
      std::optional<std::chrono::milliseconds> timeout = std::nullopt);
  void recordFunctionStart(std::string_view coll_name);
  // Make the current stream wait on the work's end event (the c10d "wait"
  // semantics for accelerator work: order subsequent current-stream ops after
  // this collective).
  void synchronizeInternal();

  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;
  std::vector<at::Tensor> outputs_;

  XcclCommInfo comm_info_;
  std::weak_ptr<XcclEventPool> event_pool_;
  // Fixed at construction so the events go back to the pool only if it is
  // still in the same timing mode.
  bool timing_enabled_{false};
  std::unique_ptr<at::xpu::XPUEvent> start_event_;
  std::unique_ptr<at::xpu::XPUEvent> end_event_;
  c10::xpu::XPUStream stream_; // not owned by this class

  std::chrono::milliseconds timeout_ms_;
  uint64_t seq_{0};
  const std::chrono::steady_clock::time_point create_time_;

  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};
  // Guards checkStatus(), the only writer of status_ and start_completed_time_.
  std::mutex check_status_mutex_;
  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
  std::optional<at::RecordFunction> recordFunction_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

class WorkXCCLQueue {
 public:
  WorkXCCLQueue() = default;
  ~WorkXCCLQueue() = default;

  WorkXCCL::WorkStatus garbageCollect();
  // Finalize function can only be called from the main thread
  WorkXCCL::WorkStatus finalize();
  void enqueueWork(
      c10::intrusive_ptr<WorkXCCL> work,
      c10::xpu::XPUStream stream);
  bool empty();

 private:
  WorkXCCL::WorkStatus garbageCollectLocked();
  std::unordered_map<
      c10::xpu::XPUStream,
      std::queue<c10::intrusive_ptr<WorkXCCL>>>
      stream_work_queues_;
  // Completed work is parked here so that the tensor references it holds are
  // released by a caller thread rather than by the watchdog that runs
  // garbageCollect(); see WorkXCCLQueue.cpp.
  std::queue<c10::intrusive_ptr<WorkXCCL>> completed_work_queue_;
  std::mutex work_queues_mutex_;
};

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
