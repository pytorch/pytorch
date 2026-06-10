// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string_view>
#include <unordered_map>

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <torch/csrc/comms/TorchWork.hpp>

namespace torch::comms {

// Forward declaration
class TorchCommNCCL;
class TorchCommWindowNCCL;

// Forward declaration for test class
namespace test {
class TorchCommNCCLTest;
}

class TorchWorkNCCL : public TorchWork {
 public:
  TorchWorkNCCL(
      std::shared_ptr<TorchCommNCCL> comm,
      cudaStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);
  TorchWorkNCCL(
      std::shared_ptr<TorchCommNCCL> comm,
      cudaStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const at::Tensor& inputTensor);
  ~TorchWorkNCCL() override;

  // Delete copy and move operations
  TorchWorkNCCL(const TorchWorkNCCL&) = delete;
  TorchWorkNCCL(TorchWorkNCCL&&) = delete;
  TorchWorkNCCL& operator=(const TorchWorkNCCL&) = delete;
  TorchWorkNCCL& operator=(TorchWorkNCCL&&) = delete;

  // Override virtual functions from TorchWork
  void wait() override;
  std::chrono::milliseconds getTimeout() const override {
    return timeout_ms_;
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class TorchCommNCCL;
  friend class TorchCommWindowNCCL;
  friend class TorchWorkNCCLQueue;

 private:
  // Check the status of the work object
  WorkStatus checkStatus();

  void recordFunctionStart(std::string_view coll_name);

  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;

  std::shared_ptr<TorchCommNCCL> comm_;
  cudaEvent_t start_event_;
  cudaEvent_t end_event_;
  cudaStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;

  std::optional<at::RecordFunction> recordFunction_;
};

class TorchWorkNCCLQueue {
 public:
  TorchWorkNCCLQueue() = default;
  ~TorchWorkNCCLQueue() = default;

  TorchWorkNCCL::WorkStatus garbageCollect();
  // Finalize function can only be called from the main thread
  TorchWorkNCCL::WorkStatus finalize();
  void enqueueWork(c10::intrusive_ptr<TorchWorkNCCL> work, cudaStream_t stream);

 private:
  TorchWorkNCCL::WorkStatus garbageCollectLocked();
  std::
      unordered_map<cudaStream_t, std::queue<c10::intrusive_ptr<TorchWorkNCCL>>>
          stream_work_queues_;
  std::mutex work_queues_mutex_;
};

} // namespace torch::comms
