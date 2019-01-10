// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_CORE_CUDNN_WRAPPERS_H_
#define CAFFE2_CORE_CUDNN_WRAPPERS_H_

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

class CuDNNWrapper;

/**
 * CuDNNWorkspace is a wrapper around a raw cuda pointer that holds the cudnn
 * scratch space. This struct is meant to be only used in CuDNNWrapper to
 * provide a program-wide scratch space for CuDNN. The reason behind it is that
 * cudnn function calls are usually very efficient, hence one probably does not
 * want to run multiple cudnn calls at the same time. As a result, one should
 * not need more than one cudnn workspace per device.
 */
struct CuDNNWorkspace {
  ~CuDNNWorkspace() noexcept {}

  void* get(size_t nbytes) {
    if (nbytes_ < nbytes) {
      reset();
      auto data_and_deleter = CUDAContext::New(nbytes);
      data_ = {data_and_deleter.first, data_and_deleter.second};
      nbytes_ = nbytes;
    }
    CAFFE_ENFORCE_GE(nbytes_, nbytes);
    return data_.get();
  }

  void reset() {
    data_ = nullptr;
    nbytes_ = 0;
  }

 private:
  std::unique_ptr<void, MemoryDeleter> data_{nullptr, NoDelete};
  size_t nbytes_{0};
};

// CuDNNState is the owner of the CuDNNWorkspace, and serializes all
// executions of operations that use the state onto it's own stream
// (so multiple Net workers can reuse the same workspace from
// different threads and CUDA streams).
class CuDNNState {
 public:
  explicit CuDNNState(size_t gpu_id) : gpu_id_(gpu_id) {
    DeviceGuard g(gpu_id_);
    CUDNN_ENFORCE(cudnnCreate(&cudnn_handle_));
    CUDA_ENFORCE(cudaEventCreate(&before_));
    CUDA_ENFORCE(cudaEventCreate(&after_));
    CUDA_ENFORCE(cudaStreamCreate(&stream_));
    CUDNN_ENFORCE(cudnnSetStream(cudnn_handle_, stream_));
  }

  ~CuDNNState() noexcept {
    DeviceGuard g(gpu_id_);
    CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
    CUDA_CHECK(cudaStreamDestroy(stream_));
    CUDA_CHECK(cudaEventDestroy(after_));
    CUDA_CHECK(cudaEventDestroy(before_));
  }

  cudnnHandle_t& cudnn_handle() {
    return cudnn_handle_;
  }

  CuDNNWorkspace& workspace() {
    return workspace_;
  }

  template <typename F>
  void execute(cudaStream_t stream, F&& f) {
    CUDA_ENFORCE(cudaEventRecord(before_, stream));
    CUDA_ENFORCE(cudaStreamWaitEvent(stream_, before_, 0));
    f(this);
    CUDA_ENFORCE(cudaEventRecord(after_, stream_));
    CUDA_ENFORCE(cudaStreamWaitEvent(stream, after_, 0));
  }

 private:
  cudnnHandle_t cudnn_handle_{nullptr};
  cudaEvent_t before_{nullptr};
  cudaEvent_t after_{nullptr};
  cudaStream_t stream_{nullptr};
  CuDNNWorkspace workspace_;
  size_t gpu_id_{0};
  DISABLE_COPY_AND_ASSIGN(CuDNNState);
};

/**
 * CuDNNWrapper is a class that wraps the cudnn handles and cudnn workspaces.
 *
 * The wrapper ensures that for each thread and each gpu, there is one
 * identical cudnn handle, which is also associated with the thread-local
 * per-device cuda stream. The wrapper also hosts the device-specific cudnn
 * workspace (scratch space for some cudnn functions).
 *
 */
class CuDNNWrapper {
 public:
  /**
   * Creates a cudnn wrapper associated with a CUDAContext object. Note that
   * the CUDAContext object should outlive the CuDNNWrapper.
   */
  explicit CuDNNWrapper(CUDAContext* context) : context_(context) {}

  /**
   * Returns the inline cudnn handle that executes on the current
   * thread's cuda_stream.
   */
  cudnnHandle_t inline_cudnn_handle() {
    return context_->cudnn_handle();
  }

  // Executes the closure F on the CuDNNState associated with state_idx
  template <typename F>
  void with_cudnn_state(size_t state_idx, F&& f) {
    CAFFE_ENFORCE(
        state_idx < CAFFE2_COMPILE_TIME_MAX_CUDNN_STATES, "Invalid state_idx");
    auto& sync_state = cudnn_states()[context_->cuda_gpu_id()][state_idx];

    DeviceGuard dg(context_->cuda_gpu_id());

    // We need to serialize execution on the CuDNNState as we can't
    // allow multiple threads to race through the cudaEventRecord
    // calls (so a worker thread might wait on another worker thread's
    // execution)
    std::lock_guard<std::mutex> g(sync_state.mutex);
    if (!sync_state.state.get()) {
      sync_state.state.reset(new CuDNNState(context_->cuda_gpu_id()));
    }
    CHECK_NOTNULL(sync_state.state.get())->execute(context_->cuda_stream(), f);
  }

 protected:
  // Pointer to an external cuda context that the cudnn wrapper will use.
  CUDAContext* context_;

  static constexpr size_t CAFFE2_COMPILE_TIME_MAX_CUDNN_STATES = 4;

  struct SyncedCuDNNState {
    std::mutex mutex;
    std::unique_ptr<CuDNNState> state;
  };

  using PerGPUCuDNNStates = std::array<
      std::array<SyncedCuDNNState, CAFFE2_COMPILE_TIME_MAX_CUDNN_STATES>,
      CAFFE2_COMPILE_TIME_MAX_GPUS>;
  static PerGPUCuDNNStates& cudnn_states();

  DISABLE_COPY_AND_ASSIGN(CuDNNWrapper);
};

}; // namespace caffe2

#endif
