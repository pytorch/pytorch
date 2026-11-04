#pragma once

#ifdef USE_CUDA
// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>

#include <cuda.h>
#include <cuda_runtime.h>
#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

#include <memory>
#include <type_traits>

namespace torch::aot_inductor {

inline void delete_cuda_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_cuda_guard(reinterpret_cast<CUDAGuardHandle>(ptr)));
}

inline void delete_cuda_stream_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_cuda_stream_guard(
      reinterpret_cast<CUDAStreamGuardHandle>(ptr)));
}

inline void delete_cuda_tunableop_dynamic_dims_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_cuda_tunableop_dynamic_dims_guard(
          reinterpret_cast<CUDATunableOpDynamicDimsGuardHandle>(ptr)));
}

class AOTICudaGuard {
 public:
  AOTICudaGuard(int32_t device_index) : guard_(nullptr, delete_cuda_guard) {
    CUDAGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_cuda_guard(device_index, &ptr));
    guard_.reset(ptr);
  }

  void set_index(int32_t device_index) {
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_cuda_guard_set_index(guard_.get(), device_index));
  }

 private:
  std::unique_ptr<CUDAGuardOpaque, DeleterFnPtr> guard_;
};

class AOTICudaStreamGuard {
 public:
  AOTICudaStreamGuard(cudaStream_t stream, int32_t device_index)
      : guard_(nullptr, delete_cuda_stream_guard) {
    CUDAStreamGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_cuda_stream_guard(stream, device_index, &ptr));
    guard_.reset(ptr);
  }

 private:
  std::unique_ptr<CUDAStreamGuardOpaque, DeleterFnPtr> guard_;
};

class AOTICudaTunableOpDynamicDimsGuard {
 public:
  explicit AOTICudaTunableOpDynamicDimsGuard(uint8_t dynamic_dims_mask)
      : guard_(nullptr, delete_cuda_tunableop_dynamic_dims_guard) {
    CUDATunableOpDynamicDimsGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_cuda_tunableop_dynamic_dims_guard(
            dynamic_dims_mask, &ptr));
    guard_.reset(ptr);
  }

 private:
  std::unique_ptr<CUDATunableOpDynamicDimsGuardOpaque, DeleterFnPtr> guard_;
};

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0
// Deleters run in destructors, so they log errors instead of throwing.
inline void log_cuda_error(cudaError_t code, const char* call) {
  if (code != cudaSuccess) {
    std::cerr << "AOTInductor: " << call
              << " failed: " << cudaGetErrorString(code) << '\n';
    (void)cudaGetLastError();
  }
}

inline void delete_cuda_stream(void* ptr) {
  log_cuda_error(
      cudaStreamDestroy(static_cast<cudaStream_t>(ptr)), "cudaStreamDestroy");
}

inline void delete_cuda_event(void* ptr) {
  log_cuda_error(
      cudaEventDestroy(static_cast<cudaEvent_t>(ptr)), "cudaEventDestroy");
}

inline void delete_cuda_mem_pool(void* ptr) {
  log_shim_error(
      aoti_torch_delete_cuda_mem_pool(reinterpret_cast<CUDAMemPoolHandle>(ptr)),
      "failed to delete a CUDA memory pool, which leaks");
}

inline void delete_cuda_graph(void* ptr) {
  log_shim_error(
      aoti_torch_delete_cuda_graph(reinterpret_cast<CUDAGraphHandle>(ptr)),
      "failed to delete a CUDA graph, which leaks");
}

// A non-blocking CUDA stream, created on a device; capture streams must not
// synchronize with the legacy default stream.
class AOTICudaStream {
 public:
  explicit AOTICudaStream(int32_t device_index)
      : stream_(nullptr, delete_cuda_stream) {
    AOTICudaGuard device_guard(device_index);
    cudaStream_t stream = nullptr;
    AOTI_RUNTIME_CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    stream_.reset(stream);
  }

  cudaStream_t get() const {
    return stream_.get();
  }

 private:
  std::unique_ptr<std::remove_pointer_t<cudaStream_t>, DeleterFnPtr> stream_;
};

// A CUDA event without timing, created on a device.
class AOTICudaEvent {
 public:
  explicit AOTICudaEvent(int32_t device_index)
      : event_(nullptr, delete_cuda_event) {
    AOTICudaGuard device_guard(device_index);
    cudaEvent_t event = nullptr;
    AOTI_RUNTIME_CUDA_CHECK(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    event_.reset(event);
  }

  cudaEvent_t get() const {
    return event_.get();
  }

 private:
  std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, DeleterFnPtr> event_;
};

// A private memory pool for CUDA graphs to capture into. See the caller rules
// above the CUDA graph functions in shim.h on when it may be deleted.
class AOTICudaMemPool {
 public:
  explicit AOTICudaMemPool(int32_t device_index)
      : device_index_(device_index), pool_(nullptr, delete_cuda_mem_pool) {
    // A real index, which AOTICudaGraph's cleanup guards to.
    AOTI_RUNTIME_CHECK(device_index >= 0, "AOTICudaMemPool: bad device index");
    CUDAMemPoolHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_cuda_mem_pool(device_index, &ptr));
    pool_.reset(ptr);
  }

  CUDAMemPoolHandle get() const {
    return pool_.get();
  }

  int32_t device() const {
    return device_index_;
  }

 private:
  int32_t device_index_;
  std::unique_ptr<CUDAMemPoolOpaque, DeleterFnPtr> pool_;
};

// A CUDA graph without random number state, so capturing work that uses
// random numbers fails; see the caller rules above the CUDA graph functions in
// shim.h. A capture still open when the graph is destroyed, e.g. because the
// captured code threw, is ended first, so destroy the graph on the thread that
// began the capture, before its pool and its capture stream. If that capture
// recorded no work, ATen warns that the graph is empty; the warning is
// harmless there, since the graph is discarded.
class AOTICudaGraph {
 public:
  AOTICudaGraph() : graph_(nullptr, delete_cuda_graph) {
    CUDAGraphHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_cuda_graph(
        /*keep_graph=*/0, /*num_rng_states=*/0, &ptr));
    graph_.reset(ptr);
  }

  ~AOTICudaGraph() {
    end_open_capture();
  }

  AOTICudaGraph(const AOTICudaGraph&) = delete;
  AOTICudaGraph& operator=(const AOTICudaGraph&) = delete;

  // Captures the work enqueued on the current stream, which must be on the
  // pool's device; capture_end() ends it on that stream even if the captured
  // code changed the current stream.
  void capture_begin(const AOTICudaMemPool& pool, cudaStreamCaptureMode mode) {
    void* stream = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(pool.device(), &stream));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_cuda_graph_capture_begin(graph_.get(), pool.get(), mode));
    capture_stream_ = static_cast<cudaStream_t>(stream);
    device_index_ = pool.device();
  }

  void capture_end() {
    AOTI_RUNTIME_CHECK(
        capture_stream_ != nullptr, "AOTICudaGraph: no capture to end");
    AOTICudaStreamGuard guard(capture_stream_, device_index_);
    // Called once per capture, as the shim requires, even if it fails.
    capture_stream_ = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_cuda_graph_capture_end(graph_.get()));
  }

  void replay() {
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_graph_replay(graph_.get()));
  }

 private:
  // Active if the status can't be read.
  cudaStreamCaptureStatus capture_status() const noexcept {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(capture_stream_, &status) != cudaSuccess) {
      (void)cudaGetLastError();
      return cudaStreamCaptureStatusActive;
    }
    return status;
  }

  // Ends the capture if capture_end() wasn't called and it is still open,
  // active or invalidated, with its stream current, since the captured code
  // may have changed the current stream. An open capture on the stream is
  // then this graph's: no other capture can begin on it while this one is.
  void end_open_capture() noexcept {
    if (capture_stream_ == nullptr ||
        capture_status() == cudaStreamCaptureStatusNone) {
      return;
    }
    try {
      AOTICudaStreamGuard guard(capture_stream_, device_index_);
      (void)aoti_torch_cuda_graph_capture_end(graph_.get());
    } catch (...) {
    }
    (void)cudaGetLastError();
    if (capture_status() != cudaStreamCaptureStatusNone) {
      std::cerr << "AOTInductor: failed to end an abandoned CUDA graph "
                   "capture, so its stream stays capturing\n";
    }
  }

  std::unique_ptr<CUDAGraphOpaque, DeleterFnPtr> graph_;
  cudaStream_t capture_stream_ = nullptr;
  int32_t device_index_ = 0;
};
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

} // namespace torch::aot_inductor
#endif // USE_CUDA
