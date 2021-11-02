#ifndef CAFFE2_CORE_CONTEXT_GPU_H_
#define CAFFE2_CORE_CONTEXT_GPU_H_

#include <ctime>
#include <mutex>

#include "caffe2/core/common.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context.h"
#include "caffe2/core/context_base.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/numa.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"

// Since we are using the macro CAFFE2_USE_CUDNN, we will need to include this
// file after common.h is included.
#ifdef CAFFE2_USE_CUDNN
#include "caffe2/core/common_cudnn.h"
#endif // CAFFE2_USE_CUDNN

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace caffe2 {

enum class CudaMemoryPoolType {
  NONE = 0,
  CUB = 1,
  THC = 2,
};

/**
 * Gets the current memory pool type used by Caffe2.
 *
 * The memory pool is set up during caffe2's global initialization time.
 */
CAFFE2_CUDA_API CudaMemoryPoolType GetCudaMemoryPoolType();

/**
 * A struct to host thread-local cuda objects.
 *
 * In Caffe2, each thread has its own non-default cuda stream as well as
 * related objects such as cublas and curand handles. This is achieved by
 * having the ThreadLocalCUDAObjects wrapper that takes care of allocating
 * and deallocating these objects at the thread scope. This class is solely
 * used inside CUDAContext and should not be used externally.
 *
 * This class manages the mapping from logical stream ID (int stream_id
 * passed around in Caffe2) and CUDAStream objects.  We intend to eventually
 * deprecate the logical stream ID interface, but not for now.
 */
class CAFFE2_CUDA_API ThreadLocalCUDAObjects {
  friend class CUDAContext;

 private:
  ThreadLocalCUDAObjects() {
    for (DeviceIndex i = 0; i < C10_COMPILE_TIME_MAX_GPUS; ++i) {
      cuda_streams_[i] = vector<c10::cuda::CUDAStream>();
    }
  }

  // Record current stream id for the current thread.
  // This is the new API we're trying to migrate use cases to and get rid of
  // explicit stream id passing. For now it's invoked in
  // CUDAContext::SwitchToDevice
  void SetCurrentStreamId(DeviceIndex gpu, StreamId stream_id) {
    // TODO: use current device id from thread local instead of passing gpu in
    if (stream_id != -1) {
      c10::cuda::setCurrentCUDAStream(GetCUDAStream(gpu, stream_id));
    }
  }

  // Retrieves the CUDAStream corresponding to a logical stream ID, ensuring
  // that it exists in cuda_streams_ if it has not been allocated yet.
  c10::cuda::CUDAStream GetCUDAStream(DeviceIndex gpu, StreamId stream_id) {
    vector<c10::cuda::CUDAStream>& gpu_streams = cuda_streams_[gpu];
    while (gpu_streams.size() <= static_cast<size_t>(stream_id)) {
      // NB: This streams are not guaranteed to be unique; we'll
      // wrap around once we run out of streams in the pool.
      gpu_streams.emplace_back(c10::cuda::getStreamFromPool(/* high priority */ false, gpu));
    }
    return gpu_streams[stream_id];
  }

  // Uses the logical stream id from the thread local to pick the stream
  // We're going to migrate all usages to this case API instead of passing the
  // stream id directly
  cudaStream_t GetStream(DeviceIndex gpu) {
    return c10::cuda::getCurrentCUDAStream(gpu).stream();
  }

  cudaStream_t GetStream(DeviceIndex gpu, StreamId stream_id) {
    return GetCUDAStream(gpu, stream_id).stream();
  }

  // Uses the logical stream id from the thread local to pick the stream
  // We're going to migrate all usages to this case API instead of passing the
  // stream id directly
  cublasHandle_t GetHandle(DeviceIndex gpu) {
    return GetHandle(c10::cuda::getCurrentCUDAStream(gpu));
  }

  cublasHandle_t GetHandle(c10::cuda::CUDAStream cuda_stream) {
    CUDAGuard guard(cuda_stream.device_index());
    // Default construct in the map if it doesn't exist, and return a mutable
    // reference to it.
    auto& r = cublas_handles_[cuda_stream];
    if (r == nullptr) {
      CUBLAS_ENFORCE(cublasCreate(&r));
      // The default is CUBLAS_POINTER_MODE_HOST. You can override
      // it after obtaining the cublas handle, but do that with
      // caution.
      CUBLAS_ENFORCE(cublasSetPointerMode(r, CUBLAS_POINTER_MODE_HOST));
      CUBLAS_ENFORCE(cublasSetStream(r, cuda_stream));
    }
    return r;
  }

#ifdef CAFFE2_USE_CUDNN
  // Uses the logical stream id from the thread local to pick the stream
  // We're going to migrate all usages to this case API instead of passing the
  // stream id directly
  cudnnHandle_t GetCudnnHandle(DeviceIndex gpu) {
    return GetCudnnHandle(c10::cuda::getCurrentCUDAStream(gpu));
  }

  cudnnHandle_t GetCudnnHandle(c10::cuda::CUDAStream cuda_stream) {
    CUDAGuard guard(cuda_stream.device_index());
    auto& r = cudnn_handles_[cuda_stream];
    if (r == nullptr) {
      CUDNN_ENFORCE(cudnnCreate(&r));
      CUDNN_ENFORCE(cudnnSetStream(r, cuda_stream));
    }
    return r;
  }
#endif // CAFFE2_USE_CUDNN

  ~ThreadLocalCUDAObjects() noexcept {
    for (auto element : cublas_handles_) {
      if (element.second) {
        CUBLAS_CHECK(cublasDestroy(element.second));
      }
    }
#ifdef CAFFE2_USE_CUDNN
    for (auto element : cudnn_handles_) {
      if (element.second) {
        CUDNN_CHECK(cudnnDestroy(element.second));
      }
    }
#endif // CAFFE2_USE_CUDNN
  }
  // WARNING: mapping from logical stream ID to c10::cuda::CUDAStream
  // is NOT bijective; multiple logical stream IDs may map to the
  // same underlying stream ID.
  vector<c10::cuda::CUDAStream> cuda_streams_[C10_COMPILE_TIME_MAX_GPUS];
  std::unordered_map<c10::cuda::CUDAStream, cublasHandle_t> cublas_handles_;
#ifdef CAFFE2_USE_CUDNN
  std::unordered_map<c10::cuda::CUDAStream, cudnnHandle_t> cudnn_handles_;
#endif // CAFFE2_USE_CUDNN
};

class CAFFE2_CUDA_API CUDAContext final : public BaseContext {
 public:
  // The default cuda context constructor.
  explicit CUDAContext(DeviceIndex gpu_id = -1);
  explicit CUDAContext(const DeviceOption& option);
  explicit CUDAContext(Device device)
      : CUDAContext(DeviceToOption(device)) {}

  ~CUDAContext() override;

  inline void SwitchToDevice(StreamId stream_id) override {
    getCudaObjects().SetCurrentStreamId(gpu_id_, stream_id);
    CaffeCudaSetDevice(gpu_id_);
  }

  // void SwitchToDevice()
  using BaseContext::SwitchToDevice;

  inline void WaitEvent(const Event& ev) override {
    ev.Wait(CUDA, this);
  }

  inline void Record(Event* ev, const char* err_msg = nullptr) const override {
    CAFFE_ENFORCE(ev, "Event must not be null.");
    ev->Record(CUDA, this, err_msg);
  }

  // Note on current use cases:
  // FinishDeviceComputation must be called on the same cpu thread as
  // SwitchToDevice()
  void FinishDeviceComputation() override {
    CUDA_ENFORCE(cudaStreamSynchronize(getCudaObjects().GetStream(gpu_id_)));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      CAFFE_THROW("Encountered CUDA error: ", cudaGetErrorString(error));
    }
  }

  inline int device_id() const {
    return gpu_id_;
  }

  inline cudaStream_t cuda_stream() const {
    return getCudaObjects().GetStream(gpu_id_);
  }

  static cudaStream_t cuda_stream(DeviceIndex gpu_id, StreamId stream_id) {
    return getCudaObjects().GetStream(gpu_id, stream_id);
  }

  cublasHandle_t cublas_handle() {
    return getCudaObjects().GetHandle(gpu_id_);
  }

#ifdef CAFFE2_USE_CUDNN
  cudnnHandle_t cudnn_handle() {
    return getCudaObjects().GetCudnnHandle(gpu_id_);
  }
#endif // CAFFE2_USE_CUDNN

  curandGenerator_t& curand_generator() {
    if (!curand_generator_) {
      CUDAGuard guard(gpu_id_);
      CURAND_ENFORCE(
          curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
      CURAND_ENFORCE(
          curandSetPseudoRandomGeneratorSeed(curand_generator_, random_seed_));
      CHECK_NOTNULL(curand_generator_);
    }
    CURAND_ENFORCE(curandSetStream(curand_generator_, cuda_stream()));
    return curand_generator_;
  }

  inline static at::DataPtr New(size_t nbytes) {
    return GetAllocator(CUDA)->allocate(nbytes);
  }

  // Get a mutex to lock out cudaMalloc / cudaFree calls when
  // NCCL kernels are being launched. Should remove threat of
  // deadlocks
  static std::mutex& mutex();

  // Functions to query memory stats. Only available if flag
  // --caffe2_gpu_memory_tracking is enabled.
  static std::vector<long> TotalMemoryByGpu();
  static std::vector<long> MaxMemoryByGpu();

  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void* src, void* dst) {
    CUDA_ENFORCE(cudaMemcpyAsync(
        dst,
        src,
        nbytes,
        cudaMemcpyDefault,
        getCudaObjects().GetStream(gpu_id_)));
  }

  void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) override {
    CopyBytes<CUDAContext, CUDAContext>(nbytes, src, dst);
  }

  void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytes<CUDAContext, CPUContext>(nbytes, src, dst);
  }

  void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytes<CPUContext, CUDAContext>(nbytes, src, dst);
  }

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T* src, T* dst) {
    CopyBytes<SrcContext, DstContext>(n * sizeof(T),
                                 static_cast<const void*>(src),
                                 static_cast<void*>(dst));
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "CUDAContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }

  static void CopyBytesAsync(
      size_t nbytes,
      const void* src,
      Device src_device,
      void* dst,
      Device dst_device);
  static void CopyBytesSync(
      size_t nbytes,
      const void* src,
      Device src_device,
      void* dst,
      Device dst_device);

  // By default CUDA operators have async device parts
  static bool HasAsyncPartDefault() {
    return true;
  }

  static bool SupportsAsyncScheduling() {
    return true;
  }

  static bool IsStreamFree(const DeviceOption& option, StreamId stream_id) {
    auto stream = CUDAContext::cuda_stream(option.device_id(), stream_id);
    auto status = cudaStreamQuery(stream);
    if (status == cudaErrorNotReady) {
      // ignore and clear the error if not ready
      (void)cudaGetLastError();
    }
    return status == cudaSuccess;
  }

  at::Device device() const override {
    return at::Device(CUDA, gpu_id_);
  }

  DeviceType device_type() const override {
    return CUDA;
  }

  static constexpr DeviceType GetDeviceType() {
    return CUDA;
  }

 protected:
  int gpu_id_;
  int random_seed_;
  curandGenerator_t curand_generator_{nullptr};
  static ThreadLocalCUDAObjects& getCudaObjects();
};

using TensorCUDA = Tensor;

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_GPU_H_
