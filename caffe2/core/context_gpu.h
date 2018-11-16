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
 */
class CAFFE2_CUDA_API ThreadLocalCUDAObjects {
  friend class CUDAContext;

 private:
  ThreadLocalCUDAObjects() {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_GPUS; ++i) {
      cuda_streams_[i] = vector<cudaStream_t>();
      cublas_handles_[i] = vector<cublasHandle_t>();
#ifdef CAFFE2_USE_CUDNN
      cudnn_handles_[i] = vector<cudnnHandle_t>();
#endif // CAFFE2_USE_CUDNN
      current_stream_id_[i] = 0;
    }
  }

  // Record current stream id for the current thread.
  // This is the new API we're trying to migrate use cases to and get rid of
  // explicit stream id passing. For now it's invoked in
  // CUDAContext::SwitchToDevice
  void SetCurrentStreamId(int gpu, int stream_id) {
    // TODO: use current device id from thread local instead of passing gpu in
    current_stream_id_[gpu] = stream_id;
  }

  // Uses the logical stream id from the thread local to pick the stream
  // We're going to migrate all usages to this case API instead of passing the
  // stream id directly
  cudaStream_t GetStream(int gpu) {
    return GetStream(gpu, current_stream_id_[gpu]);
  }

  cudaStream_t GetStream(int gpu, int stream_id) {
    vector<cudaStream_t>& gpu_streams = cuda_streams_[gpu];
    if (gpu_streams.size() <= (unsigned)stream_id) {
      gpu_streams.resize(stream_id + 1, nullptr);
    }
    if (!gpu_streams[stream_id]) {
      DeviceGuard guard(gpu);
      CUDA_ENFORCE(cudaStreamCreateWithFlags(
          &gpu_streams[stream_id], cudaStreamNonBlocking));
    }
    return gpu_streams[stream_id];
  }

  // Uses the logical stream id from the thread local to pick the stream
  // We're going to migrate all usages to this case API instead of passing the
  // stream id directly
  cublasHandle_t GetHandle(int gpu) {
    return GetHandle(gpu, current_stream_id_[gpu]);
  }

  cublasHandle_t GetHandle(int gpu, int stream_id) {
    DeviceGuard guard(gpu);
    vector<cublasHandle_t>& gpu_handles = cublas_handles_[gpu];
    if (gpu_handles.size() <= (unsigned)stream_id) {
      gpu_handles.resize(stream_id + 1, nullptr);
    }
    if (!gpu_handles[stream_id]) {
      CUBLAS_ENFORCE(cublasCreate(&gpu_handles[stream_id]));
      // The default is CUBLAS_POINTER_MODE_HOST. You can override
      // it after obtaining the cublas handle, but do that with
      // caution.
      CUBLAS_ENFORCE(cublasSetPointerMode(
          gpu_handles[stream_id], CUBLAS_POINTER_MODE_HOST));
      CUBLAS_ENFORCE(
          cublasSetStream(gpu_handles[stream_id], GetStream(gpu, stream_id)));
    }
    return gpu_handles[stream_id];
  }

#ifdef CAFFE2_USE_CUDNN
  // Uses the logical stream id from the thread local to pick the stream
  // We're going to migrate all usages to this case API instead of passing the
  // stream id directly
  cudnnHandle_t GetCudnnHandle(int gpu) {
    return GetCudnnHandle(gpu, current_stream_id_[gpu]);
  }

  cudnnHandle_t GetCudnnHandle(int gpu, int stream_id) {
    DeviceGuard guard(gpu);
    vector<cudnnHandle_t>& gpu_handles = cudnn_handles_[gpu];
    if (gpu_handles.size() <= (unsigned)stream_id) {
      gpu_handles.resize(stream_id + 1, nullptr);
    }
    if (!gpu_handles[stream_id]) {
      CUDNN_ENFORCE(cudnnCreate(&gpu_handles[stream_id]));
      CUDNN_ENFORCE(
          cudnnSetStream(gpu_handles[stream_id], GetStream(gpu, stream_id)));
    }
    return gpu_handles[stream_id];
  }
#endif // CAFFE2_USE_CUDNN

  ~ThreadLocalCUDAObjects() noexcept {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_GPUS; ++i) {
      for (auto& handle : cublas_handles_[i]) {
        if (handle) {
          CUBLAS_CHECK(cublasDestroy(handle));
        }
      }
      for (auto& stream : cuda_streams_[i]) {
        if (stream) {
          CUDA_CHECK(cudaStreamDestroy(stream));
        }
      }

#ifdef CAFFE2_USE_CUDNN
      for (auto& handle : cudnn_handles_[i]) {
        if (handle) {
          CUDNN_CHECK(cudnnDestroy(handle));
        }
      }
#endif // CAFFE2_USE_CUDNN
    }
  }
  vector<cudaStream_t> cuda_streams_[CAFFE2_COMPILE_TIME_MAX_GPUS];
  vector<cublasHandle_t> cublas_handles_[CAFFE2_COMPILE_TIME_MAX_GPUS];
#ifdef CAFFE2_USE_CUDNN
  vector<cudnnHandle_t> cudnn_handles_[CAFFE2_COMPILE_TIME_MAX_GPUS];
#endif // CAFFE2_USE_CUDNN
  int current_stream_id_[CAFFE2_COMPILE_TIME_MAX_GPUS];
};

class CAFFE2_CUDA_API CUDAContext final : public BaseContext {
 public:
  // The default cuda context constructor.
  explicit CUDAContext(const int gpu_id = -1);
  explicit CUDAContext(const DeviceOption& option);
  explicit CUDAContext(const at::Device& device)
      : CUDAContext(DeviceToOption(device)) {}

  ~CUDAContext() override {
    if (curand_generator_) {
      CURAND_CHECK(curandDestroyGenerator(curand_generator_));
    }
    // CUDAContext is used in 2 cases now:
    // - long-lived instance inside OperatorBase in which case what happens in
    //   destructor doesn't really matter
    // - short-lived on-the-fly instances that are utilized as CUDAGuard - in
    //   this case there's only one stream id (passed to SwitchToDevice) and
    //   it's preferrable to synchronize in the destructor
    FinishDeviceComputation();
  }

  inline void SwitchToDevice(int stream_id) override {
    getCudaObjects().SetCurrentStreamId(gpu_id_, stream_id);
    CaffeCudaSetDevice(gpu_id_);
  }

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
    cudaStreamSynchronize(getCudaObjects().GetStream(gpu_id_));
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

  static cudaStream_t cuda_stream(int gpu_id, int stream_id) {
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
      DeviceGuard guard(gpu_id_);
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
  CopyItems(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "CUDAContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }

  // By default CUDA operators have async device parts
  static bool HasAsyncPartDefault() {
    return true;
  }

  static bool SupportsAsyncScheduling() {
    return true;
  }

  static bool IsStreamFree(const DeviceOption& option, int stream_id) {
    auto stream = CUDAContext::cuda_stream(option.device_id(), stream_id);
    return cudaStreamQuery(stream) == cudaSuccess;
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

// For the CPU context, we also allow a (probably expensive) function
// to copy the data from a cuda context. Inside the function, we create
// a temporary CUDAContext object to carry out the copy. From the caller's
// side, these functions are synchronous with respect to the host, similar
// to a normal CPUContext::CopyBytes<CPUContext, CPUContext> call.
template<>
inline void CPUContext::CopyBytes<CUDAContext, CPUContext>(
    size_t nbytes, const void* src, void* dst) {
  CUDAContext context(GetGPUIDForPointer(src));
  context.CopyBytes<CUDAContext, CPUContext>(nbytes, src, dst);
}
template<>
inline void CPUContext::CopyBytes<CPUContext, CUDAContext>(
    size_t nbytes, const void* src, void* dst) {
  CUDAContext context(GetGPUIDForPointer(dst));
  context.CopyBytes<CPUContext, CUDAContext>(nbytes, src, dst);
}

/**
 * An allocator that does the CPU memory allocation with pinned memory.
 *
 * This is needed because if we want to do any asynchronous cuda memcpy,
 * the underlying CPU memory also needs to be allocated into pinned memory
 * space. As a result, whenever Caffe2 is built with GPU and there is
 * GPU present during runtime, at global initialization time we will set
 * the CPU memory allocator to allocate pinned memory.
 */
struct CAFFE2_CUDA_API PinnedCPUAllocator final : public at::Allocator {
  PinnedCPUAllocator() {}
  ~PinnedCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data;
    at::DataPtr data_ptr;
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());
    if (IsNUMAEnabled()) {
      data_ptr = baseAllocator_.allocate(nbytes);
      data = data_ptr.get();
      CAFFE_ENFORCE(data);
      CUDA_ENFORCE(cudaHostRegister(data, nbytes, cudaHostRegisterDefault));
    } else {
      CUDA_ENFORCE(cudaMallocHost(&data, nbytes));
      data_ptr = {data, data, &Delete, at::Device(CPU)};
    }
    memset(data, 0, nbytes);
    return data_ptr;
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &Delete;
  }

 private:
  static void Delete(void* data) {
    // Caffe2 uses a lazy way to figure out if one is actually going to use GPUs
    // or not. If a CUDAContext::New() call is made, inside the CUDAContext
    // function we will switch the cpu side allocator to a PinnedCPUAllocator.
    // But, if one calls CPUContext::New() before any cuda allocations,
    // PinnedCPUAllocator can still delete the corresponding memory.
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());
    if (IsNUMAEnabled()) {
      CUDA_ENFORCE(cudaHostUnregister(data));
      DefaultCPUAllocator::Delete(data);
    } else {
      cudaError_t err = cudaFreeHost(data);
      if (err == cudaErrorInvalidValue) {
        free(data);
        // Calling cudaGetLastError will reset the cuda error.
        cudaGetLastError();
      } else {
        // For all other errors, still do a cuda check.
        CUDA_ENFORCE(err);
      }
    }
  }

  DefaultCPUAllocator baseAllocator_;
};

using TensorCUDA = Tensor;

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_GPU_H_
