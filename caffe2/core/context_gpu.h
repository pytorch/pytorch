#ifndef CAFFE2_CORE_CONTEXT_GPU_H_
#define CAFFE2_CORE_CONTEXT_GPU_H_

#include <ctime>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

enum class CudaMemoryPoolType {
  NONE = 0,
  CNMEM = 1,
  CUB = 2,
};

/**
 * Gets the current memory pool type used by Caffe2.
 *
 * The memory pool is set up during caffe2's global initialization time.
 */
CudaMemoryPoolType GetCudaMemoryPoolType();

/**
 * An allocator that does the CPU memory allocation with pinned memory.
 *
 * This is needed because if we want to do any asynchronous cuda memcpy,
 * the underlying CPU memory also needs to be allocated into pinned memory
 * space. As a result, whenever Caffe2 is built with GPU and there is
 * GPU present during runtime, at global initialization time we will set
 * the CPU memory allocator to allocate pinned memory.
 */ 
struct PinnedCPUAllocator final : CPUAllocator {
  PinnedCPUAllocator() {}
  ~PinnedCPUAllocator() {}
  void* New(size_t nbytes) override {
    void* data;
    CUDA_CHECK(cudaMallocHost(&data, nbytes));
    memset(data, 0, nbytes);
    return data;
  }
  void Delete(void* data) override {
    CUDA_CHECK(cudaFreeHost(data));
  }
};

class CUDAContext;

/**
 * A struct to host thread-local cuda objects.
 *
 * In Caffe2, each thread has its own non-default cuda stream as well as
 * related objects such as cublas and curand handles. This is achieved by
 * having the ThreadLocalCUDAObjects wrapper that takes care of allocating
 * and deallocating these objects at the thread scope. This class is solely
 * used inside CUDAContext and should not be used externally.
 */
class ThreadLocalCUDAObjects {
  friend class CUDAContext;
 private:
  ThreadLocalCUDAObjects() {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_GPUS; ++i) {
      cuda_stream_[i] = nullptr;
      cublas_handle_[i] = nullptr;
    }
    for (int i = 0; i < NumCudaDevices(); ++i) {
      DeviceGuard guard(i);
      CUDA_CHECK(cudaStreamCreateWithFlags(
          &cuda_stream_[i], cudaStreamNonBlocking));
    }
  }

  ~ThreadLocalCUDAObjects() {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_GPUS; ++i) {
      if (cublas_handle_[i]) {
        cublasDestroy(cublas_handle_[i]);
      }
      cudaStreamDestroy(cuda_stream_[i]);
    }
  }
  cudaStream_t cuda_stream_[CAFFE2_COMPILE_TIME_MAX_GPUS];
  cublasHandle_t cublas_handle_[CAFFE2_COMPILE_TIME_MAX_GPUS];
};

class CUDAContext final {
 public:
  // The default cuda context constructor.
  explicit CUDAContext(const int gpu_id = -1)
      : gpu_id_(gpu_id == -1 ? GetDefaultGPUID() : gpu_id)
      , random_seed_(math::randomNumberSeed()) {
  }

  explicit CUDAContext(const DeviceOption& option)
      : gpu_id_(option.has_cuda_gpu_id() ?
                option.cuda_gpu_id() : GetDefaultGPUID()),
        random_seed_(option.has_random_seed() ?
                     option.random_seed() : math::randomNumberSeed()) {
    DCHECK_EQ(option.device_type(), CUDA);
  }

  ~CUDAContext() {
    if (curand_generator_) {
      CURAND_CHECK(curandDestroyGenerator(curand_generator_));
    }
    CHECK(FinishDeviceComputation());
  }

  inline void SwitchToDevice() {
    CUDA_CHECK(cudaSetDevice(gpu_id_));
  }

  bool FinishDeviceComputation() {
    cudaStreamSynchronize(cuda_objects_.cuda_stream_[gpu_id_]);
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
      return true;
    } else {
      LOG(ERROR) << "Encountered CUDA error: "
                      << cudaGetErrorString(error);
      return false;
    }
  }

  inline int cuda_gpu_id() const { return gpu_id_; }

  inline cudaStream_t& cuda_stream() {
    return cuda_stream(gpu_id_);
  }

  inline const cudaStream_t& cuda_stream() const {
    return cuda_stream(gpu_id_);
  }

  static cudaStream_t& cuda_stream(int gpu_id) {
    return cuda_objects_.cuda_stream_[gpu_id];
  }

  cublasHandle_t& cublas_handle() {
    auto& cublas_handle_ = cuda_objects_.cublas_handle_[gpu_id_];
    if (cublas_handle_) {
      return cublas_handle_;
    } else {
      DeviceGuard guard(gpu_id_);
      auto& cuda_stream_ = cuda_objects_.cuda_stream_[gpu_id_];
      CUBLAS_CHECK(cublasCreate(&cublas_handle_));
      // The default is CUBLAS_POINTER_MODE_HOST. You can override
      // it after obtaining the cublas handle, but do that with
      // caution.
      CUBLAS_CHECK(cublasSetPointerMode(
          cublas_handle_, CUBLAS_POINTER_MODE_HOST));
      CUBLAS_CHECK(cublasSetStream(cublas_handle_, cuda_stream_));
      return cublas_handle_;
    }
  }

  curandGenerator_t& curand_generator() {
    if (!curand_generator_) {
      DeviceGuard guard(gpu_id_);
      CURAND_CHECK(
          curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
      CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
          curand_generator_, random_seed_));
      CHECK_NOTNULL(curand_generator_);
    }
    CURAND_CHECK(curandSetStream(curand_generator_, cuda_stream()));
    return curand_generator_;
  }

  static void* New(size_t nbytes);

  static void Delete(void* data);

  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void* src, void* dst) {
    CUDA_CHECK(cudaMemcpyAsync(
        dst, src, nbytes, cudaMemcpyDefault,
        cuda_objects_.cuda_stream_[gpu_id_]));
  }

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T* src, T* dst) {
    CopyBytes<SrcContext, DstContext>(n * sizeof(T),
                                 static_cast<const void*>(src),
                                 static_cast<void*>(dst));
  }

 protected:
  int gpu_id_;
  int random_seed_;
  curandGenerator_t curand_generator_{nullptr};
  static thread_local ThreadLocalCUDAObjects cuda_objects_;
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

// For simplicity, we will typedef Tensor<CPUContext> to TensorCPU.
typedef Tensor<CUDAContext> TensorCUDA;

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_GPU_H_
