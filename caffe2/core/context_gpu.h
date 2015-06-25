#ifndef CAFFE2_CORE_CONTEXT_GPU_H_
#define CAFFE2_CORE_CONTEXT_GPU_H_

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"
#include "glog/logging.h"

namespace caffe2 {

class CUDAContext {
 public:
  // The default cuda context constructor.
  CUDAContext()
      : cuda_stream_(nullptr), cublas_handle_(nullptr),
        random_seed_(1701), curand_generator_(nullptr) {
    cuda_gpu_id_ = GetDefaultGPUID();
    CUDA_CHECK(cudaSetDevice(cuda_gpu_id_));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
  }

  explicit CUDAContext(const DeviceOption& option)
      : cuda_stream_(nullptr), cublas_handle_(nullptr),
        random_seed_(option.random_seed()), curand_generator_(nullptr) {
    DCHECK_EQ(option.device_type(), CUDA);
    cuda_gpu_id_ = option.has_cuda_gpu_id() ?
                   option.cuda_gpu_id() : GetDefaultGPUID();
    CUDA_CHECK(cudaSetDevice(cuda_gpu_id_));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
  }

  virtual ~CUDAContext() {
    if (curand_generator_) {
      CURAND_CHECK(curandDestroyGenerator(curand_generator_));
    }
    if (cublas_handle_) {
      CUBLAS_CHECK(cublasDestroy(cublas_handle_));
    }
    if (cuda_stream_) {
      CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    }
  }

  inline void SwitchToDevice() {
    CUDA_CHECK(cudaSetDevice(cuda_gpu_id_));
  }

  inline bool FinishDeviceComputation() {
    cudaError_t error = cudaStreamSynchronize(cuda_stream_);
    if (error != cudaSuccess) {
      LOG(ERROR) << cudaGetErrorString(error);
      return false;
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
      LOG(ERROR) << cudaGetErrorString(error);
      return false;
    }
    return true;
  }

  int cuda_gpu_id() { return cuda_gpu_id_; }

  inline cudaStream_t& cuda_stream() { return cuda_stream_; }

  cublasHandle_t& cublas_handle() {
    if (!cublas_handle_) {
      CUBLAS_CHECK(cublasCreate(&cublas_handle_));
      CUBLAS_CHECK(cublasSetPointerMode(
          cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
      CUBLAS_CHECK(cublasSetStream(cublas_handle_, cuda_stream_));
    }
    return cublas_handle_;
  }

  curandGenerator_t& curand_generator() {
    if (!curand_generator_) {
      CURAND_CHECK(curandCreateGenerator(
          &curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
      CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
          curand_generator_, random_seed_));
      CURAND_CHECK(curandSetStream(curand_generator_, cuda_stream_));
    }
    return curand_generator_;
  }

  static void* New(size_t nbytes) {
    void* dev_ptr;
    CUDA_CHECK(cudaMalloc(&dev_ptr, nbytes));
    CUDA_CHECK(cudaMemset(dev_ptr, 0, nbytes));
    return dev_ptr;
  }

  static void Delete(void* data) {
    cudaError_t error = cudaFree(data);
    // For some reason, in Python runtime we sometimes delete a data pointer
    // after the cuda runtime exits - this is odd but is probably caused by
    // a static workspace that pycaffe2 uses, and the destruction got entangled
    // in some race condition. Anyway, since cuda runtime is exiting anyway, we
    // will not need to worry about memory leak, so we basically ignore it.
    // This is definitely not ideal but works for now.
    if (error != cudaSuccess && error != cudaErrorCudartUnloading) {
      LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": "
                 << cudaGetErrorString(error);
    }
  }

  template <class DstContext, class SrcContext>
  inline void Copy(void* dst, const void* src, size_t nbytes) {
    CUDA_CHECK(cudaMemcpyAsync(
        dst, src, nbytes, cudaMemcpyDefault, cuda_stream_));
    // TODO(Yangqing): do we want to synchronize inside copy?
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  }

  template <typename T, class DstContext, class SrcContext>
  inline void Copy(T* dst, const T* src, int n) {
    Copy<DstContext, SrcContext>(static_cast<void*>(dst),
                                 static_cast<const void*>(src),
                                 n * sizeof(T));
  }

 protected:
  int cuda_gpu_id_;
  cudaStream_t cuda_stream_;
  cublasHandle_t cublas_handle_;
  int random_seed_;
  curandGenerator_t curand_generator_;
};

// For the CPU context, we also allow a (probably expensive) function
// to copy the data from a cuda context.
template<>
inline void CPUContext::Memcpy<CPUContext, CUDAContext>(
    void* dst, const void* src, size_t nbytes) {
  CUDAContext context;
  context.Copy<CPUContext, CUDAContext>(dst, src, nbytes);
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_GPU_H_
