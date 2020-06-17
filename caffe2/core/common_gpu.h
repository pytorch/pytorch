#ifndef CAFFE2_CORE_COMMON_GPU_H_
#define CAFFE2_CORE_COMMON_GPU_H_

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __HIP_PLATFORM_HCC__
#ifdef __GNUC__
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__
#endif // __HIP_PLATFORM_HCC__

#include <cublas_v2.h>
#include <curand.h>
#include <driver_types.h>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"

#include "c10/cuda/CUDAMacros.h"
#include "c10/cuda/CUDAMathCompat.h"
#include <c10/cuda/CUDAGuard.h>

#define CAFFE2_CUDA_EXPORT C10_EXPORT
#define CAFFE2_CUDA_API TORCH_CUDA_API

#define CAFFE2_HIP_EXPORT C10_EXPORT
#define CAFFE2_HIP_API TORCH_HIP_API

// This is a macro defined for cuda fp16 support. In default, cuda fp16 is
// supported by NVCC 7.5, but it is also included in the Tegra X1 platform with
// a (custom?) NVCC 7.0. As a result, we would normally just check the cuda
// version here, but would also allow a use to pass in the flag
// CAFFE_HAS_CUDA_FP16 manually.

#ifndef CAFFE_HAS_CUDA_FP16
#define CAFFE_HAS_CUDA_FP16
#endif // CAFFE_HAS_CUDA_FP16

#ifdef CAFFE_HAS_CUDA_FP16
#include <cuda_fp16.h>
#endif

// cuda major revision number below which fp16 compute is not supoorted
#ifndef __HIP_PLATFORM_HCC__
constexpr int kFp16CUDADevicePropMajor = 6;
#else
constexpr int kFp16CUDADevicePropMajor = 3;
#endif

// Re-enable strict aliasing diagnostic if it was disabled.
#ifndef __HIP_PLATFORM_HCC__
#ifdef __GNUC__
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif
#endif // __GNUC__
#endif // __HIP_PLATFORM_HCC__

/**
 * The maximum number of peers that each gpu can have when doing p2p setup.
 * Currently, according to NVidia documentation, each device can support a
 * system-wide maximum of eight peer connections.
 * When Caffe2 sets up peer access resources, if we have more than 8 gpus,
 * we will enable peer access in groups of 8.
 */
#define CAFFE2_CUDA_MAX_PEER_SIZE 8

namespace caffe2 {

#ifndef __HIP_PLATFORM_HCC__
/**
 * Empty class to identify TensorCore-based math
 */
class TensorCoreEngine {};
#endif // __HIP_PLATFORM_HCC__

#if CUDA_VERSION >= 10000
#define CAFFE2_CUDA_PTRATTR_MEMTYPE type
#else
#define CAFFE2_CUDA_PTRATTR_MEMTYPE memoryType
#endif

/**
 * A runtime function to report the cuda version that Caffe2 is built with.
 */
inline int CudaVersion() {
  return CUDA_VERSION;
}

/**
 * Returns the number of devices.
 */
CAFFE2_CUDA_API int NumCudaDevices();

/**
 * Check if the current running session has a cuda gpu present.
 *
 * Note that this is different from having caffe2 built with cuda. Building
 * Caffe2 with cuda only guarantees that this function exists. If there are no
 * cuda gpus present in the machine, or there are hardware configuration
 * problems like an insufficient driver, this function will still return false,
 * meaning that there is no usable GPU present.
 *
 * In the open source build, it is possible that Caffe2's GPU code is
 * dynamically loaded, and as a result a library could be only linked to the
 * CPU code, but want to test if cuda is later available or not. In this case,
 * one should use HasCudaRuntime() from common.h.
 */
inline bool HasCudaGPU() {
  return NumCudaDevices() > 0;
}

/**
 * Gets the current GPU id. This is a simple wrapper around cudaGetDevice().
 */
CAFFE2_CUDA_API int CaffeCudaGetDevice();

/**
 * Gets the current GPU id. This is a simple wrapper around cudaGetDevice().
 */
CAFFE2_CUDA_API void CaffeCudaSetDevice(const int id);

/**
 * Gets the GPU id that the current pointer is located at.
 */
CAFFE2_CUDA_API int GetGPUIDForPointer(const void* ptr);

/**
 * Gets the device property for the given device. This function is thread safe.
 */
CAFFE2_CUDA_API const cudaDeviceProp& GetDeviceProperty(const int device);

/**
 * Runs a device query function and prints out the results to LOG(INFO).
 */
CAFFE2_CUDA_API void DeviceQuery(const int deviceid);

/**
 * Return a peer access pattern by returning a matrix (in the format of a
 * nested vector) of boolean values specifying whether peer access is possible.
 *
 * This function returns false if anything wrong happens during the query of
 * the GPU access pattern.
 */
CAFFE2_CUDA_API bool GetCudaPeerAccessPattern(vector<vector<bool>>* pattern);

/**
 * Return the availability of TensorCores for math
 */
CAFFE2_CUDA_API bool TensorCoreAvailable();

/**
 * Return a human readable cublas error string.
 */
CAFFE2_CUDA_API const char* cublasGetErrorString(cublasStatus_t error);

/**
 * Return a human readable curand error string.
 */
CAFFE2_CUDA_API const char* curandGetErrorString(curandStatus_t error);

// CUDA: various checks for different function calls.
#define CUDA_ENFORCE(condition, ...) \
  do {                               \
    cudaError_t error = condition;   \
    CAFFE_ENFORCE_EQ(                \
        error,                       \
        cudaSuccess,                 \
        "Error at: ",                \
        __FILE__,                    \
        ":",                         \
        __LINE__,                    \
        ": ",                        \
        cudaGetErrorString(error),   \
        ##__VA_ARGS__);              \
  } while (0)
#define CUDA_CHECK(condition)                                 \
  do {                                                        \
    cudaError_t error = condition;                            \
    CHECK(error == cudaSuccess) << cudaGetErrorString(error); \
  } while (0)

#define CUDA_DRIVERAPI_ENFORCE(condition)                            \
  do {                                                               \
    CUresult result = condition;                                     \
    if (result != CUDA_SUCCESS) {                                    \
      const char* msg;                                               \
      cuGetErrorName(result, &msg);                                  \
      CAFFE_THROW("Error at: ", __FILE__, ":", __LINE__, ": ", msg); \
    }                                                                \
  } while (0)
#define CUDA_DRIVERAPI_CHECK(condition)                                 \
  do {                                                                  \
    CUresult result = condition;                                        \
    if (result != CUDA_SUCCESS) {                                       \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": " \
                 << msg;                                                \
    }                                                                   \
  } while (0)

#define CUBLAS_ENFORCE(condition)                \
  do {                                           \
    cublasStatus_t status = condition;           \
    CAFFE_ENFORCE_EQ(                            \
        status,                                  \
        CUBLAS_STATUS_SUCCESS,                   \
        "Error at: ",                            \
        __FILE__,                                \
        ":",                                     \
        __LINE__,                                \
        ": ",                                    \
        ::caffe2::cublasGetErrorString(status)); \
  } while (0)
#define CUBLAS_CHECK(condition)                    \
  do {                                             \
    cublasStatus_t status = condition;             \
    CHECK(status == CUBLAS_STATUS_SUCCESS)         \
        << ::caffe2::cublasGetErrorString(status); \
  } while (0)

#define CURAND_ENFORCE(condition)                \
  do {                                           \
    curandStatus_t status = condition;           \
    CAFFE_ENFORCE_EQ(                            \
        status,                                  \
        CURAND_STATUS_SUCCESS,                   \
        "Error at: ",                            \
        __FILE__,                                \
        ":",                                     \
        __LINE__,                                \
        ": ",                                    \
        ::caffe2::curandGetErrorString(status)); \
  } while (0)
#define CURAND_CHECK(condition)                    \
  do {                                             \
    curandStatus_t status = condition;             \
    CHECK(status == CURAND_STATUS_SUCCESS)         \
        << ::caffe2::curandGetErrorString(status); \
  } while (0)

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

// The following helper functions are here so that you can write a kernel call
// when you are not particularly interested in maxing out the kernels'
// performance. Usually, this will give you a reasonable speed, but if you
// really want to find the best performance, it is advised that you tune the
// size of the blocks and grids more reasonably.
// A legacy note: this is derived from the old good Caffe days, when I simply
// hard-coded the number of threads and wanted to keep backward compatibility
// for different computation capabilities.
// For more info on CUDA compute capabilities, visit the NVidia website at:
//    http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

// The number of cuda threads to use. Since work is assigned to SMs at the
// granularity of a block, 128 is chosen to allow utilizing more SMs for
// smaller input sizes.
// 1D grid
constexpr int CAFFE_CUDA_NUM_THREADS = 128;
// 2D grid
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMX = 16;
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMY = 16;

// The maximum number of blocks to use in the default kernel call. We set it to
// 4096 which would work for compute capability 2.x (where 65536 is the limit).
// This number is very carelessly chosen. Ideally, one would like to look at
// the hardware at runtime, and pick the number of blocks that makes most
// sense for the specific runtime environment. This is a todo item.
// 1D grid
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;
// 2D grid
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX = 128;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY = 128;

constexpr int kCUDAGridDimMaxX = 2147483647;
constexpr int kCUDAGridDimMaxY = 65535;
constexpr int kCUDAGridDimMaxZ = 65535;

/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int CAFFE_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
          CAFFE_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}

/**
 * @brief Compute the number of blocks needed to run N threads for a 2D grid
 */
inline dim3 CAFFE_GET_BLOCKS_2D(const int N, const int /* M */) {
  dim3 grid;
  // Not calling the 1D version for each dim to keep all constants as literals

  grid.x = std::max(
      std::min(
          (N + CAFFE_CUDA_NUM_THREADS_2D_DIMX - 1) /
              CAFFE_CUDA_NUM_THREADS_2D_DIMX,
          CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX),
      // Use at least 1 block, since CUDA does not allow empty block
      1);

  grid.y = std::max(
      std::min(
          (N + CAFFE_CUDA_NUM_THREADS_2D_DIMY - 1) /
              CAFFE_CUDA_NUM_THREADS_2D_DIMY,
          CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY),
      // Use at least 1 block, since CUDA does not allow empty block
      1);

  return grid;
}

using CUDAGuard = c10::cuda::CUDAGuard;

template <typename T, int N>
struct SimpleArray {
  T data[N];
};

constexpr int kCUDATensorMaxDims = 8;

#define DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(val, Func, T, ...) \
  do {                                                            \
    CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                    \
    switch (val) {                                                \
      case 1: {                                                   \
        Func<T, 1>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 2: {                                                   \
        Func<T, 2>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 3: {                                                   \
        Func<T, 3>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 4: {                                                   \
        Func<T, 4>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 5: {                                                   \
        Func<T, 5>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 6: {                                                   \
        Func<T, 6>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 7: {                                                   \
        Func<T, 7>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      case 8: {                                                   \
        Func<T, 8>(__VA_ARGS__);                                  \
        break;                                                    \
      }                                                           \
      default: {                                                  \
        break;                                                    \
      }                                                           \
    }                                                             \
  } while (false)

#define DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(val, Func, T1, T2, ...) \
  do {                                                                 \
    CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                         \
    switch (val) {                                                     \
      case 1: {                                                        \
        Func<T1, T2, 1>(__VA_ARGS__);                                  \
        break;                                                         \
      }                                                                \
      case 2: {                                                        \
        Func<T1, T2, 2>(__VA_ARGS__);                                  \
        break;                                                         \
      }                                                                \
      case 3: {                                                        \
        Func<T1, T2, 3>(__VA_ARGS__);                                  \
        break;                                                         \
      }                                                                \
      case 4: {                                                        \
        Func<T1, T2, 4>(__VA_ARGS__);                                  \
        break;                                                         \
      }                                                                \
      case 5: {                                                        \
        Func<T1, T2, 5>(__VA_ARGS__);                                  \
        break;                                                         \
      }                                                                \
      case 6: {                                                        \
        Func<T1, T2, 6>(__VA_ARGS__);                                  \
        break;                                                         \
      }                                                                \
      case 7: {                                                        \
        Func<T1, T2, 7>(__VA_ARGS__);                                  \
        break;                                                         \
      }                                                                \
      case 8: {                                                        \
        Func<T1, T2, 8>(__VA_ARGS__);                                  \
        break;                                                         \
      }                                                                \
      default: {                                                       \
        break;                                                         \
      }                                                                \
    }                                                                  \
  } while (false)

#define DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_3(val, Func, T1, T2, T3, ...) \
  do {                                                                     \
    CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                             \
    switch (val) {                                                         \
      case 1: {                                                            \
        Func<T1, T2, T3, 1>(__VA_ARGS__);                                  \
        break;                                                             \
      }                                                                    \
      case 2: {                                                            \
        Func<T1, T2, T3, 2>(__VA_ARGS__);                                  \
        break;                                                             \
      }                                                                    \
      case 3: {                                                            \
        Func<T1, T2, T3, 3>(__VA_ARGS__);                                  \
        break;                                                             \
      }                                                                    \
      case 4: {                                                            \
        Func<T1, T2, T3, 4>(__VA_ARGS__);                                  \
        break;                                                             \
      }                                                                    \
      case 5: {                                                            \
        Func<T1, T2, T3, 5>(__VA_ARGS__);                                  \
        break;                                                             \
      }                                                                    \
      case 6: {                                                            \
        Func<T1, T2, T3, 6>(__VA_ARGS__);                                  \
        break;                                                             \
      }                                                                    \
      case 7: {                                                            \
        Func<T1, T2, T3, 7>(__VA_ARGS__);                                  \
        break;                                                             \
      }                                                                    \
      case 8: {                                                            \
        Func<T1, T2, T3, 8>(__VA_ARGS__);                                  \
        break;                                                             \
      }                                                                    \
      default: {                                                           \
        break;                                                             \
      }                                                                    \
    }                                                                      \
  } while (false)

} // namespace caffe2

#endif // CAFFE2_CORE_COMMON_GPU_H_
