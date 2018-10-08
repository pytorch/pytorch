#ifndef CAFFE2_CORE_COMMON_HIP_H_
#define CAFFE2_CORE_COMMON_HIP_H_

#define HIP_VERSION 1
#include <assert.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprand.h>
#include <rocblas.h>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"

/**
 * The maximum number of AMD HIP GPUs that caffe2 recognizes.
 */
#define CAFFE2_COMPILE_TIME_MAX_HIP_GPUS 16
/**
 * The maximum number of peers that each gpu can have when doing p2p setup.
 * Currently, according to NVidia documentation, each device can support a
 * system-wide maximum of eight peer connections.
 * When Caffe2 sets up peer access resources, if we have more than 8 gpus,
 * we will enable peer access in groups of 8.
 */
#define CAFFE2_HIP_MAX_PEER_SIZE 8

#if CUDA_VERSION >= 10000
  #define CAFFE2_HIP_PTRATTR_MEMTYPE type
#else
  #define CAFFE2_HIP_PTRATTR_MEMTYPE memoryType
#endif

namespace caffe2 {

/**
 * A runtime function to report the HIP version that Caffe2 is built with.
 */
inline int HipVersion() { return HIP_VERSION; }

/**
 * Returns the number of devices.
 */
int NumHipDevices();

/**
 * Check if the current running session has a HIP gpu present.
 *
 * Note that this is different from having caffe2 built with HIP. Building
 * Caffe2 with HIP only guarantees that this function exists. If there are no
 * HIP gpus present in the machine, or there are hardware configuration
 * problems like an insufficient driver, this function will still return false,
 * meaning that there is no usable GPU present.
 *
 * In the open source build, it is possible that Caffe2's GPU code is
 * dynamically loaded, and as a result a library could be only linked to the
 * CPU code, but want to test if HIP is later available or not. In this case,
 * one should use HasHipRuntime() from common.h.
 */
inline bool HasHipGPU() { return NumHipDevices() > 0; }

/**
 * Sets the default GPU id for Caffe2.
 *
 * If an operator is set to run on HIP GPU but no gpu id is given, we will use
 * the default gpu id to run the operator. Before this function is explicitly
 * called, GPU 0 will be the default GPU id.
 */
void SetDefaultGPUID(const int deviceid);

/**
 * Gets the default GPU id for Caffe2.
 */
int GetDefaultGPUID();

/**
 * Gets the current GPU id. This is a simple wrapper around hipGetDevice().
 */
int CaffeHipGetDevice();

/**
 * Gets the current GPU id. This is a simple wrapper around hipSetDevice().
 */
void CaffeHipSetDevice(const int id);

/**
 * Gets the GPU id that the current pointer is located at.
 */
int GetGPUIDForPointer(const void* ptr);

/**
 * Gets the device property for the given device. This function is thread safe.
 */
const hipDeviceProp_t& GetDeviceProperty(const int device);

/**
 * Runs a device query function and prints out the results to LOG(INFO).
 */
void DeviceQuery(const int deviceid);

/**
 * Return a peer access pattern by returning a matrix (in the format of a
 * nested vector) of boolean values specifying whether peer access is possible.
 *
 * This function returns false if anything wrong happens during the query of
 * the GPU access pattern.
 */
bool GetHipPeerAccessPattern(vector<vector<bool>>* pattern);

/**
 * Return the availability of TensorCores for math
 */
bool TensorCoreAvailable();

/**
 * Return a human readable curand error string.
 */
const char* hiprandGetErrorString(hiprandStatus_t error);

/**
 * Return a human readable cublas error string.
 */
const char* rocblasGetErrorString(rocblas_status error);

// HIP: various checks for different function calls.
#define HIP_ENFORCE(condition, ...) \
  do {                              \
    hipError_t error = condition;   \
    CAFFE_ENFORCE_EQ(               \
        error,                      \
        hipSuccess,                 \
        "Error at: ",               \
        __FILE__,                   \
        ":",                        \
        __LINE__,                   \
        ": ",                       \
        hipGetErrorString(error),   \
        ##__VA_ARGS__);             \
  } while (0)
#define HIP_CHECK(condition)                                \
  do {                                                      \
    hipError_t error = condition;                           \
    CHECK(error == hipSuccess) << hipGetErrorString(error); \
  } while (0)

#define ROCBLAS_ENFORCE(condition)                \
  do {                                            \
    rocblas_status status = condition;            \
    CAFFE_ENFORCE_EQ(                             \
        status,                                   \
        rocblas_status_success,                   \
        "Error at: ",                             \
        __FILE__,                                 \
        ":",                                      \
        __LINE__,                                 \
        ": ",                                     \
        ::caffe2::rocblasGetErrorString(status)); \
  } while (0)

#define ROCBLAS_CHECK(condition)                    \
  do {                                              \
    rocblas_status status = condition;              \
    CHECK(status == rocblas_status_success)         \
        << ::caffe2::rocblasGetErrorString(status); \
  } while (0)

#define HIPRAND_ENFORCE(condition)                \
  do {                                            \
    hiprandStatus_t status = condition;           \
    CAFFE_ENFORCE_EQ(                             \
        status,                                   \
        HIPRAND_STATUS_SUCCESS,                   \
        "Error at: ",                             \
        __FILE__,                                 \
        ":",                                      \
        __LINE__,                                 \
        ": ",                                     \
        ::caffe2::hiprandGetErrorString(status)); \
  } while (0)
#define HIPRAND_CHECK(condition)                    \
  do {                                              \
    hiprandStatus_t status = condition;             \
    CHECK(status == HIPRAND_STATUS_SUCCESS)         \
        << ::caffe2::hiprandGetErrorString(status); \
  } while (0)

#define HIP_1D_KERNEL_LOOP(i, n)                                           \
  for (size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < (n); \
       i += hipBlockDim_x * hipGridDim_x)

// HIP_KERNEL_ASSERT is a macro that wraps an assert() call inside cuda
// kernels.
#define HIP_KERNEL_ASSERT(...)

// The following helper functions are here so that you can write a kernel call
// when you are not particularly interested in maxing out the kernels'
// performance. Usually, this will give you a reasonable speed, but if you
// really want to find the best performance, it is advised that you tune the
// size of the blocks and grids more reasonably.
// A legacy note: this is derived from the old good Caffe days, when I simply
// hard-coded the number of threads and wanted to keep backward compatibility
// for different computation capabilities.

// The number of HIP threads to use. 512 is used for backward compatibility,
// and it is observed that setting it to 1024 usually does not bring much
// performance gain (which makes sense, because warp size being 32 means that
// blindly setting a huge block for a random kernel isn't optimal).
constexpr int CAFFE_HIP_NUM_THREADS = 512;
// The maximum number of blocks to use in the default kernel call. We set it to
// 4096 which would work for compute capability 2.x (where 65536 is the limit).
// This number is very carelessly chosen. Ideally, one would like to look at
// the hardware at runtime, and pick the number of blocks that makes most
// sense for the specific runtime environment. This is a todo item.
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;

/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int CAFFE_GET_BLOCKS(const int N) {
  return std::min(
      (N + CAFFE_HIP_NUM_THREADS - 1) / CAFFE_HIP_NUM_THREADS,
      CAFFE_MAXIMUM_NUM_BLOCKS);
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int newDevice) : previous_(CaffeHipGetDevice()) {
    if (previous_ != newDevice) {
      CaffeHipSetDevice(newDevice);
    }
  }

  ~DeviceGuard() noexcept {
    CaffeHipSetDevice(previous_);
  }

 private:
  int previous_;
};

template <typename T, int N>
struct SimpleArray {
  T data[N];
};

constexpr int kHIPTensorMaxDims = 8;

#define DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(val, Func, T, ...) \
  do {                                                            \
    CAFFE_ENFORCE_LE(val, kHIPTensorMaxDims);                     \
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
      default: { break; }                                         \
    }                                                             \
  } while (false)

#define DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(val, Func, T1, T2, ...) \
  do {                                                                 \
    CAFFE_ENFORCE_LE(val, kHIPTensorMaxDims);                          \
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
      default: { break; }                                              \
    }                                                                  \
  } while (false)

#define DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_3(val, Func, T1, T2, T3, ...) \
  do {                                                                     \
    CAFFE_ENFORCE_LE(val, kHIPTensorMaxDims);                              \
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
      default: { break; }                                                  \
    }                                                                      \
  } while (false)

} // namespace caffe2

#endif // CAFFE2_CORE_COMMON_HIP_H_
