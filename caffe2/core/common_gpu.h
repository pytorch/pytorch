#ifndef CAFFE2_CORE_COMMON_GPU_H_
#define CAFFE2_CORE_COMMON_GPU_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
// #include <thrust/device_vector.h>
// #include <thrust/functional.h>

#include "glog/logging.h"
#include "caffe2/core/common.h"

namespace caffe2 {

// Sets and gets the default GPU id. If the function is not called, we will use
// GPU 0 ast he default gpu id. If there is an operator that says it runs on the
// GPU but did not specify which GPU, this default gpuid is going to be used.
void SetDefaultGPUID(const int deviceid);
int GetDefaultGPUID();
void DeviceQuery(const int deviceid);
// Return a peer access pattern by returning a matrix (in the format of a
// nested vector) of boolean values specifying whether peer access is possible.
bool GetCudaPeerAccessPattern(vector<vector<bool> >* pattern);

namespace internal {
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);
}  // namespace internal

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    CHECK_EQ(error, cudaSuccess)                                               \
        << "Error at: " << __FILE__ << ":" << __LINE__ << ": "                 \
        << cudaGetErrorString(error);                                          \
  } while (0)

#define CUBLAS_CHECK(condition)                                                \
  do {                                                                         \
    cublasStatus_t status = condition;                                         \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS)                                    \
        << "Error at: " << __FILE__ << ":" << __LINE__ << ": "                 \
        << ::caffe2::internal::cublasGetErrorString(status);                   \
  } while (0)

#define CURAND_CHECK(condition)                                                \
  do {                                                                         \
    curandStatus_t status = condition;                                         \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS)                                    \
        << "Error at: " << __FILE__ << ":" << __LINE__ << ": "                 \
        << ::caffe2::internal::curandGetErrorString(status);                   \
  } while (0)

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;                          \
       i < (n);                                                                \
       i += blockDim.x * gridDim.x)

// TODO(Yangqing): Yuck. Figure out a better way?
const int CAFFE_CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

}  // namespace caffe2
#endif  // CAFFE2_CORE_COMMON_GPU_H_
