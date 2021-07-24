#ifndef CAFFE2_UTILS_GPU_ATOMICS_H_
#define CAFFE2_UTILS_GPU_ATOMICS_H_

#include <cuda_runtime.h>

namespace caffe2 {

namespace {

template <typename T>
inline __device__ void gpu_atomic_add(T* address, const T val) {
  atomicAdd(address, val);
}

template <>
inline __device__ void gpu_atomic_add(float* address, const float val) {
#if defined(__HIP_PLATFORM_HCC__) && defined(__gfx908__)
  atomicAddNoRet(address, val);
#else
  atomicAdd(address, val);
#endif
}

} // namespace

} // namespace caffe2

#endif  // CAFFE2_UTILS_GPU_ATOMICS_H_
