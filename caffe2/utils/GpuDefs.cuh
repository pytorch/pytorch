#ifndef CAFFE2_UTILS_GPU_DEFS_H_
#define CAFFE2_UTILS_GPU_DEFS_H_

#include <cuda_runtime.h>

namespace caffe2 {

// Static definition of GPU warp size for unrolling and code generation

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ <= 620
constexpr int kWarpSize = 32;
#else
#error Unknown __CUDA_ARCH__; please define parameters for compute capability
#endif // __CUDA_ARCH__ types
#endif // __CUDA_ARCH__

#ifndef __CUDA_ARCH__
// dummy value for host compiler
constexpr int kWarpSize = 32;
#endif // !__CUDA_ARCH__

//
// Interfaces to PTX instructions for which there appears to be no
// intrinsic
//

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__ __forceinline__
  unsigned int getBitfield(unsigned int val, int pos, int len) {
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__
  unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
  }
};

template <>
struct Bitfield<unsigned long long int> {
  static __device__ __forceinline__
  unsigned long long int getBitfield(unsigned long long int val, int pos, int len) {
    unsigned long long int ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__
  unsigned long long int setBitfield(unsigned long long int val, unsigned long long int toInsert, int pos, int len) {
    unsigned long long int ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;" :
        "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
    return ret;
  }
};

__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %laneid;" : "=r"(laneId) );
  return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

}  // namespace caffe2

#endif  // CAFFE2_UTILS_GPU_DEFS_H_
