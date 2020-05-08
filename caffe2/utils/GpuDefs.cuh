#ifndef CAFFE2_UTILS_GPU_DEFS_H_
#define CAFFE2_UTILS_GPU_DEFS_H_

#include <cuda_runtime.h>

namespace caffe2 {

// Static definition of GPU warp size for unrolling and code generation

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ <= 750
constexpr int kWarpSize = 32;
#else
#error Unknown __CUDA_ARCH__; please define parameters for compute capability
#endif // __CUDA_ARCH__ types
#elif defined(__HIP_PLATFORM_HCC__)
constexpr int kWarpSize = warpSize;   // = 64 (Defined in hip_runtime.h)
#else
// dummy value for host compiler
constexpr int kWarpSize = 32;
#endif // __CUDA_ARCH__

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
#if defined(__HIP_PLATFORM_HCC__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
#endif // __HIP_PLATFORM_HCC__
  }

  static __device__ __forceinline__
  unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
#if defined(__HIP_PLATFORM_HCC__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
#else
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
#endif // __HIP_PLATFORM_HCC__
  }
};

template <>
struct Bitfield<unsigned long long int> {
  static __device__ __forceinline__
  unsigned long long int getBitfield(unsigned long long int val, int pos, int len) {
#if defined(__HIP_PLATFORM_HCC__)
    pos &= 0xff;
    len &= 0xff;

    unsigned long long int m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    unsigned long long int ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
#endif // __HIP_PLATFORM_HCC__
  }

  static __device__ __forceinline__
  unsigned long long int setBitfield(unsigned long long int val, unsigned long long int toInsert, int pos, int len) {
#if defined(__HIP_PLATFORM_HCC__)
    pos &= 0xff;
    len &= 0xff;

    unsigned long long int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
#else
    unsigned long long int ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;" :
        "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
    return ret;
#endif // __HIP_PLATFORM_HCC__
  }
};

__device__ __forceinline__ int getLaneId() {
#if defined(__HIP_PLATFORM_HCC__)
  return __lane_id();
#else
  int laneId;
  asm("mov.s32 %0, %%laneid;" : "=r"(laneId) );
  return laneId;
#endif // __HIP_PLATFORM_HCC__
}

#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ unsigned long long int getLaneMaskLt() {
  unsigned long long int m = (1ull << getLaneId()) - 1ull;
  return m;
}

__device__ __forceinline__ unsigned long long int getLaneMaskLe() {
  unsigned long long int m = UINT64_MAX >> (sizeof(std::uint64_t) * CHAR_BIT - (getLaneId() + 1));
  return m;
}

__device__ __forceinline__ unsigned long long int getLaneMaskGt() {
  unsigned long long int m = getLaneMaskLe();
  return m ? ~m : m;
}

__device__ __forceinline__ unsigned long long int getLaneMaskGe() {
  unsigned long long int m = getLaneMaskLt();
  return ~m;
}
#else
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
#endif // __HIP_PLATFORM_HCC__

}  // namespace caffe2

#endif  // CAFFE2_UTILS_GPU_DEFS_H_
