#pragma once

#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

// Single place where <nccl_device.h> is included together with the HIP
// compatibility shims its content requires. TUs that need the device API
// should include this header instead of including <nccl_device.h> directly,
// so the shims are always defined before the header is parsed.

#ifdef NCCL_HAS_DEVCOMM_STORAGE
#if defined(NCCL_HAS_LSA_PEER_PTR)
// RCCL's device reduce/copy header expects CUDA compatibility helpers that
// hipcc does not provide under PyTorch's compile flags. Keep the definitions
// visible after the include so kernels in the same TU can rely on them.
#pragma push_macro("__CUDACC_EXTENDED_LAMBDA__")
#ifndef __CUDACC_EXTENDED_LAMBDA__
#define __CUDACC_EXTENDED_LAMBDA__ 1
#endif

static __device__ __forceinline__ unsigned int __vadd4(
    unsigned int a,
    unsigned int b) {
  unsigned int result;
  auto* out = reinterpret_cast<unsigned char*>(&result);
  auto* lhs = reinterpret_cast<unsigned char*>(&a);
  auto* rhs = reinterpret_cast<unsigned char*>(&b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = static_cast<unsigned char>(lhs[i] + rhs[i]);
  }
  return result;
}

#if defined(__HIP_NO_HALF_OPERATORS__)
static __device__ __forceinline__ __half operator+(
    const __half& a,
    const __half& b) {
  return __float2half(__half2float(a) + __half2float(b));
}
#endif

#include <nccl_device.h>

#pragma pop_macro("__CUDACC_EXTENDED_LAMBDA__")
// CUDA builds: nccl_dev_cap.hpp has already included <nccl_device.h>, so
// there is nothing left to do here.
#endif
#endif // NCCL_HAS_DEVCOMM_STORAGE
