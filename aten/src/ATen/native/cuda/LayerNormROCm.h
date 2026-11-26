// Declarations for the ROCm cached-row LayerNorm / RMSNorm kernels (see LayerNormROCm.cuh). The kernels are
// instantiated per dtype in LayerNormROCm{Float,Half,BFloat16}.cu so they compile in parallel.
#pragma once

#ifdef USE_ROCM

#include <cstdint>

#include <cuda_runtime_api.h>

namespace at::native::rocm_ln {

struct FwdArgs {
  const void* X;
  const void* gamma;  // may be null
  const void* beta;   // may be null (always null for RMSNorm)
  void* Y;
  float* mean;  // null for RMSNorm
  float* rstd;
  int64_t M, N;
  float eps;
  bool rms;
  cudaStream_t stream;
};

struct BwdArgs {
  const void* dY;
  const void* X;
  const float* mean;  // null for RMSNorm
  const float* rstd;
  const void* gamma;  // required when dgamma / dbeta are requested
  void* dX;           // required
  void* dgamma;       // may be null
  void* dbeta;        // may be null (always null for RMSNorm)
  int64_t M, N;
  bool rms;
  int sm_count;
  cudaStream_t stream;
};

// Workspace provider for the backward kernel: alloc(n) returns n floats that stay valid until the stream has run
// the launched kernels (a caching-allocator tensor kept alive by the caller is enough).
struct WsAlloc {
  float* (*fn)(void* ctx, int64_t n);
  void* ctx;
  float* operator()(int64_t n) const { return fn(ctx, n); }
};

// Return false (and launch nothing) when the problem is not taken; the caller keeps its existing kernels.
template <typename T>
bool forward(const FwdArgs& k);
template <typename T>
bool backward(const BwdArgs& k, const WsAlloc& alloc);

}  // namespace at::native::rocm_ln

#endif  // USE_ROCM
