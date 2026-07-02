#pragma once
#include <c10/metal/common.h>

template <unsigned N = c10::metal::max_ndim>
struct OrgqrParams {
  int32_t num_batch_dims;

  uint32_t m;
  uint32_t m2;
  uint32_t n;
  uint32_t k;

  ::c10::metal::array<uint32_t, N> A_strides;
  ::c10::metal::array<uint32_t, N> tau_strides;
  ::c10::metal::array<uint32_t, N> H_strides;
  ::c10::metal::array<uint32_t, N> H_sizes;
};

struct UnpackPivotsParams {
  uint32_t perm_batch_stride;
  uint32_t pivots_batch_stride;
  uint32_t dim_size;
};

struct QrParams {
  uint32_t m;
  uint32_t n;
};

struct SvdParams {
  uint32_t m; // staged rows = max(orig m,n) >= n
  uint32_t n; // staged cols = k = min(orig m,n)
  uint32_t max_sweeps;
  uint32_t compute_uv;
  float tol;
  uint32_t u_ld;
  uint32_t u_bstride;
  uint32_t v_ld;
  uint32_t v_bstride;
  uint32_t transposed; // 1 if SVD ran on A^T (left/right vectors swap targets)
  uint32_t stage_v; // 1: V accumulator in threadgroup mem (Vtg); 0: device mem
                    // (Vacc)
};

struct EighParams {
  uint32_t n;
  uint32_t max_sweeps;
  uint32_t compute_v;
  uint32_t upper; // UPLO: 1 read upper triangle, 0 read lower
  float tol;
};
