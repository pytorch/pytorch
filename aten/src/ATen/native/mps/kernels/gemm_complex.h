#pragma once
// gemm_complex.h - deinterleave / interleave helpers for the decomposed complex
// GEMM. A complex matmul C = A @ B is computed as four real GEMMs on the
// real/imag planes:
//   C = (ar@br - ai@bi) + i*(ar@bi + ai@br)
// complex_split peels an interleaved complex buffer into two contiguous real
// planes (so the real GEMM kernels see packed inputs), and complex_combine folds
// the four real products back into one interleaved complex result. The alpha/beta
// addmm epilogue is applied host-side (elementwise) on the combined result, which
// keeps the complex64 sub-GEMMs at full fp32 precision and avoids broadcasting the
// bias index into the kernel.
//
// Fully-templated port of metalBLAS complex_pack.h: C2 is the interleaved element
// (float2 / half2), R the real-component scalar (float / half).
#include <ATen/native/mps/kernels/gemm_common.h>
#include <metal_stdlib>

using namespace metal;

namespace at_gemm {

// Deinterleave src[i] = (re, im) into two contiguous real planes. One thread per
// complex element; src is read once as a coalesced C2 load.
template <typename C2, typename R>
kernel void complex_split(
    device const C2* src [[buffer(0)]],
    device R* re [[buffer(1)]],
    device R* im [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) {
    return;
  }
  C2 v = src[i];
  re[i] = v.x;
  im[i] = v.y;
}

// Fold the four real products into one interleaved complex result:
//   C = (P - Q) + i*(S + T), with P = ar@br, Q = ai@bi, S = ar@bi, T = ai@br.
template <typename R, typename C2>
kernel void complex_combine(
    device const R* P [[buffer(0)]],
    device const R* Q [[buffer(1)]],
    device const R* S [[buffer(2)]],
    device const R* T [[buffer(3)]],
    device C2* dst [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) {
    return;
  }
  float re = (float)P[i] - (float)Q[i];
  float im = (float)S[i] + (float)T[i];
  dst[i] = C2((R)re, (R)im);
}

} // namespace at_gemm
