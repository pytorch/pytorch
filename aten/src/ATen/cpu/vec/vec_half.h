#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <c10/util/Exception.h>

#include <torch/headeronly/cpu/vec/vec_half.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// Transpose a [2, 32] matrix to [32, 2]
// Note: the output leading dimension should be 2,
// that is, the output must be contiguous
template <typename scalar_t, typename = std::enable_if_t<sizeof(scalar_t) == 2>>
static inline void transpose_pad_2x32_block(
    const scalar_t* src,
    scalar_t* dst,
    int64_t ld_src,
    int krem = 2,
    int nrem = 32) {
#if defined(CPU_CAPABILITY_AVX512)
  __m512i r0, r1;
  __m512i d0, d1;
  // load
  if (nrem < 32) {
    __mmask32 mask_krem_v = (1LL << nrem) - 1;
    r0 = _mm512_maskz_loadu_epi16(mask_krem_v, src);
    // if krem is not 2, pad with zeros
    if (krem == 2) {
      r1 = _mm512_maskz_loadu_epi16(mask_krem_v, src + ld_src);
    } else {
      r1 = _mm512_setzero_si512();
    }
  } else {
    r0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src));
    if (krem == 2) {
      r1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + ld_src));
    } else {
      r1 = _mm512_setzero_si512();
    }
  }
  // transpose
  d0 = _mm512_unpacklo_epi16(r0, r1);
  d1 = _mm512_unpackhi_epi16(r0, r1);
  r0 = _mm512_shuffle_i32x4(d0, d1, 0x88);
  r1 = _mm512_shuffle_i32x4(d0, d1, 0xdd);
  d0 = _mm512_shuffle_i32x4(r0, r1, 0x88);
  d1 = _mm512_shuffle_i32x4(r0, r1, 0xdd);

  // store
  if (nrem < 16) {
    __mmask32 mask_rem_v = (1LL << (nrem * 2)) - 1;
    _mm512_mask_storeu_epi16(dst, mask_rem_v, d0);
  } else if (nrem == 16) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
  } else if (nrem < 32) {
    __mmask32 mask_rem_v = (1LL << (nrem * 2 - 32)) - 1;
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
    _mm512_mask_storeu_epi16(
        reinterpret_cast<__m512i*>(dst + 32), mask_rem_v, d1);
  } else {
    // normal store
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), d0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 32), d1);
  }
#else
  TORCH_CHECK(
      false,
      "transpose_pad_2x32_block is only supported when avx512 is supported")
#endif
}

// To use AMX to accelerate GEMM,
// reorder the memory format [K, N] -> [K/2, N, 2]
// Note: If K % 2 != 0, pad K implicitly
template <typename scalar_t, typename = std::enable_if_t<sizeof(scalar_t) == 2>>
static inline void pack_vnni2(
    const scalar_t* src,
    scalar_t* dst,
    int64_t ld_src,
    int64_t K,
    int64_t N) {
#if defined(CPU_CAPABILITY_AVX512)
  int64_t bk = 0;
  int64_t _K = K / 2 * 2;
  int64_t _N = N / 32 * 32;
  for (; bk < _K; bk += 2) {
    int64_t bn = 0;
    for (; bn < _N; bn += 32) {
      transpose_pad_2x32_block(
          src + bk * ld_src + bn, dst + bk * N + bn * 2, ld_src);
    }
    int64_t nrem = N - bn;
    if (nrem > 0) {
      transpose_pad_2x32_block(
          src + bk * ld_src + bn, dst + bk * N + bn * 2, ld_src, 2, nrem);
    }
  }
  if (K % 2 == 1) {
    int64_t bn = 0;
    for (; bn < _N; bn += 32) {
      transpose_pad_2x32_block(
          src + bk * ld_src + bn, dst + bk * N + bn * 2, ld_src, 1);
    }
    int64_t nrem = N - bn;
    if (nrem > 0) {
      transpose_pad_2x32_block(
          src + bk * ld_src + bn, dst + bk * N + bn * 2, ld_src, 1, nrem);
    }
  }
#else
  TORCH_CHECK(false, "pack_vnni2 is only supported when avx512 is supported")
#endif
}

} // namespace CPU_CAPABILITY
} // namespace at::vec
