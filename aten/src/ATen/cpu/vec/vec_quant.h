#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <c10/util/Exception.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// Transpose a [4, 64] block to [64, 4] (with contiguous output, ld=4)
template <typename scalar_t, typename = std::enable_if_t<sizeof(scalar_t) == 1>>
static inline void transpose_pad_4x64_block(
    const scalar_t* src,
    scalar_t* dst,
    int64_t ld_src,
    int krem = 4,
    int nrem = 64) {
#if defined(CPU_CAPABILITY_AVX512)
  __m512i r[4];
  // Load with mask if partial
  if (nrem < 64) {
    __mmask64 mask = (1ULL << nrem) - 1;
    for (int i = 0; i < krem; ++i) {
      r[i] = _mm512_maskz_loadu_epi8(mask, src + i * ld_src);
    }
    for (int i = krem; i < 4; ++i) {
      r[i] = _mm512_setzero_si512();
    }
  } else {
    for (int i = 0; i < krem; ++i) {
      r[i] = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(src + i * ld_src));
    }
    for (int i = krem; i < 4; ++i) {
      r[i] = _mm512_setzero_si512();
    }
  }

  // Transpose 4x64 bytes using unpack and shuffle
  __m512i t0 = _mm512_unpacklo_epi8(r[0], r[1]);
  __m512i t1 = _mm512_unpackhi_epi8(r[0], r[1]);
  __m512i t2 = _mm512_unpacklo_epi8(r[2], r[3]);
  __m512i t3 = _mm512_unpackhi_epi8(r[2], r[3]);

  __m512i u0 = _mm512_unpacklo_epi16(t0, t2);
  __m512i u1 = _mm512_unpackhi_epi16(t0, t2);
  __m512i u2 = _mm512_unpacklo_epi16(t1, t3);
  __m512i u3 = _mm512_unpackhi_epi16(t1, t3);

  __m512i v0 = _mm512_shuffle_i32x4(u0, u1, 0x88);
  __m512i v1 = _mm512_shuffle_i32x4(u0, u1, 0xdd);
  __m512i v2 = _mm512_shuffle_i32x4(u2, u3, 0x88);
  __m512i v3 = _mm512_shuffle_i32x4(u2, u3, 0xdd);

  __m512i r0 = _mm512_shuffle_i32x4(v0, v2, 0x88);
  __m512i r1 = _mm512_shuffle_i32x4(v1, v3, 0x88);
  __m512i r2 = _mm512_shuffle_i32x4(v0, v2, 0xdd);
  __m512i r3 = _mm512_shuffle_i32x4(v1, v3, 0xdd);

  // Store output
  if (nrem < 16) {
    __mmask64 mask = (1ULL << (nrem * 4)) - 1;
    _mm512_mask_storeu_epi8(dst, mask, r0);
  } else if (nrem == 16) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), r0);
  } else if (nrem < 32) {
    int n_bytes1 = 64;
    int n_bytes2 = (nrem * 4) - n_bytes1;
    __mmask64 mask = (1ULL << n_bytes2) - 1;
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), r0);
    _mm512_mask_storeu_epi8(reinterpret_cast<__m512i*>(dst + 64), mask, r1);
  } else if (nrem == 32) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), r0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64), r1);
  } else if (nrem < 48) {
    int n_bytes1 = 64 * 2;
    int n_bytes2 = (nrem * 4) - n_bytes1;
    __mmask64 mask = (1ULL << n_bytes2) - 1;
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), r0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64), r1);
    _mm512_mask_storeu_epi8(reinterpret_cast<__m512i*>(dst + 64 * 2), mask, r2);
  } else if (nrem == 48) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), r0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64), r1);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64 * 2), r2);
  } else if (nrem < 64) {
    int n_bytes1 = 64 * 3;
    int n_bytes2 = (nrem * 4) - n_bytes1;
    __mmask64 mask = (1ULL << n_bytes2) - 1;
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), r0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64), r1);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64 * 2), r2);
    _mm512_mask_storeu_epi8(reinterpret_cast<__m512i*>(dst + 64 * 3), mask, r3);
  } else {
    // normal case, nrem == 64
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst), r0);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64), r1);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64 * 2), r2);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + 64 * 3), r3);
  }
#else
  TORCH_CHECK(
      false,
      "transpose_pad_4x64_block is only supported when AVX-512 is supported")
#endif
}

// Reorder [K, N] → [K/4, N, 4] (VNNI4-style layout for bit8)
template <typename scalar_t, typename = std::enable_if_t<sizeof(scalar_t) == 1>>
static inline void pack_vnni4(
    const scalar_t* src,
    scalar_t* dst,
    int64_t ld_src,
    int64_t K,
    int64_t N) {
#if defined(CPU_CAPABILITY_AVX512)
  int64_t bk = 0;
  int64_t _K = K / 4 * 4;
  int64_t _N = N / 64 * 64;
  for (; bk < _K; bk += 4) {
    int64_t bn = 0;
    for (; bn < _N; bn += 64) {
      transpose_pad_4x64_block(
          src + bk * ld_src + bn, dst + bk * N + bn * 4, ld_src);
    }
    int64_t nrem = N - bn;
    if (nrem > 0) {
      transpose_pad_4x64_block(
          src + bk * ld_src + bn, dst + bk * N + bn * 4, ld_src, 4, nrem);
    }
  }

  // Handle leftover K rows (< 4)
  if (K % 4 != 0) {
    int krem = K - bk;
    int64_t bn = 0;
    for (; bn < _N; bn += 64) {
      transpose_pad_4x64_block(
          src + bk * ld_src + bn, dst + bk * N + bn * 4, ld_src, krem);
    }
    int64_t nrem = N - bn;
    if (nrem > 0) {
      transpose_pad_4x64_block(
          src + bk * ld_src + bn, dst + bk * N + bn * 4, ld_src, krem, nrem);
    }
  }
#else
  TORCH_CHECK(false, "pack_vnni4 is only supported when AVX-512 is supported")
#endif
}

// This is a helper function for transpose_pack_vnni4
// Transform a [4, 16] block (with incontiguous output)
// Src:
// a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16
// b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16
// c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16
// d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15 d16
// Dst:
// a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 d1 d2 d3 d4
// a5 a6 a7 a8 b5 b6 b7 b8 c5 c6 c7 c8 d5 d6 d7 d8
// a9 a10 a11 a12 b9 b10 b11 b12 c9 c10 c11 c12 d9 d10 d11 d12
// a13 a14 a15 a16 b13 b14 b15 b16 c13 c14 c15 c16 d13 d14 d15 d16
template <typename scalar_t, typename = std::enable_if_t<sizeof(scalar_t) == 1>>
static inline void transpose_vnni4_pad_4x16_block(
    const scalar_t* src,
    scalar_t* dst,
    int64_t ld_src,
    int64_t ld_dst,
    int krem = 4) {
#if defined(CPU_CAPABILITY_AVX512)
  __m128i r[4];
  for (int i = 0; i < krem; ++i) {
    r[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * ld_src));
  }
  for (int i = krem; i < 4; ++i) {
    r[i] = _mm_setzero_si128();
  }

  // Transpose 4x16 bytes using unpack and shuffle
  __m128i t0 = _mm_unpacklo_epi32(r[0], r[1]);
  __m128i t1 = _mm_unpackhi_epi32(r[0], r[1]);
  __m128i t2 = _mm_unpacklo_epi32(r[2], r[3]);
  __m128i t3 = _mm_unpackhi_epi32(r[2], r[3]);

  __m128i r0 = _mm_unpacklo_epi64(t0, t2);
  __m128i r1 = _mm_unpackhi_epi64(t0, t2);
  __m128i r2 = _mm_unpacklo_epi64(t1, t3);
  __m128i r3 = _mm_unpackhi_epi64(t1, t3);

  // Store output
  if (krem == 4) {
    // normal case
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), r0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + ld_dst), r1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + ld_dst * 2), r2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + ld_dst * 3), r3);
  } else {
    // masked case
    __mmask16 mask = (1ULL << (krem * 4)) - 1;
    _mm_mask_storeu_epi8(dst, mask, r0);
    _mm_mask_storeu_epi8(reinterpret_cast<__m128i*>(dst + ld_dst), mask, r1);
    _mm_mask_storeu_epi8(
        reinterpret_cast<__m128i*>(dst + ld_dst * 2), mask, r2);
    _mm_mask_storeu_epi8(
        reinterpret_cast<__m128i*>(dst + ld_dst * 3), mask, r3);
  }
#else
  TORCH_CHECK(
      false,
      "transpose_vnni4_pad_4x16_block is only supported when AVX-512 is supported")
#endif
}

// Do the transpose packing fusion with VNNI4
// Reorder [K, N] → [N/4, K, 4] (VNNI4-style layout for bit8)
template <typename scalar_t, typename = std::enable_if_t<sizeof(scalar_t) == 1>>
static inline void transpose_pack_vnni4(
    const scalar_t* src,
    scalar_t* dst,
    int64_t ld_src,
    int64_t K,
    int64_t N) {
#if defined(CPU_CAPABILITY_AVX512)
  TORCH_CHECK(
      N % 16 == 0, "N needs to be multiple of 16 for transpose_pack_vnni4");
  int64_t bk = 0;
  int64_t _K = K / 4 * 4;
  for (; bk < _K; bk += 4) {
    int64_t bn = 0;
    for (; bn < N; bn += 16) {
      transpose_vnni4_pad_4x16_block(
          src + bk * ld_src + bn, dst + bn * K + bk * 4, ld_src, K * 4);
    }
  }

  // Handle leftover K rows (< 4)
  if (K % 4 != 0) {
    int krem = K - bk;
    int64_t bn = 0;
    for (; bn < N; bn += 16) {
      transpose_vnni4_pad_4x16_block(
          src + bk * ld_src + bn, dst + bn * K + bk * 4, ld_src, K * 4, krem);
    }
  }
#else
  TORCH_CHECK(
      false, "transpose_pack_vnni4 is only supported when AVX-512 is supported")
#endif
}

} // namespace CPU_CAPABILITY
} // namespace at::vec
