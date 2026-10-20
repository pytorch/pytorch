// Stateless Philox-4x32 PRNG implementation.
//
// Unlike PhiloxRNGEngine (PhiloxUtils.cuh), this is a pure function: given
// (seed, offset) it returns 4 pseudo-random uint32 values with no mutable
// state. This makes it suitable for use in stateless random APIs.
//
// The Philox-4x32 cipher operates on a 128-bit counter. The full counter
// is (offset_lo, offset_hi, subsequence_lo, subsequence_hi), but we fix
// subsequence=0 so that the entire 128-bit counter space is addressed by
// the 64-bit offset alone. This keeps the API simple and maintains
// cross-device consistency. For example, utilizing thread ID-based subsequence
// numbers and SM-based thread count causes different random values to
// be generated across GPU types. We avoid this situation by always setting
// subsequence=0.

#pragma once

#include <cstdint>

namespace at::cuda {

__device__ __forceinline__ uint2 mulhilo32(uint32_t a, uint32_t b) {
  return {a * b, __umulhi(a, b)};
}

__device__ __forceinline__ uint4 philox_round(uint4 ctr, uint2 key) {
  constexpr uint32_t kPhiloxSA = 0xD2511F53;
  constexpr uint32_t kPhiloxSB = 0xCD9E8D57;
  uint2 r0 = mulhilo32(kPhiloxSA, ctr.x);
  uint2 r1 = mulhilo32(kPhiloxSB, ctr.z);
  return {r1.y ^ ctr.y ^ key.x, r1.x, r0.y ^ ctr.w ^ key.y, r0.x};
}

// Stateless Philox-4x32. Returns 4 pseudo-random uint32 values (128 bits)
// determined entirely by (seed, offset). Each unique offset produces a
// distinct 128-bit output.
template <int N_ROUNDS = 10>
__device__ __forceinline__ uint4 philox_4x32(
    uint64_t seed, uint64_t offset) {
  uint2 key = {
      static_cast<uint32_t>(seed),
      static_cast<uint32_t>(seed >> 32)};
  uint4 ctr = {
      static_cast<uint32_t>(offset),
      static_cast<uint32_t>(offset >> 32),
      // restrict subsequence=0
      0, 0};

  constexpr uint32_t kPhilox10A = 0x9E3779B9;
  constexpr uint32_t kPhilox10B = 0xBB67AE85;

  #pragma unroll
  for (int i = 0; i < N_ROUNDS - 1; i++) {
    ctr = philox_round(ctr, key);
    key.x += kPhilox10A;
    key.y += kPhilox10B;
  }
  return philox_round(ctr, key);
}

} // namespace at::cuda
