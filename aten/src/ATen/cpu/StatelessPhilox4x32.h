// Stateless Philox-4x32 PRNG implementation for CPU.
//
// CPU counterpart of ATen/cuda/StatelessPhilox4x32.cuh. Given (seed, offset)
// returns 4 pseudo-random uint32 values with no mutable state.
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

#include <array>
#include <cstdint>

namespace at::cpu {

inline std::pair<uint32_t, uint32_t> mulhilo32(uint32_t a, uint32_t b) {
  uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  return {static_cast<uint32_t>(product), static_cast<uint32_t>(product >> 32)};
}

inline std::array<uint32_t, 4> philox_round(
    std::array<uint32_t, 4> ctr, std::array<uint32_t, 2> key) {
  constexpr uint32_t kPhiloxSA = 0xD2511F53;
  constexpr uint32_t kPhiloxSB = 0xCD9E8D57;
  auto [lo0, hi0] = mulhilo32(kPhiloxSA, ctr[0]);
  auto [lo1, hi1] = mulhilo32(kPhiloxSB, ctr[2]);
  return {hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
}

// Stateless Philox-4x32. Returns 4 pseudo-random uint32 values (128 bits)
// determined entirely by (seed, offset). Each unique offset produces a
// distinct 128-bit output.
template <int N_ROUNDS = 10>
inline std::array<uint32_t, 4> philox_4x32(uint64_t seed, uint64_t offset) {
  std::array<uint32_t, 2> key = {
      static_cast<uint32_t>(seed),
      static_cast<uint32_t>(seed >> 32)};
  std::array<uint32_t, 4> ctr = {
      static_cast<uint32_t>(offset),
      static_cast<uint32_t>(offset >> 32),
      // restrict subsequence=0
      0, 0};

  constexpr uint32_t kPhilox10A = 0x9E3779B9;
  constexpr uint32_t kPhilox10B = 0xBB67AE85;

  for (int i = 0; i < N_ROUNDS - 1; i++) {
    ctr = philox_round(ctr, key);
    key[0] += kPhilox10A;
    key[1] += kPhilox10B;
  }
  return philox_round(ctr, key);
}

} // namespace at::cpu
