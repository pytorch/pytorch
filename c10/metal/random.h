// Philox Counter based RNG implemntation for Metal
// Borrowed from aten/src/ATen/core/PhiloxRNGEngine.h
// Which in turn borrowed from
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
#pragma once
#include <metal_stdlib>

namespace c10 {
namespace metal {

namespace detail {

constexpr float uint32_to_uniform_float(uint32_t value) {
  // maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
  constexpr float scale = 4.6566127342e-10;
  return static_cast<float>(value & 0x7FFFFFFF) * scale;
}

inline uint2 splitlong(ulong v) {
  return uint2(v >> 32, v & 0xffffffff);
}

} // namespace detail

namespace philox4 {

uint2 mulhilo(uint a, uint b) {
  auto rc = static_cast<ulong>(a) * b;
  return detail::splitlong(rc);
}
uint4 single_round(uint4 ctr, uint2 key) {
  constexpr uint kPhiloxSA = 0xD2511F53;
  constexpr uint kPhiloxSB = 0xCD9E8D57;
  auto rc0 = mulhilo(kPhiloxSA, ctr.x);
  auto rc1 = mulhilo(kPhiloxSB, ctr.z);
  return uint4(rc1.y ^ ctr.y ^ key.x, rc1.x, rc0.y ^ ctr.w ^ key.y, rc0.x);
}

uint4 multiple_rounds(uint4 ctr, uint2 key, uint rounds) {
  constexpr uint2 kPhilox10 = {0x9E3779B9, 0xBB67AE85};
  for (uint round = 0; round < rounds - 1; ++round) {
    ctr = single_round(ctr, key);
    key += kPhilox10;
  }
  return ctr;
}

uint4 rand(long seed, long index) {
  uint4 ctr = 0;
  ctr.zw = detail::splitlong(index);
  return multiple_rounds(ctr, detail::splitlong(seed), 10);
}

} // namespace philox4

float randn(long seed, long index) {
  auto value = philox4::rand(seed, index);
  float u1 = 1.0 - detail::uint32_to_uniform_float(value.x);
  float u2 = 1.0 - detail::uint32_to_uniform_float(value.y);
  return ::metal::sqrt(-2.0 * ::metal::log(u1)) *
      ::metal::cos(2.0 * M_PI_F * u2);
}

float rand(long seed, long index) {
  auto value = philox4::rand(seed, index);
  return detail::uint32_to_uniform_float(value.x);
}

long randint64(long seed, long index, long low, long high) {
  auto range = high - low;
  auto value = philox4::rand(seed, index);
  // TODO: Implement better algorithm for large ranges
  return low +
      static_cast<long>(detail::uint32_to_uniform_float(value.x) * range);
}

} // namespace metal
} // namespace c10
