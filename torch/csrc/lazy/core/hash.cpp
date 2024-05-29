/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/util.h
 */
#include <iomanip>
#include <sstream>

#include <torch/csrc/lazy/core/hash.h>

namespace torch {
namespace lazy {
namespace {

hash_t LoadHash(const uint8_t** data, const uint8_t* top) {
  std::ptrdiff_t size = top - (*data);
  if (size >= (int)sizeof(hash_t)) {
    hash_t v;
    std::memcpy(&v, *data, sizeof(v));
    *data += sizeof(hash_t);
    return v;
  }
  union {
    hash_t h;
    std::array<uint8_t, sizeof(hash_t)> b;
#ifdef _MSC_VER
    // MSVC (or some versions we use) doesn't support C99 union field init
    // but it initializes the first member of the union.
  } uval = {hash_t(0)};
#else
  } uval = {.h = hash_t(0)};
#endif
  // use memcpy for compatibility with platforms not supporting unaligned access
  // note: compiled as single `movl` instr on x64.
  std::memcpy(uval.b.data(), *data, size);
  *data += size;
  return uval.h;
}

} // namespace

hash_t HashBlock(const void* data, size_t n, const hash_t& seed) {
  const hash_t m(static_cast<uint64_t>(0xc6a4a7935bd1e995));
  const int r = 47;

  const uint8_t* u8_data = reinterpret_cast<const uint8_t*>(data);
  const uint8_t* top = u8_data + n;
  hash_t h(seed ^ ((uint64_t)n * m));
  while (u8_data < top) {
    hash_t k = LoadHash(&u8_data, top);
    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }
  h ^= h >> r;
  h *= m;
  h ^= h >> r;
  return h;
}

hash_t DataHash(const void* data, size_t size) {
  return HashBlock(
      data, size, hash_t(static_cast<uint64_t>(0xc2b2ae3d27d4eb4f)));
}

size_t StdDataHash(const void* data, size_t size) {
  return HashReduce(DataHash(data, size));
}

size_t StdHashCombine(uintmax_t a, uintmax_t b) {
  return a ^
      (b * 0x27d4eb2f165667c5 + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}

hash_t HashCombine(const hash_t& a, const hash_t& b) {
  static const hash_t kb(101, 0x27d4eb2f165667c5);
  return hash_t(
      a ^ (b * kb + (uint64_t)0x9e3779b97f4a7c15 + (a << 6) + (a >> 2)));
}

size_t HashReduce(const hash_t& a) {
  return StdHashCombine(c10::Uint128Low64(a), c10::Uint128High64(a));
}

std::string HashToString(const hash_t& a) {
  std::stringstream ss;
  ss << std::hex << c10::Uint128High64(a) << std::setfill('0') << std::setw(16)
     << Uint128Low64(a);
  return ss.str();
}

hash_t Hash(const std::vector<bool>& values) {
  // We can't assume a DataHash size/dataptr approach here bc
  // vector<bool> can be optimized as vector<bit> and storage details
  // are decoupled from actual size of 'bool' type
  hash_t h(static_cast<uint64_t>(0xad2ed1983bbf2e28));
  static const hash_t h_true(static_cast<uint64_t>(0x74f6b5198daa2b2));
  static const hash_t h_false(static_cast<uint64_t>(0xe39f30789cab5382));
  for (const auto& b : values) {
    if (b) {
      h = HashCombine(h, h_true);
    } else {
      h = HashCombine(h, h_false);
    }
  }
  return h;
}

} // namespace lazy
} // namespace torch
