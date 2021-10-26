#include "lazy_tensors/core/platform/hash.h"

#include <cstring>

#include "lazy_tensors/core/platform/macros.h"

namespace lazy_tensors {
namespace {

// 0xff is in case char is signed.
inline uint64_t ByteAs64(char c) { return static_cast<uint64_t>(c) & 0xff; }

uint64_t DecodeFixed64(const char* ptr) {
  // Load the raw bytes
  uint64_t result;
  memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
  return result;
}

}  // namespace

// Same as:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hash.cc

uint64_t Hash64(const char* data, size_t n, uint64_t seed) {
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64_t h = seed ^ (n * m);

  while (n >= 8) {
    uint64_t k = DecodeFixed64(data);
    data += 8;
    n -= 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
      TF_FALLTHROUGH_INTENDED;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
      TF_FALLTHROUGH_INTENDED;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
      TF_FALLTHROUGH_INTENDED;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
      TF_FALLTHROUGH_INTENDED;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
      TF_FALLTHROUGH_INTENDED;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
      TF_FALLTHROUGH_INTENDED;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

}  // namespace lazy_tensors
