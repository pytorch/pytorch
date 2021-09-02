#include "lazy_tensors/computation_client/util.h"

#include <sstream>

namespace lazy_tensors {
namespace util {
namespace {

hash_t LoadHash(const uint8** data, const uint8* top) {
  std::ptrdiff_t size = top - (*data);
  if (size >= sizeof(hash_t)) {
    hash_t v;
    std::memcpy(&v, *data, sizeof(v));
    *data += sizeof(hash_t);
    return v;
  }

  union {
    hash_t h;
    uint8 b[sizeof(hash_t)];
  } uval;
  uval.h = 0;
  std::memcpy(uval.b, *data, size);
  *data += size;
  return uval.h;
}

}  // namespace


hash_t HashBlock(const void* data, size_t n, const hash_t& seed) {
  const hash_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  const uint8* u8_data = reinterpret_cast<const uint8*>(data);
  const uint8* top = u8_data + n;
  hash_t h = seed ^ (n * m);
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
  return HashBlock(data, size, 0xc2b2ae3d27d4eb4f);
}

size_t StdDataHash(const void* data, size_t size) {
  return HashReduce(DataHash(data, size));
}

size_t StdHashCombine(uintmax_t a, uintmax_t b) {
  return a ^
         (b * 0x27d4eb2f165667c5 + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}

hash_t HashCombine(const hash_t& a, const hash_t& b) {
  static const hash_t kb = lazy_tensors::MakeUint128(101, 0x27d4eb2f165667c5);
  return a ^ (b * kb + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}

size_t HashReduce(const hash_t& a) {
  return StdHashCombine(lazy_tensors::Uint128Low64(a),
                        lazy_tensors::Uint128High64(a));
}

std::string HexHash(const hash_t& a) {
  std::stringstream ss;
  ss << std::hex << lazy_tensors::Uint128High64(a) << std::setfill('0')
     << std::setw(16) << lazy_tensors::Uint128Low64(a);
  return ss.str();
}

}  // namespace util
}  // namespace lazy_tensors
