#pragma once

#if defined(CPU_CAPABILITY_AVX512)
#include <ATen/cpu/vec/vec512/vec512.h>
#else
#include <ATen/cpu/vec/vec256/vec256.h>
#endif

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

inline Vectorized<bool> convert_to_bool(Vectorized<int8_t> x) {
  __at_align__ bool buffer[x.size()];
  x.ne(Vectorized<int8_t>(0)).store(buffer);

  Vectorized<bool> ret;
  static_assert(x.size() == ret.size(), "");
  std::memcpy(ret, buffer, ret.size() * sizeof(bool));
  return ret;
}

template <>
inline Vectorized<bool> Vectorized<bool>::loadu(const void* ptr) {
  // See NOTE [Loading boolean values]
  return convert_to_bool(Vectorized<int8_t>::loadu(ptr));
}

template <>
inline Vectorized<bool> Vectorized<bool>::loadu(const void* ptr, int64_t count) {
  // See NOTE [Loading boolean values]
  return convert_to_bool(Vectorized<int8_t>::loadu(ptr, count));
}

}}} // namespace at::vec::CPU_CAPABILITY
