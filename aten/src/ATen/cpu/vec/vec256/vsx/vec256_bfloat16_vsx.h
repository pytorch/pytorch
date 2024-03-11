#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_bfloat16_float(
    const Vectorized<BFloat16>& a) {
  constexpr int64_t K = Vectorized<BFloat16>::size();
  __at_align__ float arr[K];
  __at_align__ BFloat16 arr2[K];
  a.store(arr2);
  convert(arr2, arr, K);
  return std::make_tuple(
      Vectorized<float>::loadu(arr),
      Vectorized<float>::loadu(arr + Vectorized<float>::size()));
}

inline Vectorized<BFloat16> convert_float_bfloat16(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  constexpr int64_t K = Vectorized<BFloat16>::size();
  __at_align__ float arr[K];
  __at_align__ BFloat16 arr2[K];
  a.store(arr);
  b.store(arr + Vectorized<float>::size());
  convert(arr, arr2, K);
  return Vectorized<BFloat16>::loadu(arr2);
}

inline void load_fp32_from_bf16(const c10::BFloat16* data, Vectorized<float>& out) {
  __at_align__ float values[Vectorized<float>::size()];
  for (const auto k : c10::irange(Vectorized<float>::size())) {
    values[k] = data[k];
  }
  out = Vectorized<float>::loadu(values);
}

inline void load_fp32_from_bf16(
    const c10::BFloat16* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  load_fp32_from_bf16(data, out1);
  data += Vectorized<float>::size();
  load_fp32_from_bf16(data, out2);
}

} // namespace
} // namespace vec
} // namespace at
