#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>

// Note: header order is important here
#include <ATen/cpu/vec/vec256/vsx/vec256_double_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_float_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_int16_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_int32_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_int64_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_qint32_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_qint8_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_quint8_vsx.h>

#include <ATen/cpu/vec/vec256/vsx/vec256_complex_float_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_complex_double_vsx.h>

#include <ATen/cpu/vec/vec256/vsx/vec256_bfloat16_vsx.h>

namespace at {
namespace vec {

inline namespace CPU_CAPABILITY {

DEFINE_CLAMP_FUNCS(c10::quint8)
DEFINE_CLAMP_FUNCS(c10::qint8)
DEFINE_CLAMP_FUNCS(c10::qint32)
DEFINE_CLAMP_FUNCS(int16_t)
DEFINE_CLAMP_FUNCS(int32_t)
DEFINE_CLAMP_FUNCS(int64_t)
DEFINE_CLAMP_FUNCS(float)
DEFINE_CLAMP_FUNCS(double)

template <>
Vectorized<double> C10_ALWAYS_INLINE fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return Vectorized<double>{
      vec_madd(a.vec0(), b.vec0(), c.vec0()),
      vec_madd(a.vec1(), b.vec1(), c.vec1())};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b,
    const Vectorized<int64_t>& c) {
  return Vectorized<int64_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}
template <>
Vectorized<int32_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b,
    const Vectorized<int32_t>& c) {
  return Vectorized<int32_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}
template <>
Vectorized<int16_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b,
    const Vectorized<int16_t>& c) {
  return Vectorized<int16_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(float)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(double)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int64_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int32_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int16_t)

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<double>(const Vectorized<double>& src) {
  return Vectorized<int64_t>{vec_signed(src.vec0()), vec_signed(src.vec1())};
}

template <>
Vectorized<int32_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<float>(
    const Vectorized<float>& src) {
  return Vectorized<int32_t>{vec_signed(src.vec0()), vec_signed(src.vec1())};
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  // int32_t and float have same size
  int64_t i;
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    const int32_t* src_a = src + i;
    float* dst_a = dst + i;
    vint32 input_vec0 = vec_vsx_ld(offset0, reinterpret_cast<const vint32*>(src_a));
    vint32 input_vec1 =
        vec_vsx_ld(offset16, reinterpret_cast<const vint32*>(src_a));
    vfloat32 c0 = vec_float(input_vec0);
    vfloat32 c1 = vec_float(input_vec1);
    vec_vsx_st(c0, offset0, dst_a);
    vec_vsx_st(c1, offset16, dst_a);
  }

  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
inline void convert(const int64_t* src, double* dst, int64_t n) {
  int64_t i;
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    const int64_t* src_a = src + i;
    double* dst_a = dst + i;
    vint64 input_vec0 =
        vec_vsx_ld(offset0, reinterpret_cast<const vint64*>(src_a));
    vint64 input_vec1 =
        vec_vsx_ld(offset16, reinterpret_cast<const vint64*>(src_a));
    vfloat64 c0 = vec_double(input_vec0);
    vfloat64 c1 = vec_double(input_vec1);
    vec_vsx_st(c0, offset0, reinterpret_cast<double*>(dst_a));
    vec_vsx_st(c1, offset16, reinterpret_cast<double*>(dst_a));
  }
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);
  }
}
//Generic implementation to fix compiler error
//TO-DO : Add optimized version for ppc64
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(
    const Vectorized<Half>& a) {
  constexpr int64_t K = Vectorized<Half>::size();
  __at_align__ float arr[K];
  __at_align__ Half arr2[K];
  a.store(arr2);
  convert(arr2, arr, K);
  return std::make_tuple(
       Vectorized<float>::loadu(arr),
       Vectorized<float>::loadu(arr + Vectorized<float>::size()));
}

inline Vectorized<Half> convert_float_half(
    const Vectorized<float>& a, const Vectorized<float>& b) {
  constexpr int64_t K = Vectorized<Half>::size();
  __at_align__ float arr[K];
  __at_align__ Half arr2[K];
  a.store(arr);
  b.store(arr + Vectorized<float>::size());
  convert(arr, arr2, K);
  return Vectorized<Half>::loadu(arr2);
};

template <>
std::pair<Vectorized<double>, Vectorized<double>> inline interleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  // inputs:
  //   a      = {a0, a1, a2, a3}
  //   b      = {b0, b1, b2, b3}

  vfloat64 ab00 = vec_xxpermdi(a.vec0(), b.vec0(), 0);
  vfloat64 ab11 = vec_xxpermdi(a.vec0(), b.vec0(), 3);
  vfloat64 ab2_00 = vec_xxpermdi(a.vec1(), b.vec1(), 0);
  vfloat64 ab2_11 = vec_xxpermdi(a.vec1(), b.vec1(), 3);
  //   return {a0, b0, a1, b1}
  //          {a2, b2, a3, b3}
  return std::make_pair(
      Vectorized<double>{ab00, ab11}, Vectorized<double>{ab2_00, ab2_11});
}

template <>
std::pair<Vectorized<double>, Vectorized<double>> inline deinterleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}
  vfloat64 aa01 = vec_xxpermdi(a.vec0(), a.vec1(), 0);
  vfloat64 aa23 = vec_xxpermdi(b.vec0(), b.vec1(), 0);

  vfloat64 bb_01 = vec_xxpermdi(a.vec0(), a.vec1(), 3);
  vfloat64 bb_23 = vec_xxpermdi(b.vec0(), b.vec1(), 3);

  // swap lanes:
  //   return {a0, a1, a2, a3}
  //          {b0, b1, b2, b3}
  return std::make_pair(
      Vectorized<double>{aa01, aa23}, Vectorized<double>{bb_01, bb_23});
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline interleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3,, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3,, b4, b5, b6, b7}

  vfloat32 ab0011 = vec_mergeh(a.vec0(), b.vec0());
  vfloat32 ab2233 = vec_mergel(a.vec0(), b.vec0());

  vfloat32 ab2_0011 = vec_mergeh(a.vec1(), b.vec1());
  vfloat32 ab2_2233 = vec_mergel(a.vec1(), b.vec1());
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1,, a2, b2, a3, b3}
  //          {a4, b4, a5, b5,, a6, b6, a7, b7}

  return std::make_pair(
      Vectorized<float>{ab0011, ab2233}, Vectorized<float>{ab2_0011, ab2_2233});
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline deinterleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1,, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5,, a6, b6, a7, b7}

  // {a0,a2,b0,b2} {a1,a3,b1,b3}
  vfloat32 a0a2b0b2 = vec_mergeh(a.vec0(), a.vec1());
  vfloat32 a1a3b1b3 = vec_mergel(a.vec0(), a.vec1());

  vfloat32 aa0123 = vec_mergeh(a0a2b0b2, a1a3b1b3);
  vfloat32 bb0123 = vec_mergel(a0a2b0b2, a1a3b1b3);

  vfloat32 a0a2b0b2_2 = vec_mergeh(b.vec0(), b.vec1());
  vfloat32 a1a3b1b3_2 = vec_mergel(b.vec0(), b.vec1());

  vfloat32 aa0123_2 = vec_mergeh(a0a2b0b2_2, a1a3b1b3_2);
  vfloat32 bb0123_2 = vec_mergel(a0a2b0b2_2, a1a3b1b3_2);

  // it could be done with vec_perm ,too
  // swap lanes:
  //   return {a0, a1, a2, a3,, a4, a5, a6, a7}
  //          {b0, b1, b2, b3,, b4, b5, b6, b7}

  return std::make_pair(
      Vectorized<float>{aa0123, aa0123_2}, Vectorized<float>{bb0123, bb0123_2});
}

} // namespace
} // namespace vec
} // namespace at
