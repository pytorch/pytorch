#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <ATen/cpu/vec256/vsx/vsx_helpers.h>
#include <ATen/cpu/vec256/vsx/vec256_double_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_float_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_int16_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_int32_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_int64_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_qint32_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_qint8_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_quint8_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_complex_float_vsx.h>
#include <ATen/cpu/vec256/vsx/vec256_complex_double_vsx.h>
namespace at {
namespace vec256 {

namespace {

DEFINE_CLAMP_FUNCS(c10::quint8)
DEFINE_CLAMP_FUNCS(c10::qint8)
DEFINE_CLAMP_FUNCS(c10::qint32)
DEFINE_CLAMP_FUNCS(int16_t)
DEFINE_CLAMP_FUNCS(int32_t)
DEFINE_CLAMP_FUNCS(int64_t)
DEFINE_CLAMP_FUNCS(float)
DEFINE_CLAMP_FUNCS(double)

template <>
Vec256<double> C10_ALWAYS_INLINE fmadd(
    const Vec256<double>& a,
    const Vec256<double>& b,
    const Vec256<double>& c) {
  return Vec256<double>{
      vec_madd(a.vec0(), b.vec0(), c.vec0()),
      vec_madd(a.vec1(), b.vec1(), c.vec1())};
}

template <>
Vec256<int64_t> C10_ALWAYS_INLINE fmadd(
    const Vec256<int64_t>& a,
    const Vec256<int64_t>& b,
    const Vec256<int64_t>& c) {
  return Vec256<int64_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}
template <>
Vec256<int32_t> C10_ALWAYS_INLINE fmadd(
    const Vec256<int32_t>& a,
    const Vec256<int32_t>& b,
    const Vec256<int32_t>& c) {
  return Vec256<int32_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}
template <>
Vec256<int16_t> C10_ALWAYS_INLINE fmadd(
    const Vec256<int16_t>& a,
    const Vec256<int16_t>& b,
    const Vec256<int16_t>& c) {
  return Vec256<int16_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(float)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(double)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int64_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int32_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int16_t)

template <>
Vec256<int64_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<double>(const Vec256<double>& src) {
  return Vec256<int64_t>{vec_signed(src.vec0()), vec_signed(src.vec1())};
}

template <>
Vec256<int32_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<float>(
    const Vec256<float>& src) {
  return Vec256<int32_t>{vec_signed(src.vec0()), vec_signed(src.vec1())};
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  // int32_t and float have same size
  int64_t i;
  for (i = 0; i <= (n - Vec256<float>::size()); i += Vec256<float>::size()) {
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
  for (i = 0; i <= (n - Vec256<double>::size()); i += Vec256<double>::size()) {
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

template <>
std::pair<Vec256<double>, Vec256<double>> inline interleave2<double>(
    const Vec256<double>& a,
    const Vec256<double>& b) {
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
      Vec256<double>{ab00, ab11}, Vec256<double>{ab2_00, ab2_11});
}

template <>
std::pair<Vec256<double>, Vec256<double>> inline deinterleave2<double>(
    const Vec256<double>& a,
    const Vec256<double>& b) {
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
      Vec256<double>{aa01, aa23}, Vec256<double>{bb_01, bb_23});
}

template <>
std::pair<Vec256<float>, Vec256<float>> inline interleave2<float>(
    const Vec256<float>& a,
    const Vec256<float>& b) {
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
      Vec256<float>{ab0011, ab2233}, Vec256<float>{ab2_0011, ab2_2233});
}

template <>
std::pair<Vec256<float>, Vec256<float>> inline deinterleave2<float>(
    const Vec256<float>& a,
    const Vec256<float>& b) {
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
      Vec256<float>{aa0123, aa0123_2}, Vec256<float>{bb0123, bb0123_2});
}

} // namespace
} // namespace vec256
} // namespace at
