#pragma once
#include <cstdint>
#include <c10/macros/Macros.h>
#include <ATen/cpu/vec256/intrinsics.h>

using vbool8   =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) char;
using vbool16  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) short;
using vbool32  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) int;
using vbool64  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) long long;
using vint8    =  __attribute__((altivec(vector__)))  signed char;
using vint16   =  __attribute__((altivec(vector__)))  signed short;
using vint32   =  __attribute__((altivec(vector__)))  signed int;
using vint64   =  __attribute__((altivec(vector__)))  signed long long;
using vuint8   =  __attribute__((altivec(vector__)))  unsigned char;
using vuint16  =  __attribute__((altivec(vector__)))  unsigned short;
using vuint32  =  __attribute__((altivec(vector__)))  unsigned  int;
using vuint64  =  __attribute__((altivec(vector__)))  unsigned long long;
using vfloat32 =  __attribute__((altivec(vector__)))  float;
using vfloat64 =  __attribute__((altivec(vector__)))  double;

#if !defined(vec_float)
C10_ALWAYS_INLINE vfloat32 vec_float(const vint32& vec_in) {
  vfloat32 vec_out;
  __asm__("xvcvsxwsp %x0,%x1" : "=wf"(vec_out) : "wa"(vec_in));
  return vec_out;
}
#endif

#if !defined(vec_signed)
C10_ALWAYS_INLINE vint32 vec_signed(const vfloat32& vec_in) {
  vint32 vec_out;
  __asm__("xvcvspsxws %x0,%x1" : "=wa"(vec_out) : "wf"(vec_in));
  return vec_out;
}

C10_ALWAYS_INLINE vint64 vec_signed(const vfloat64& vec_in) {
  vint64 vec_out;
  __asm__("xvcvdpsxds %x0,%x1" : "=wa"(vec_out) : "wd"(vec_in));
  return vec_out;
}
#endif

#if !defined(vec_neg)
C10_ALWAYS_INLINE vfloat32 vec_neg(const vfloat32& vec_in) {
  vfloat32 vec_out;
  __asm__("xvnegsp %x0,%x1" : "=wf"(vec_out) : "wf"(vec_in));
  return vec_out;
}

C10_ALWAYS_INLINE vfloat64 vec_neg(const vfloat64& vec_in) {
  vfloat64 vec_out;
  __asm__("xvnegdp %x0,%x1" : "=wd"(vec_out) : "wd"(vec_in));
  return vec_out;
}

C10_ALWAYS_INLINE vint16 vec_neg(const vint16& vec_in) {
  vint16 vint0 = {0, 0, 0, 0 ,0, 0, 0, 0};
  return vec_vsubuhm(vint0, vec_in);
}

C10_ALWAYS_INLINE vint32 vec_neg(const vint32& vec_in) {
  vint32 vint0 = {0, 0, 0, 0};
  return vec_vsubuwm(vint0, vec_in);
}

C10_ALWAYS_INLINE vint64 vec_neg(const vint64& vec_in) {
  vint64 vint0 = {0, 0};
  return vec_vsubudm(vint0, vec_in);
}
#endif

#if !defined(vec_sldw)
template <unsigned int C>
C10_ALWAYS_INLINE vfloat32
vec_sldw_aux(const vfloat32& vec_in0, const vfloat32& vec_in1) {
  vfloat32 vec_out;
  __asm("xxsldwi %x0, %x1, %x2, %3 "
        : "=wa"(vec_out)
        : "wa"(vec_in0), "wa"(vec_in1), "I"(C));
  return vec_out;
}

#define vec_sldw(a, b, c) vec_sldw_aux<c>(a, b)
#endif

#define vec_not(a) vec_nor(a, a)

#define DEFINE_MEMBER_UNARY_OP(op, op_type, func)     \
  Vec256<op_type> C10_ALWAYS_INLINE op() const {      \
    return Vec256<op_type>{func(_vec0), func(_vec1)}; \
  }

#define DEFINE_MEMBER_OP(op, op_type, func)                                  \
  Vec256<op_type> C10_ALWAYS_INLINE op(const Vec256<op_type>& other) const { \
    return Vec256<op_type>{                                                  \
        func(_vec0, other._vec0), func(_vec1, other._vec1)};                 \
  }

#define DEFINE_MEMBER_BITWISE_OP(op, op_type, func)                          \
  Vec256<op_type> C10_ALWAYS_INLINE op(const Vec256<op_type>& other) const { \
    return Vec256<op_type>{                                                  \
        func(_vecb0, other._vecb0), func(_vecb1, other._vecb1)};             \
  }

#define DEFINE_MEMBER_TERNARY_OP(op, op_type, func)                    \
  Vec256<op_type> C10_ALWAYS_INLINE op(                                \
      const Vec256<op_type>& b, const Vec256<op_type>& c) const {      \
    return Vec256<op_type>{                                            \
        func(_vec0, b._vec0, c._vec0), func(_vec1, b._vec1, c._vec1)}; \
  }

#define DEFINE_MEMBER_EMULATE_BINARY_OP(op, op_type, binary_op)          \
  Vec256<op_type> C10_ALWAYS_INLINE op(const Vec256<op_type>& b) const { \
    Vec256<op_type>::vec_internal_type ret_0;                         \
    Vec256<op_type>::vec_internal_type ret_1;                         \
    for (int i = 0; i < Vec256<op_type>::size() / 2; i++) {           \
      ret_0[i] = _vec0[i] binary_op b._vec0[i];                       \
      ret_1[i] = _vec1[i] binary_op b._vec1[i];                       \
    }                                                                 \
    return Vec256<op_type>{ret_0, ret_1};                             \
  }


#define DEFINE_MEMBER_OP_AND_ONE(op, op_type, func)                          \
  Vec256<op_type> C10_ALWAYS_INLINE op(const Vec256<op_type>& other) const { \
    using vvtype = Vec256<op_type>::vec_internal_type;                       \
    const vvtype v_one = vec_splats(static_cast<op_type>(1.0));              \
    vvtype ret0 = (vvtype)func(_vec0, other._vec0);                          \
    vvtype ret1 = (vvtype)func(_vec1, other._vec1);                          \
    return Vec256<op_type>{vec_and(ret0, v_one), vec_and(ret1, v_one)};      \
  }

#define DEFINE_CLAMP_FUNCS(operand_type)                                \
  template <>                                                           \
  Vec256<operand_type> C10_ALWAYS_INLINE clamp(                         \
      const Vec256<operand_type>& a,                                    \
      const Vec256<operand_type>& min,                                  \
      const Vec256<operand_type>& max) {                                \
    return Vec256<operand_type>{                                        \
        vec_min(max.vec0(), vec_max(a.vec0(), min.vec0())),             \
        vec_min(max.vec1(), vec_max(a.vec1(), min.vec1()))};            \
  }                                                                     \
  template <>                                                           \
  Vec256<operand_type> C10_ALWAYS_INLINE clamp_min(                     \
      const Vec256<operand_type>& a, const Vec256<operand_type>& min) { \
    return Vec256<operand_type>{                                        \
        vec_max(a.vec0(), min.vec0()), vec_max(a.vec1(), min.vec1())};  \
  }                                                                     \
  template <>                                                           \
  Vec256<operand_type> C10_ALWAYS_INLINE clamp_max(                     \
      const Vec256<operand_type>& a, const Vec256<operand_type>& max) { \
    return Vec256<operand_type>{                                        \
        vec_min(a.vec0(), max.vec0()), vec_min(a.vec1(), max.vec1())};  \
  }

#define DEFINE_REINTERPRET_CAST_FUNCS(                             \
    first_type, cast_type, cast_inner_vector_type)                 \
  template <>                                                      \
  C10_ALWAYS_INLINE Vec256<cast_type> cast<cast_type, first_type>( \
      const Vec256<first_type>& src) {                                 \
    return Vec256<cast_type>{(cast_inner_vector_type)src.vec0(),       \
                             (cast_inner_vector_type)src.vec1()};      \
  }

#define DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(first_type)     \
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, double, vfloat64)    \
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, float, vfloat32)     \
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, int64_t, vint64) \
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, int32_t, vint32)   \
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, int16_t, vint16)

// it can be used to emulate blend faster
constexpr int blendChoice(uint32_t mask, uint32_t half1 = 0xF, uint32_t half2 = 0xF0) {
  uint32_t none = 0;
  uint32_t both = half1 | half2;
  // clamp it between 0 and both
  mask = mask & both;
  // return  (a._vec0, a._vec1)
  if (mask == none) return 0;
  // return (b._vec0,b._vec1)
  else if (mask == both)
    return 1;
  // return  (b._vec0,a._vec1)
  else if (mask == half1)
    return 2;
  // return  (a._vec0,b._vec1)
  else if (mask == half2)
    return 3;
  // return  (*_vec0,a._vec1)
  else if (mask > 0 && mask < half1)
    return 4;
  // return  (*_vec0,b._vec1)
  else if ((mask & half2) == half2)
    return 5;
  // return (a._vec0,*_vec1)
  else if ((mask & half1) == 0 && mask > half1)
    return 6;
  // return (b._vec0,*_vec1)
  else if ((mask & half1) == half1 && mask > half1)
    return 7;
  // return (*_vec0,*_vec1)
  return 8;
}

// it can be used to emulate blend faster
constexpr int blendChoiceDbl(uint32_t mask) {
  // clamp it 0 and 0xF
  return blendChoice(mask, 0x3, 0xC);
}

constexpr vbool32 VsxMask1(uint32_t mask) {
  uint32_t g0 = (mask & 1) * 0xffffffff;
  uint32_t g1 = ((mask & 2) >> 1) * 0xffffffff;
  uint32_t g2 = ((mask & 4) >> 2) * 0xffffffff;
  uint32_t g3 = ((mask & 8) >> 3) * 0xffffffff;
  return (vbool32){g0, g1, g2, g3};
}

constexpr vbool32 VsxMask2(uint32_t mask) {
  uint32_t mask2 = (mask & 0xFF) >> 4;
  return VsxMask1(mask2);
}

constexpr vbool64 VsxDblMask1(uint32_t mask) {
  uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
  uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
  return (vbool64){g0, g1};
}

constexpr vbool64 VsxDblMask2(uint32_t mask) {
  uint32_t mask2 = (mask & 0xF) >> 2;
  return VsxDblMask1(mask2);
}

constexpr int maskForComplex(uint32_t mask) {
  mask = mask & 0xF;
  int complex_mask = 0;
  if (mask & 1) complex_mask |= 3;
  if (mask & 2) complex_mask |= (3 << 2);
  if (mask & 4) complex_mask |= (3 << 4);
  if (mask & 8) complex_mask |= (3 << 6);
  return complex_mask;
}

constexpr int maskForComplexDbl(uint32_t mask) {
  mask = mask & 0x3;
  int complex_mask = 0;
  if (mask & 1) complex_mask |= 3;
  if (mask & 2) complex_mask |= (3 << 2);
  return complex_mask;
}

constexpr int blendChoiceComplex(uint32_t mask) {
  return blendChoice(maskForComplex(mask));
}

constexpr int blendChoiceComplexDbl(uint32_t mask) {
  return blendChoiceDbl(maskForComplexDbl(mask));
}

constexpr vbool32 VsxComplexMask1(uint32_t mask) {
  return VsxMask1(maskForComplex(mask));
}

constexpr vbool32 VsxComplexMask2(uint32_t mask) {
  uint32_t mask2 = (mask & 0xF) >> 2;
  return VsxMask1(maskForComplex(mask2));
}

constexpr vbool64 VsxComplexDblMask1(uint32_t mask) { return VsxDblMask1(mask); }

constexpr vbool64 VsxComplexDblMask2(uint32_t mask) {
  uint32_t mask2 = (mask & 0xF) >> 2;
  return VsxDblMask1(mask2);
}

// constants
namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {
//
constexpr int offset0 = 0;
constexpr int offset16 = 16;

// #Constants
const vuint8 mask_zero_bits = vuint8{128, 128, 128, 128, 128, 128, 128, 128,
                                128, 128, 128, 128, 96,  64,  32,  0};

const vuint8 swap_mask =
    vuint8{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};

const vint32 v0x7f = vec_splats(0x7f);
const vint32 vi_0 = vec_splats((int)(0));
const vint32 vi_1 = vec_splats((int)1);
const vint32 vi_2 = vec_splats((int)2);
const vint32 vi_4 = vec_splats((int)4);
const vint32 vi_inv1 = vec_splats((int)~1);
const vuint32 vu_29 = vec_splats(29u);
const vuint32 vu_23 = vec_splats(23u);

const vbool32 inv_mant_mask = (vbool32)vec_splats((unsigned int)~0xff800000);
const vbool32 sign_mask = (vbool32)vec_splats((int)0x80000000);
const vbool32 real_mask = vbool32{0xFFFFFFFF, 0x0, 0xFFFFFFFF, 0x0};
const vbool32 imag_mask = vbool32{0x0, 0xFFFFFFFF, 0x0, 0xFFFFFFFF};
const vbool32 isign_mask = vbool32{0x0, 0x80000000, 0x0, 0x80000000};
const vbool32 rsign_mask = vbool32{0x80000000, 0x0, 0x80000000, 0x0};

const vbool64 vd_imag_mask  = vbool64{0x0, 0xFFFFFFFFFFFFFFFF};
const vbool64 vd_real_mask  = vbool64{0xFFFFFFFFFFFFFFFF, 0x0};
const vbool64 vd_isign_mask = vbool64{0x0, 0x8000000000000000};
const vbool64 vd_rsign_mask = vbool64{0x8000000000000000, 0x0};

const vfloat32 zero = vec_splats(0.f);
const vfloat32 half = vec_splats(0.5f);
const vfloat32 one = vec_splats(1.f);
const vfloat32 two = vec_splats(2.0f);
const vfloat32 _4div_pi = vec_splats(1.27323954473516f);
const vfloat32 v_inf = (vfloat32)vec_splats(0x7f800000u);
const vfloat32 v_minus_inf = vfloat32{ 0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u };
const vfloat32 v_nan = (vfloat32)vec_splats(0x7fffffff);
const vfloat32 log10e_inv = vec_splats(0.43429448190325176f);
const vfloat32 log2e_inv = vec_splats(1.4426950408889634f);
const vfloat32 log2eB_inv = vec_splats(1.442695036924675f);
const vfloat32 cephes_SQRTHF = vec_splats(0.707106781186547524f);
const vfloat32 coscof_p0 = vec_splats(2.443315711809948E-005f);
const vfloat32 coscof_p1 = vec_splats(-1.388731625493765E-003f);
const vfloat32 coscof_p2 = vec_splats(4.166664568298827E-002f);
const vfloat32 exp_hi = vec_splats(104.f);
const vfloat32 exp_lo = vec_splats(-104.f);
const vfloat32 exp_p0 = vec_splats(0.000198527617612853646278381f);
const vfloat32 exp_p1 = vec_splats((0.00139304355252534151077271f));
const vfloat32 exp_p2 = vec_splats(0.00833336077630519866943359f);
const vfloat32 exp_p3 = vec_splats(0.0416664853692054748535156f);
const vfloat32 exp_p4 = vec_splats(0.166666671633720397949219f);
const vfloat32 exp_p5 = vec_splats(0.5f);
const vfloat32 log_p0 = vec_splats(7.0376836292E-2f);
const vfloat32 log_p1 = vec_splats(-1.1514610310E-1f);
const vfloat32 log_p2 = vec_splats(1.1676998740E-1f);
const vfloat32 log_p3 = vec_splats(-1.2420140846E-1f);
const vfloat32 log_p4 = vec_splats(+1.4249322787E-1f);
const vfloat32 log_p5 = vec_splats(-1.6668057665E-1f);
const vfloat32 log_p6 = vec_splats(+2.0000714765E-1f);
const vfloat32 log_p7 = vec_splats(-2.4999993993E-1f);
const vfloat32 log_p8 = vec_splats(+3.3333331174E-1f);
const vfloat32 log_q1 = vec_splats(-2.12194440e-4f);
const vfloat32 log_q2 = vec_splats(0.693359375f);
const vfloat32 max_logf = vec_splats(88.02969187150841f);
const vfloat32 max_numf = vec_splats(1.7014117331926442990585209174225846272e38f);
const vfloat32 min_inf = (vfloat32)vec_splats(0xff800000u);
const vfloat32 min_norm_pos = (vfloat32)vec_splats(0x0800000u);
const vfloat32 minus_cephes_dp1 = vec_splats(-0.78515625f);
const vfloat32 minus_cephes_dp2 = vec_splats(-2.4187564849853515625e-4f);
const vfloat32 minus_cephes_dp3 = vec_splats(-3.77489497744594108e-8f);
const vfloat32 negln2f_hi = vec_splats(-0.693145751953125f);
const vfloat32 negln2f_lo = vec_splats(-1.428606765330187045e-06f);
const vfloat32 p0 = vec_splats(2.03721912945E-4f);
const vfloat32 p1 = vec_splats(8.33028376239E-3f);
const vfloat32 p2 = vec_splats(1.66667160211E-1f);
const vfloat32 sincof_p0 = vec_splats(-1.9515295891E-4f);
const vfloat32 sincof_p1 = vec_splats(8.3321608736E-3f);
const vfloat32 sincof_p2 = vec_splats(-1.6666654611E-1f);
const vfloat32 tanh_0p625 = vec_splats(0.625f);
const vfloat32 tanh_half_max = vec_splats(44.014845935754205f);
const vfloat32 tanh_p0 = vec_splats(-5.70498872745E-3f);
const vfloat32 tanh_p1 = vec_splats(2.06390887954E-2f);
const vfloat32 tanh_p2 = vec_splats(-5.37397155531E-2f);
const vfloat32 tanh_p3 = vec_splats(1.33314422036E-1f);
const vfloat32 tanh_p4 = vec_splats(-3.33332819422E-1f);
const vfloat32 vcheck = vec_splats((float)(1LL << 24));
const vfloat32 imag_one = vfloat32{0.f, 1.f, 0.f, 1.f};
const vfloat32 imag_half = vfloat32{0.f, 0.5f, 0.f, 0.5f};
const vfloat32 sqrt2_2 = vfloat32{0.70710676908493042f, 0.70710676908493042,
                          0.70710676908493042, 0.70710676908493042};
const vfloat32 pi_2 = vfloat32{M_PI / 2, 0.0, M_PI / 2, 0.0};
const vfloat32 vf_89 = vfloat32{89.f, 89.f, 89.f, 89.f};
const vfloat64 vd_one = vec_splats(1.0);
const vfloat64 vd_zero = vec_splats(0.0);
const vfloat64 vd_log10e_inv = vec_splats(0.43429448190325176);
const vfloat64 vd_log2e_inv = vec_splats(1.4426950408889634);
const vfloat64 vd_imag_one = vfloat64{0.0, 1.0};
const vfloat64 vd_imag_half = vfloat64{0.0, 0.5};
const vfloat64 vd_sqrt2_2 = vfloat64{0.70710678118654757, 0.70710678118654757};
const vfloat64 vd_pi_2 = vfloat64{M_PI / 2.0, 0.0};

} // namespace
} // namespace vec256
} // namespace at
