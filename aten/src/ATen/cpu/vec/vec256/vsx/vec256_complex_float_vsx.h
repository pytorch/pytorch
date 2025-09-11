
#pragma once
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/complex.h>
#include <c10/util/irange.h>

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {
using ComplexFlt = c10::complex<float>;

template <>
struct is_vec_specialized_for<ComplexFlt> : std::bool_constant<true> {};

template <>
class Vectorized<ComplexFlt> {
 private:
  union {
    struct {
      vfloat32 _vec0;
      vfloat32 _vec1;
    };
    struct {
      vbool32 _vecb0;
      vbool32 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  using value_type = ComplexFlt;
  using vec_internal_type = vfloat32;
  using vec_internal_mask_type = vbool32;
  using size_type = int;

  static constexpr size_type size() {
    return 4;
  }
  Vectorized() {}

  C10_ALWAYS_INLINE Vectorized(vfloat32 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(vbool32 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorized(vfloat32 v1, vfloat32 v2)
      : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(vbool32 v1, vbool32 v2)
      : _vecb0{v1}, _vecb1{v2} {}

  Vectorized(ComplexFlt val) {
    float real_value = val.real();
    float imag_value = val.imag();
    _vec0 = vfloat32{real_value, imag_value, real_value, imag_value};
    _vec1 = vfloat32{real_value, imag_value, real_value, imag_value};
  }

  Vectorized(
      ComplexFlt val1,
      ComplexFlt val2,
      ComplexFlt val3,
      ComplexFlt val4) {
    _vec0 = vfloat32{val1.real(), val1.imag(), val2.real(), val2.imag()};
    _vec1 = vfloat32{val3.real(), val3.imag(), val4.real(), val4.imag()};
  }

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 0, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    return a;
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 1, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    return b;
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 2, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    return {b._vec0, a._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 3, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    return {a._vec0, b._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 4, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    const vbool32 mask_1st = VsxComplexMask1(mask);
    return {(vfloat32)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 5, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    const vbool32 mask_1st = VsxComplexMask1(mask);
    return {(vfloat32)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1};
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 6, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    const vbool32 mask_2nd = VsxComplexMask2(mask);
    // generated masks
    return {a._vec0, (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 7, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    const vbool32 mask_2nd = VsxComplexMask2(mask);
    // generated masks
    return {b._vec0, (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 8, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    const vbool32 mask_1st = VsxComplexMask1(mask);
    const vbool32 mask_2nd = VsxComplexMask2(mask);
    return {
        (vfloat32)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int64_t mask>
  static Vectorized<ComplexFlt> C10_ALWAYS_INLINE
  el_blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    const vbool32 mask_1st = VsxMask1(mask);
    const vbool32 mask_2nd = VsxMask2(mask);
    return {
        (vfloat32)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  static Vectorized<ComplexFlt> blendv(
      const Vectorized<ComplexFlt>& a,
      const Vectorized<ComplexFlt>& b,
      const Vectorized<ComplexFlt>& mask) {
    // convert std::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_complex = Vectorized<ComplexFlt>(
        vec_mergeh(mask._vec0, mask._vec0), vec_mergeh(mask._vec1, mask._vec1));
    return {
        vec_sel(
            a._vec0, b._vec0, reinterpret_cast<vbool32>(mask_complex._vec0)),
        vec_sel(
            a._vec1, b._vec1, reinterpret_cast<vbool32>(mask_complex._vec1)),
    };
  }

  static Vectorized<ComplexFlt> elwise_blendv(
      const Vectorized<ComplexFlt>& a,
      const Vectorized<ComplexFlt>& b,
      const Vectorized<ComplexFlt>& mask) {
    return {
        vec_sel(a._vec0, b._vec0, reinterpret_cast<vbool32>(mask._vec0)),
        vec_sel(a._vec1, b._vec1, reinterpret_cast<vbool32>(mask._vec1)),
    };
  }

  template <typename step_t>
  static Vectorized<ComplexFlt> arange(
      ComplexFlt base = 0.,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<ComplexFlt>(
        base,
        base + step,
        base + ComplexFlt(2) * step,
        base + ComplexFlt(3) * step);
  }
  static Vectorized<ComplexFlt> set(
      const Vectorized<ComplexFlt>& a,
      const Vectorized<ComplexFlt>& b,
      int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
    }
    return b;
  }

  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const float*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const float*>(ptr))};
    }

    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    return {
        vec_vsx_ld(offset0, reinterpret_cast<const float*>(tmp_values)),
        vec_vsx_ld(offset16, reinterpret_cast<const float*>(tmp_values))};
  }

  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<float*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<float*>(ptr));
    } else if (count > 0) {
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, reinterpret_cast<float*>(tmp_values));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<float*>(tmp_values));
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }

  const ComplexFlt& operator[](int idx) const = delete;
  ComplexFlt& operator[](int idx) = delete;

  Vectorized<ComplexFlt> map(ComplexFlt (*const f)(ComplexFlt)) const {
    __at_align__ ComplexFlt tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  Vectorized<ComplexFlt> map(ComplexFlt (*const f)(const ComplexFlt&)) const {
    __at_align__ ComplexFlt tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  static Vectorized<ComplexFlt> horizontal_add(
      Vectorized<ComplexFlt>& first,
      Vectorized<ComplexFlt>& second) {
    // Operates on individual floats, see _mm_hadd_ps
    // {f0+f1, s0+s1, f2+f3, s2+s3, ...}
    // i.e. it sums the re and im of each value and interleaves first and
    // second: {f_re0 + f_im0, s_re0 + s_im0, f_re1 + f_im1, s_re1 + s_im1, ...}
    return el_mergee(first, second) + el_mergeo(first, second);
  }

  static Vectorized<ComplexFlt> horizontal_sub_permD8(
      Vectorized<ComplexFlt>& first,
      Vectorized<ComplexFlt>& second) {
    // we will simulate it differently with 6 instructions total
    // lets permute second so that we can add it getting horizontal sums
    auto first_perm = first.el_swapped(); // 2perm
    auto second_perm = second.el_swapped(); // 2perm
    // sum
    auto first_ret = first - first_perm; // 2sub
    auto second_ret = second - second_perm; // 2 sub
    // now lets choose evens
    return el_mergee(first_ret, second_ret); // 2 mergee's
  }

  Vectorized<ComplexFlt> abs_2_() const {
    auto a = (*this).elwise_mult(*this);
    auto permuted = a.el_swapped();
    a = a + permuted;
    return a.el_mergee();
  }

  Vectorized<ComplexFlt> abs_() const {
    auto vi = el_mergeo();
    auto vr = el_mergee();
    return {
        Sleef_hypotf4_u05vsx(vr._vec0, vi._vec0),
        Sleef_hypotf4_u05vsx(vr._vec1, vi._vec1)};
  }

  Vectorized<ComplexFlt> abs() const {
    return abs_() & real_mask;
  }

  Vectorized<ComplexFlt> real_() const {
    return *this & real_mask;
  }
  Vectorized<ComplexFlt> real() const {
    return *this & real_mask;
  }
  Vectorized<ComplexFlt> imag_() const {
    return *this & imag_mask;
  }
  Vectorized<ComplexFlt> imag() const {
    // we can use swap_mask or sldwi
    auto ret = imag_();
    return {
        vec_sldw(ret._vec0, ret._vec0, 3), vec_sldw(ret._vec1, ret._vec1, 3)};
  }

  Vectorized<ComplexFlt> conj_() const {
    return *this ^ isign_mask;
  }
  Vectorized<ComplexFlt> conj() const {
    return *this ^ isign_mask;
  }

  Vectorized<ComplexFlt> log() const {
    // Most trigonomic ops use the log() op to improve complex number
    // performance.
    return map(std::log);
  }

  Vectorized<ComplexFlt> log2() const {
    // log2eB_inv
    auto ret = log();
    return ret.elwise_mult(log2e_inv);
  }
  Vectorized<ComplexFlt> log10() const {
    auto ret = log();
    return ret.elwise_mult(log10e_inv);
  }

  Vectorized<ComplexFlt> log1p() const {
    return map(std::log1p);
  }

  Vectorized<ComplexFlt> el_swapped() const {
    vfloat32 v0 = vec_perm(_vec0, _vec0, swap_mask);
    vfloat32 v1 = vec_perm(_vec1, _vec1, swap_mask);
    return {v0, v1};
  }

  Vectorized<ComplexFlt> el_mergee() const {
    // as mergee phased in , we can use vec_perm with mask
    return {vec_mergee(_vecb0, _vecb0), vec_mergee(_vecb1, _vecb1)};
  }

  Vectorized<ComplexFlt> el_mergeo() const {
    // as mergeo phased in , we can use vec_perm with mask
    return {vec_mergeo(_vecb0, _vecb0), vec_mergeo(_vecb1, _vecb1)};
  }

  Vectorized<ComplexFlt> el_madd(
      const Vectorized<ComplexFlt>& multiplier,
      const Vectorized<ComplexFlt>& val) const {
    return {
        vec_madd(_vec0, multiplier._vec0, val._vec0),
        vec_madd(_vec1, multiplier._vec1, val._vec1)};
  }

  static Vectorized<ComplexFlt> el_mergee(
      const Vectorized<ComplexFlt>& first,
      const Vectorized<ComplexFlt>& second) {
    return {
        vec_mergee(first._vecb0, second._vecb0),
        vec_mergee(first._vecb1, second._vecb1)};
  }

  static Vectorized<ComplexFlt> el_mergeo(
      const Vectorized<ComplexFlt>& first,
      const Vectorized<ComplexFlt>& second) {
    return {
        vec_mergeo(first._vecb0, second._vecb0),
        vec_mergeo(first._vecb1, second._vecb1)};
  }

  Vectorized<ComplexFlt> angle_() const {
    // angle = atan2(b/a)
    // auto b_a = _mm256_permute_ps(values, 0xB1); // b        a
    // return Sleef_atan2f8_u10(values, b_a); // 90-angle angle
    Vectorized<ComplexFlt> ret;
    for (int i = 0; i < 4; i += 2) {
      ret._vec0[i] = std::atan2(_vec0[i + 1], _vec0[i]);
      ret._vec1[i] = std::atan2(_vec1[i + 1], _vec1[i]);
    }
    return ret;
  }

  Vectorized<ComplexFlt> angle() const {
    return angle_() & real_mask;
  }

  Vectorized<ComplexFlt> sin() const {
    return map(std::sin);
  }
  Vectorized<ComplexFlt> sinh() const {
    return map(std::sinh);
  }
  Vectorized<ComplexFlt> cos() const {
    return map(std::cos);
  }
  Vectorized<ComplexFlt> cosh() const {
    return map(std::cosh);
  }
  Vectorized<ComplexFlt> ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  Vectorized<ComplexFlt> floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  Vectorized<ComplexFlt> neg() const {
    auto z = Vectorized<ComplexFlt>(zero);
    return z - *this;
  }
  Vectorized<ComplexFlt> round() const {
    return {vec_round(_vec0), vec_round(_vec1)};
  }
  Vectorized<ComplexFlt> tan() const {
    return map(std::tan);
  }
  Vectorized<ComplexFlt> tanh() const {
    return map(std::tanh);
  }
  Vectorized<ComplexFlt> trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<ComplexFlt> elwise_sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }

  Vectorized<ComplexFlt> sqrt() const {
    return map(std::sqrt);
  }

  Vectorized<ComplexFlt> reciprocal() const {
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2() = c/abs_2()
    // im = (bc - ad)/abs_2() = d/abs_2()
    auto c_d = *this ^ isign_mask; // c       -d
    auto abs = abs_2_();
    return c_d.elwise_div(abs);
  }

  Vectorized<ComplexFlt> rsqrt() const {
    return sqrt().reciprocal();
  }

  Vectorized<ComplexFlt> pow(const Vectorized<ComplexFlt>& exp) const {
    __at_align__ ComplexFlt x_tmp[size()];
    __at_align__ ComplexFlt y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (const auto i : c10::irange(size())) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }

  Vectorized<ComplexFlt> atan() const {
    // atan(x) = i/2 * ln((i + z)/(i - z))
    auto ione = Vectorized(imag_one);
    auto sum = ione + *this;
    auto sub = ione - *this;
    auto ln = (sum / sub).log(); // ln((i + z)/(i - z))
    return ln * imag_half; // i/2*ln()
  }
  Vectorized<ComplexFlt> atanh() const {
    return map(std::atanh);
  }

  Vectorized<ComplexFlt> acos() const {
    // acos(x) = pi/2 - asin(x)
    return Vectorized(pi_2) - asin();
  }

  Vectorized<ComplexFlt> inline operator*(
      const Vectorized<ComplexFlt>& b) const {
    //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i

#if 1
    // this is more vsx friendly than simulating horizontal from x86

    auto vi = b.el_mergeo();
    auto vr = b.el_mergee();
    vi = vi ^ rsign_mask;
    auto ret = elwise_mult(vr);
    auto vx_swapped = el_swapped();
    ret = vx_swapped.elwise_mult(vi) + ret;
    return ret;

#else

    auto ac_bd = elwise_mult(b);
    auto d_c = b.el_swapped();
    d_c = d_c ^ isign_mask;
    auto ad_bc = elwise_mult(d_c);
    auto ret = horizontal_sub_permD8(ac_bd, ad_bc);
    return ret;
#endif
  }

  Vectorized<ComplexFlt> inline operator/(
      const Vectorized<ComplexFlt>& b) const {
#if 1
    __at_align__ c10::complex<float>
        tmp1[Vectorized<c10::complex<float>>::size()];
    __at_align__ c10::complex<float>
        tmp2[Vectorized<c10::complex<float>>::size()];
    __at_align__ c10::complex<float>
        out[Vectorized<c10::complex<float>>::size()];
    this->store(tmp1);
    b.store(tmp2);

    for (const auto i : c10::irange(Vectorized<c10::complex<float>>::size())) {
      out[i] = tmp1[i] / tmp2[i];
    }
    return loadu(out);
#else
    auto fabs_cd = Vectorized{
        vec_andc(b._vec0, sign_mask), vec_andc(b._vec1, sign_mask)}; // |c| |d|
    auto fabs_dc = fabs_cd.el_swapped(); // |d|            |c|
    auto scale = fabs_cd.elwise_max(fabs_dc); // sc = max(|c|, |d|)
    auto a2 = elwise_div(scale); // a/sc           b/sc
    auto b2 = b.elwise_div(scale); // c/sc           d/sc
    auto acbd2 = a2.elwise_mult(b2); // ac/sc^2        bd/s
    auto dc2 = b2.el_swapped(); // d/sc           c/sc
    dc2 = dc2 ^ rsign_mask; // -d/sc          c/sc
    auto adbc2 = a2.elwise_mult(dc2); // -ad/sc^2       bc/sc^2
    auto ret = horizontal_add(acbd2, adbc2); // (ac+bd)/sc^2   (bc-ad)/sc^2
    auto denom2 = b2.abs_2_(); // (c^2+d^2)/sc^2 (c^2+d^2)/sc^2
    ret = ret.elwise_div(denom2);
    return ret;
#endif
  }

  Vectorized<ComplexFlt> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))

#if 1
    auto conj = conj_();
    auto b_a = conj.el_swapped();
    auto ab = conj.elwise_mult(b_a);
    auto im = ab + ab;
    auto val_2 = (*this).elwise_mult(*this);
    auto val_2_swapped = val_2.el_swapped();
    auto re = horizontal_sub_permD8(val_2, val_2_swapped);
    re = Vectorized<ComplexFlt>(one) - re;
    auto root = el_blend<0xAA>(re, im).sqrt();
    auto ln = (b_a + root).log();
    return ln.el_swapped().conj();
#else
    return map(std::asin);
#endif
  }

  Vectorized<ComplexFlt> exp() const {
    return map(std::exp);
  }
  Vectorized<ComplexFlt> exp2() const {
    return map(exp2_impl);
  }
  Vectorized<ComplexFlt> expm1() const {
    return map(std::expm1);
  }

  Vectorized<ComplexFlt> eq(const Vectorized<ComplexFlt>& other) const {
    auto eq = (*this == other); // compares real and imag individually
    // If both real numbers and imag numbers are equal, then the complex numbers
    // are equal
    return (eq.real() & eq.imag()) & one;
  }
  Vectorized<ComplexFlt> ne(const Vectorized<ComplexFlt>& other) const {
    auto ne = (*this != other); // compares real and imag individually
    // If either real numbers or imag numbers are not equal, then the complex
    // numbers are not equal
    return (ne.real() | ne.imag()) & one;
  }

  Vectorized<ComplexFlt> sgn() const {
    return map(at::native::sgn_impl);
  }

  Vectorized<ComplexFlt> operator<(const Vectorized<ComplexFlt>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<ComplexFlt> operator<=(const Vectorized<ComplexFlt>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<ComplexFlt> operator>(const Vectorized<ComplexFlt>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<ComplexFlt> operator>=(const Vectorized<ComplexFlt>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  DEFINE_MEMBER_OP(operator==, ComplexFlt, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, ComplexFlt, vec_cmpne)

  DEFINE_MEMBER_OP(operator+, ComplexFlt, vec_add)
  DEFINE_MEMBER_OP(operator-, ComplexFlt, vec_sub)
  DEFINE_MEMBER_OP(operator&, ComplexFlt, vec_and)
  DEFINE_MEMBER_OP(operator|, ComplexFlt, vec_or)
  DEFINE_MEMBER_OP(operator^, ComplexFlt, vec_xor)
  // elementwise helpers
  DEFINE_MEMBER_OP(elwise_mult, ComplexFlt, vec_mul)
  DEFINE_MEMBER_OP(elwise_div, ComplexFlt, vec_div)
  DEFINE_MEMBER_OP(elwise_gt, ComplexFlt, vec_cmpgt)
  DEFINE_MEMBER_OP(elwise_ge, ComplexFlt, vec_cmpge)
  DEFINE_MEMBER_OP(elwise_lt, ComplexFlt, vec_cmplt)
  DEFINE_MEMBER_OP(elwise_le, ComplexFlt, vec_cmple)
  DEFINE_MEMBER_OP(elwise_max, ComplexFlt, vec_max)
};

template <>
Vectorized<ComplexFlt> inline maximum(
    const Vectorized<ComplexFlt>& a,
    const Vectorized<ComplexFlt>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_LT_OQ);
  // auto max = _mm256_blendv_ps(a, b, mask);
  auto mask = abs_a.elwise_lt(abs_b);
  auto max = Vectorized<ComplexFlt>::elwise_blendv(a, b, mask);

  return max;
  // Exploit the fact that all-ones is a NaN.
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(max, isnan);
}

template <>
Vectorized<ComplexFlt> inline minimum(
    const Vectorized<ComplexFlt>& a,
    const Vectorized<ComplexFlt>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_GT_OQ);
  // auto min = _mm256_blendv_ps(a, b, mask);
  auto mask = abs_a.elwise_gt(abs_b);
  auto min = Vectorized<ComplexFlt>::elwise_blendv(a, b, mask);
  return min;
  // Exploit the fact that all-ones is a NaN.
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(min, isnan);
}

template <>
Vectorized<ComplexFlt> C10_ALWAYS_INLINE
operator+(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
  return Vectorized<ComplexFlt>{
      vec_add(a.vec0(), b.vec0()), vec_add(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexFlt> C10_ALWAYS_INLINE
operator-(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
  return Vectorized<ComplexFlt>{
      vec_sub(a.vec0(), b.vec0()), vec_sub(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexFlt> C10_ALWAYS_INLINE
operator&(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
  return Vectorized<ComplexFlt>{
      vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexFlt> C10_ALWAYS_INLINE
operator|(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
  return Vectorized<ComplexFlt>{
      vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexFlt> C10_ALWAYS_INLINE
operator^(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
  return Vectorized<ComplexFlt>{
      vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexFlt> C10_ALWAYS_INLINE
operator*(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
  // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
  // Split into real and imaginary parts
  auto a_real = a.el_mergee(); // real part of a
  auto a_imag = a.el_mergeo(); // imag part of a
  auto b_real = b.el_mergee(); // real part of b
  auto b_imag = b.el_mergeo(); // imag part of b

  auto b_imag_neg = b_imag ^ rsign_mask;
  // Compute components
  auto ac = a_real.elwise_mult(b_real); // real * real
  auto bd = a_imag.elwise_mult(b_imag_neg); // imag * imag
  auto ad = a_real.elwise_mult(b_imag); // real * imag
  auto bc = a_imag.elwise_mult(b_real); // imag * real

  // Real = ac - bd (fix the negative bd part)
  auto real = ac + bd; // Real part calculation
  auto imag = ad + bc; // Imaginary part calculation

  // Step 1: Extract from real and imag
  __vector float r0 = real.vec0(); // {r0, r1, r2, r3}
  __vector float i0 = imag.vec0(); // {i0, i1, i2, i3}

  __vector float r1 = real.vec1(); // imag[0..3]
  __vector float i1 = imag.vec1(); // imag[4..7]

  __vector unsigned char perm_lo = {
      0,
      1,
      2,
      3, // r0
      16,
      17,
      18,
      19, //
      8,
      9,
      10,
      11, // r1
      24,
      25,
      26,
      27};
  __vector float v0 =
      vec_perm(r0, i0, perm_lo); // Interleave r0 and i0, r1 and i1
  __vector float v1 = vec_perm(r1, i1, perm_lo);
  Vectorized<ComplexFlt> result(v0, v1);
  return result;
}

template <>
Vectorized<ComplexFlt> C10_ALWAYS_INLINE
operator/(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
  // Take absolute values of real and imaginary parts of b
  __at_align__ c10::complex<float>
      tmp1[Vectorized<c10::complex<float>>::size()];
  __at_align__ c10::complex<float>
      tmp2[Vectorized<c10::complex<float>>::size()];
  __at_align__ c10::complex<float> out[Vectorized<c10::complex<float>>::size()];
  a.store(tmp1);
  b.store(tmp2);
  for (const auto i :
       c10::irange(Vectorized<c10::complex<float>>::
                       size())) { //{Vectorized<c10::complex<float>>::size()))
                                  //{
    out[i] = tmp1[i] / tmp2[i];
  }
  return Vectorized<ComplexFlt>::loadu(out);
}

} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at
