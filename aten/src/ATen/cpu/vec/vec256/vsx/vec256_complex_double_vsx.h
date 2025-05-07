#pragma once
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <c10/util/complex.h>
#include <c10/util/irange.h>

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {
using ComplexDbl = c10::complex<double>;

template <>
class Vectorized<ComplexDbl> {
  union {
    struct {
      vfloat64 _vec0;
      vfloat64 _vec1;
    };
    struct {
      vbool64 _vecb0;
      vbool64 _vecb1;
    };

  } __attribute__((__may_alias__));

 public:
  using value_type = ComplexDbl;
  using vec_internal_type = vfloat64;
  using vec_internal_mask_type = vbool64;
  using size_type = int;
  static constexpr size_type size() {
    return 2;
  }
  Vectorized() {}
  C10_ALWAYS_INLINE Vectorized(vfloat64 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorized(vbool64 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorized(vfloat64 v1, vfloat64 v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorized(vbool64 v1, vbool64 v2) : _vecb0{v1}, _vecb1{v2} {}

  Vectorized(ComplexDbl val) {
    double real_value = val.real();
    double imag_value = val.imag();
    _vec0 = vfloat64{real_value, imag_value};
    _vec1 = vfloat64{real_value, imag_value};
  }
  Vectorized(ComplexDbl val1, ComplexDbl val2) {
    _vec0 = vfloat64{val1.real(), val1.imag()};
    _vec1 = vfloat64{val2.real(), val2.imag()};
  }

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 0, Vectorized<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    return a;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 1, Vectorized<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    return b;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 2, Vectorized<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    return {b._vec0, a._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 3, Vectorized<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    return {a._vec0, b._vec1};
  }

  template <int64_t mask>
  static Vectorized<ComplexDbl> C10_ALWAYS_INLINE
  el_blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    const vbool64 mask_1st = VsxDblMask1(mask);
    const vbool64 mask_2nd = VsxDblMask2(mask);
    return {
        (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  static Vectorized<ComplexDbl> blendv(
      const Vectorized<ComplexDbl>& a,
      const Vectorized<ComplexDbl>& b,
      const Vectorized<ComplexDbl>& mask) {
    // convert std::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_complex =
        Vectorized<ComplexDbl>(vec_splat(mask._vec0, 0), vec_splat(mask._vec1, 0));
    return {
        vec_sel(a._vec0, b._vec0, mask_complex._vecb0),
        vec_sel(a._vec1, b._vec1, mask_complex._vecb1)};
  }

  static Vectorized<ComplexDbl> C10_ALWAYS_INLINE elwise_blendv(
      const Vectorized<ComplexDbl>& a,
      const Vectorized<ComplexDbl>& b,
      const Vectorized<ComplexDbl>& mask) {
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }
  template <typename step_t>
  static Vectorized<ComplexDbl> arange(
      ComplexDbl base = 0.,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<ComplexDbl>(base, base + step);
  }
  static Vectorized<ComplexDbl> set(
      const Vectorized<ComplexDbl>& a,
      const Vectorized<ComplexDbl>& b,
      int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
    }
    return b;
  }

  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const double*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const double*>(ptr))};
    }

    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    return {
        vec_vsx_ld(offset0, reinterpret_cast<const double*>(tmp_values)),
        vec_vsx_ld(offset16, reinterpret_cast<const double*>(tmp_values))};
  }
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<double*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<double*>(ptr));
    } else if (count > 0) {
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, reinterpret_cast<double*>(tmp_values));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<double*>(tmp_values));
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }

  const ComplexDbl& operator[](int idx) const = delete;
  ComplexDbl& operator[](int idx) = delete;

  Vectorized<ComplexDbl> map(ComplexDbl (*const f)(ComplexDbl)) const {
    __at_align__ ComplexDbl tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  Vectorized<ComplexDbl> map(ComplexDbl (*const f)(const ComplexDbl&)) const {
    __at_align__ ComplexDbl tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  Vectorized<ComplexDbl> el_swapped() const {
    vfloat64 v0 = vec_xxpermdi(_vec0, _vec0, 2);
    vfloat64 v1 = vec_xxpermdi(_vec1, _vec1, 2);
    return {v0, v1};
  }

  Vectorized<ComplexDbl> el_madd(
      const Vectorized<ComplexDbl>& multiplier,
      const Vectorized<ComplexDbl>& val) const {
    return {
        vec_madd(_vec0, multiplier._vec0, val._vec0),
        vec_madd(_vec1, multiplier._vec1, val._vec1)};
  }

  Vectorized<ComplexDbl> el_mergeo() const {
    vfloat64 v0 = vec_splat(_vec0, 1);
    vfloat64 v1 = vec_splat(_vec1, 1);
    return {v0, v1};
  }

  Vectorized<ComplexDbl> el_mergee() const {
    vfloat64 v0 = vec_splat(_vec0, 0);
    vfloat64 v1 = vec_splat(_vec1, 0);
    return {v0, v1};
  }

  static Vectorized<ComplexDbl> el_mergee(
      const Vectorized<ComplexDbl>& first,
      const Vectorized<ComplexDbl>& second) {
    return {
        vec_mergeh(first._vec0, second._vec0),
        vec_mergeh(first._vec1, second._vec1)};
  }

  static Vectorized<ComplexDbl> el_mergeo(
      const Vectorized<ComplexDbl>& first,
      const Vectorized<ComplexDbl>& second) {
    return {
        vec_mergel(first._vec0, second._vec0),
        vec_mergel(first._vec1, second._vec1)};
  }

  Vectorized<ComplexDbl> abs_2_() const {
    auto a = (*this).elwise_mult(*this);
    auto permuted = a.el_swapped();
    a = a + permuted;
    return a;
  }

  Vectorized<ComplexDbl> abs_() const {
    auto vi = el_mergeo();
    auto vr = el_mergee();
    return {Sleef_hypotd2_u05vsx(vr._vec0, vi._vec0), Sleef_hypotd2_u05vsx(vr._vec1, vi._vec1)};
  }

  Vectorized<ComplexDbl> abs() const {
    return abs_() & vd_real_mask;
  }

  Vectorized<ComplexDbl> angle_() const {
    // angle = atan2(b/a)
    // auto b_a = _mm256_permute_pd(values, 0x05);     // b        a
    // return Sleef_atan2d4_u10(values, b_a);          // 90-angle angle
    Vectorized<ComplexDbl> ret;
    ret._vec0[0] = std::atan2(_vec0[1], _vec0[0]);
    ret._vec1[0] = std::atan2(_vec1[1], _vec1[0]);
    return ret;
  }

  Vectorized<ComplexDbl> angle() const {
    return angle_() & vd_real_mask;
  }

  Vectorized<ComplexDbl> real_() const {
    return *this & vd_real_mask;
  }
  Vectorized<ComplexDbl> real() const {
    return *this & vd_real_mask;
  }
  Vectorized<ComplexDbl> imag_() const {
    return *this & vd_imag_mask;
  }
  Vectorized<ComplexDbl> imag() const {
    return imag_().el_swapped();
  }

  Vectorized<ComplexDbl> conj_() const {
    return *this ^ vd_isign_mask;
  }
  Vectorized<ComplexDbl> conj() const {
    return *this ^ vd_isign_mask;
  }

  Vectorized<ComplexDbl> log() const {
    // Most trigonomic ops use the log() op to improve complex number
    // performance.
    return map(std::log);
  }

  Vectorized<ComplexDbl> log2() const {
    // log2eB_inv
    auto ret = log();
    return ret.elwise_mult(vd_log2e_inv);
  }
  Vectorized<ComplexDbl> log10() const {
    auto ret = log();
    return ret.elwise_mult(vd_log10e_inv);
  }

  Vectorized<ComplexDbl> log1p() const {
    return map(std::log1p);
  }

  Vectorized<ComplexDbl> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    auto conj = conj_();
    auto b_a = conj.el_swapped();
    auto ab = conj.elwise_mult(b_a);
    auto im = ab + ab;
    auto val_2 = (*this).elwise_mult(*this);
    auto val_2_swapped = val_2.el_swapped();
    auto re = horizontal_sub(val_2, val_2_swapped);
    re = Vectorized<ComplexDbl>(vd_one) - re;
    auto root = el_blend<0x0A>(re, im).sqrt();
    auto ln = (b_a + root).log();
    return ln.el_swapped().conj();
  }

  Vectorized<ComplexDbl> acos() const {
    // acos(x) = pi/2 - asin(x)
    return Vectorized(vd_pi_2) - asin();
  }

  Vectorized<ComplexDbl> atan() const {
    // atan(x) = i/2 * ln((i + z)/(i - z))
    auto ione = Vectorized(vd_imag_one);
    auto sum = ione + *this;
    auto sub = ione - *this;
    auto ln = (sum / sub).log(); // ln((i + z)/(i - z))
    return ln * vd_imag_half; // i/2*ln()
  }
  Vectorized<ComplexDbl> atanh() const {
    return map(std::atanh);
  }

  Vectorized<ComplexDbl> sin() const {
    return map(std::sin);
  }
  Vectorized<ComplexDbl> sinh() const {
    return map(std::sinh);
  }
  Vectorized<ComplexDbl> cos() const {
    return map(std::cos);
  }
  Vectorized<ComplexDbl> cosh() const {
    return map(std::cosh);
  }

  Vectorized<ComplexDbl> tan() const {
    return map(std::tan);
  }
  Vectorized<ComplexDbl> tanh() const {
    return map(std::tanh);
  }
  Vectorized<ComplexDbl> ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  Vectorized<ComplexDbl> floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  Vectorized<ComplexDbl> neg() const {
    auto z = Vectorized<ComplexDbl>(vd_zero);
    return z - *this;
  }
  Vectorized<ComplexDbl> round() const {
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }

  Vectorized<ComplexDbl> trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<ComplexDbl> elwise_sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }

  Vectorized<ComplexDbl> sqrt() const {
    return map(std::sqrt);
  }

  Vectorized<ComplexDbl> reciprocal() const {
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2() = c/abs_2()
    // im = (bc - ad)/abs_2() = d/abs_2()
    auto c_d = *this ^ vd_isign_mask; // c       -d
    auto abs = abs_2_();
    return c_d.elwise_div(abs);
  }

  Vectorized<ComplexDbl> rsqrt() const {
    return sqrt().reciprocal();
  }

  static Vectorized<ComplexDbl> horizontal_add(
      Vectorized<ComplexDbl>& first,
      Vectorized<ComplexDbl>& second) {
    // Operates on individual floats, see _mm_hadd_ps
    // {f0+f1, s0+s1, f2+f3, s2+s3, ...}
    // i.e. it sums the re and im of each value and interleaves first and second:
    // {f_re0 + f_im0, s_re0 + s_im0, f_re1 + f_im1, s_re1 + s_im1, ...}
    return el_mergee(first, second) + el_mergeo(first, second);
  }

  static Vectorized<ComplexDbl> horizontal_sub(
      Vectorized<ComplexDbl>& first,
      Vectorized<ComplexDbl>& second) {
    // we will simulate it differently with 6 instructions total
    // lets permute second so that we can add it getting horizontal sums
    auto first_perm = first.el_swapped(); // 2perm
    auto second_perm = second.el_swapped(); // 2perm
    // summ
    auto first_ret = first - first_perm; // 2sub
    auto second_ret = second - second_perm; // 2 sub
    // now lets choose evens
    return el_mergee(first_ret, second_ret); // 2 mergee's
  }

  Vectorized<ComplexDbl> inline operator*(const Vectorized<ComplexDbl>& b) const {
    //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
#if 1
    // this is more vsx friendly than simulating horizontal from x86
    auto vi = b.el_mergeo();
    auto vr = b.el_mergee();
    vi = vi ^ vd_rsign_mask;
    auto ret = elwise_mult(vr);
    auto vx_swapped = el_swapped();
    ret = vx_swapped.elwise_mult(vi) + ret;
#else
    auto ac_bd = elwise_mult(b);
    auto d_c = b.el_swapped();
    d_c = d_c ^ vd_isign_mask;
    auto ad_bc = elwise_mult(d_c);
    auto ret = horizontal_sub(ac_bd, ad_bc);
#endif
    return ret;
  }

  Vectorized<ComplexDbl> inline operator/(const Vectorized<ComplexDbl>& b) const {
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2()
    // im = (bc - ad)/abs_2()
    //auto fabs_cd =  Vectorized{
    //    vec_andc(b._vec0, vd_sign_mask),
    //    vec_andc(b._vec1, vd_sign_mask)};       // |c|            |d|
    //auto fabs_dc =  fabs_cd.el_swapped();     // |d|            |c|
    //auto scale = fabs_cd.elwise_max(fabs_dc); // sc = max(|c|, |d|)
    //auto a2 = elwise_div(scale);              // a/sc           b/sc
    //auto b2 = b.elwise_div(scale);            // c/sc           d/sc
    //auto acbd2 = a2.elwise_mult(b2);          // ac/sc^2        bd/sc^2
    //auto dc2 = b2.el_swapped();               // d/sc           c/sc
    //dc2 = dc2 ^ vd_rsign_mask;                // -d/sc          c/sc
    //auto adbc2 = a2.elwise_mult(dc2);         // -ad/sc^2       bc/sc^2
    //auto ret = horizontal_add(acbd2, adbc2);  // (ac+bd)/sc^2   (bc-ad)/sc^2
    //auto denom2 = b2.abs_2_();                // (c^2+d^2)/sc^2 (c^2+d^2)/sc^2
    //ret = ret.elwise_div(denom2);
    //return ret;

    __at_align__ c10::complex<double> tmp1[Vectorized<c10::complex<double>>::size()];
    __at_align__ c10::complex<double> tmp2[Vectorized<c10::complex<double>>::size()];
    __at_align__ c10::complex<double> out[Vectorized<c10::complex<double>>::size()];
    this->store(tmp1);
    b.store(tmp2);

    for (const auto i : c10::irange(Vectorized<c10::complex<float>>::size())){
        out[i] = tmp1[i] / tmp2[i];
    }
    return loadu(out);
  }

  Vectorized<ComplexDbl> exp() const {
    return map(std::exp);
  }
  Vectorized<ComplexDbl> exp2() const {
    return map(exp2_impl);
  }
  Vectorized<ComplexDbl> expm1() const {
    return map(std::expm1);
  }

  Vectorized<ComplexDbl> pow(const Vectorized<ComplexDbl>& exp) const {
    __at_align__ ComplexDbl x_tmp[size()];
    __at_align__ ComplexDbl y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (const auto i : c10::irange(size())) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }

  Vectorized<ComplexDbl> sgn() const {
    return map(at::native::sgn_impl);
  }

  Vectorized<ComplexDbl> operator<(const Vectorized<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<ComplexDbl> operator<=(const Vectorized<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<ComplexDbl> operator>(const Vectorized<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<ComplexDbl> operator>=(const Vectorized<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<ComplexDbl> eq(const Vectorized<ComplexDbl>& other) const {
    auto eq = (*this == other);  // compares real and imag individually
    // If both real numbers and imag numbers are equal, then the complex numbers are equal
    return (eq.real() & eq.imag()) & vd_one;
  }
  Vectorized<ComplexDbl> ne(const Vectorized<ComplexDbl>& other) const {
    auto ne = (*this != other);  // compares real and imag individually
    // If either real numbers or imag numbers are not equal, then the complex numbers are not equal
    return (ne.real() | ne.imag()) & vd_one;
  }

  DEFINE_MEMBER_OP(operator==, ComplexDbl, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, ComplexDbl, vec_cmpne)

  DEFINE_MEMBER_OP(operator+, ComplexDbl, vec_add)
  DEFINE_MEMBER_OP(operator-, ComplexDbl, vec_sub)
  DEFINE_MEMBER_OP(operator&, ComplexDbl, vec_and)
  DEFINE_MEMBER_OP(operator|, ComplexDbl, vec_or)
  DEFINE_MEMBER_OP(operator^, ComplexDbl, vec_xor)
  // elementwise helpers
  DEFINE_MEMBER_OP(elwise_mult, ComplexDbl, vec_mul)
  DEFINE_MEMBER_OP(elwise_div, ComplexDbl, vec_div)
  DEFINE_MEMBER_OP(elwise_gt, ComplexDbl, vec_cmpgt)
  DEFINE_MEMBER_OP(elwise_ge, ComplexDbl, vec_cmpge)
  DEFINE_MEMBER_OP(elwise_lt, ComplexDbl, vec_cmplt)
  DEFINE_MEMBER_OP(elwise_le, ComplexDbl, vec_cmple)
  DEFINE_MEMBER_OP(elwise_max, ComplexDbl, vec_max)
};

template <>
Vectorized<ComplexDbl> inline maximum(
    const Vectorized<ComplexDbl>& a,
    const Vectorized<ComplexDbl>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_LT_OQ);
  // auto max = _mm256_blendv_ps(a, b, mask);
  auto mask = abs_a.elwise_lt(abs_b);
  auto max = Vectorized<ComplexDbl>::elwise_blendv(a, b, mask);

  return max;
  // Exploit the fact that all-ones is a NaN.
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(max, isnan);
}

template <>
Vectorized<ComplexDbl> inline minimum(
    const Vectorized<ComplexDbl>& a,
    const Vectorized<ComplexDbl>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_GT_OQ);
  // auto min = _mm256_blendv_ps(a, b, mask);
  auto mask = abs_a.elwise_gt(abs_b);
  auto min = Vectorized<ComplexDbl>::elwise_blendv(a, b, mask);
  return min;
  // Exploit the fact that all-ones is a NaN.
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(min, isnan);
}

template <>
Vectorized<ComplexDbl> C10_ALWAYS_INLINE operator+(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
  return Vectorized<ComplexDbl>{vec_add(a.vec0(), b.vec0()), vec_add(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexDbl> C10_ALWAYS_INLINE operator-(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
  return Vectorized<ComplexDbl>{vec_sub(a.vec0(), b.vec0()), vec_sub(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexDbl> C10_ALWAYS_INLINE operator&(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
  return Vectorized<ComplexDbl>{vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexDbl> C10_ALWAYS_INLINE operator|(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
  return Vectorized<ComplexDbl>{vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexDbl> C10_ALWAYS_INLINE operator^(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
  return Vectorized<ComplexDbl>{vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

template <>
Vectorized<ComplexDbl> C10_ALWAYS_INLINE operator*(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    // Split into real and imaginary parts
    auto a_real = a.el_mergee();  // real part of a
    auto a_imag = a.el_mergeo();  // imag part of a
    auto b_real = b.el_mergee();  // real part of b
    auto b_imag = b.el_mergeo();  // imag part of b

    // Compute components
    auto ac = a_real.elwise_mult(b_real); // real*real
    auto bd = a_imag.elwise_mult(b_imag); // imag*imag

    // Real part: ac - bd
    auto real = ac - bd;

    auto ad = a_real.elwise_mult(b_imag); // real*imag
    auto bc = a_imag.elwise_mult(b_real); // imag*real

    // Imag = ad + bc
    auto imag = ad + bc;

    // Merge real and imaginary parts into vectors
    __vector double v0 = vec_mergeh(real.vec0(), imag.vec0()); // [r0, i0]
    __vector double v1 = vec_mergeh(real.vec1(), imag.vec1()); // [r1, i1]

    // Create the final result
    auto result = Vectorized<ComplexDbl>{v0, v1};
    return result;
}

template <>
Vectorized<ComplexDbl> C10_ALWAYS_INLINE operator/(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2()
    // im = (bc - ad)/abs_2()
    // Take absolute values of real and imaginary parts of b
    __at_align__ c10::complex<double> tmp1[Vectorized<c10::complex<double>>::size()];
    __at_align__ c10::complex<double> tmp2[Vectorized<c10::complex<double>>::size()];
    __at_align__ c10::complex<double> out[Vectorized<c10::complex<double>>::size()];
    a.store(tmp1);
    b.store(tmp2);
    for (const auto i : c10::irange(Vectorized<c10::complex<double>>::size())){
        out[i] = tmp1[i] / tmp2[i];
    }
    return Vectorized<ComplexDbl>::loadu(out);
}

} // namespace
} // namespace vec
} // namespace at
