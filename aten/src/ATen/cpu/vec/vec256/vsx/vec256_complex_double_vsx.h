#pragma once
#include <ATen/cpu/vec/vec256/intrinsics.h>
#include <ATen/cpu/vec/vec256/vec256_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <c10/util/complex.h>

namespace at {
namespace vec {
// See Note [Acceptable use of anonymous namespace in header]
namespace {
using ComplexDbl = c10::complex<double>;

template <>
class Vectorize<ComplexDbl> {
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
  Vectorize() {}
  C10_ALWAYS_INLINE Vectorize(vfloat64 v) : _vec0{v}, _vec1{v} {}
  C10_ALWAYS_INLINE Vectorize(vbool64 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  C10_ALWAYS_INLINE Vectorize(vfloat64 v1, vfloat64 v2) : _vec0{v1}, _vec1{v2} {}
  C10_ALWAYS_INLINE Vectorize(vbool64 v1, vbool64 v2) : _vecb0{v1}, _vecb1{v2} {}

  Vectorize(ComplexDbl val) {
    double real_value = val.real();
    double imag_value = val.imag();
    _vec0 = vfloat64{real_value, imag_value};
    _vec1 = vfloat64{real_value, imag_value};
  }
  Vectorize(ComplexDbl val1, ComplexDbl val2) {
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
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 0, Vectorize<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorize<ComplexDbl>& a, const Vectorize<ComplexDbl>& b) {
    return a;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 1, Vectorize<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorize<ComplexDbl>& a, const Vectorize<ComplexDbl>& b) {
    return b;
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 2, Vectorize<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorize<ComplexDbl>& a, const Vectorize<ComplexDbl>& b) {
    return {b._vec0, a._vec1};
  }

  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 3, Vectorize<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorize<ComplexDbl>& a, const Vectorize<ComplexDbl>& b) {
    return {a._vec0, b._vec1};
  }

  template <int64_t mask>
  static Vectorize<ComplexDbl> C10_ALWAYS_INLINE
  el_blend(const Vectorize<ComplexDbl>& a, const Vectorize<ComplexDbl>& b) {
    const vbool64 mask_1st = VsxDblMask1(mask);
    const vbool64 mask_2nd = VsxDblMask2(mask);
    return {
        (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  static Vectorize<ComplexDbl> blendv(
      const Vectorize<ComplexDbl>& a,
      const Vectorize<ComplexDbl>& b,
      const Vectorize<ComplexDbl>& mask) {
    // convert std::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_complex =
        Vectorize<ComplexDbl>(vec_splat(mask._vec0, 0), vec_splat(mask._vec1, 0));
    return {
        vec_sel(a._vec0, b._vec0, mask_complex._vecb0),
        vec_sel(a._vec1, b._vec1, mask_complex._vecb1)};
  }

  static Vectorize<ComplexDbl> C10_ALWAYS_INLINE elwise_blendv(
      const Vectorize<ComplexDbl>& a,
      const Vectorize<ComplexDbl>& b,
      const Vectorize<ComplexDbl>& mask) {
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }
  template <typename step_t>
  static Vectorize<ComplexDbl> arange(
      ComplexDbl base = 0.,
      step_t step = static_cast<step_t>(1)) {
    return Vectorize<ComplexDbl>(base, base + step);
  }
  static Vectorize<ComplexDbl> set(
      const Vectorize<ComplexDbl>& a,
      const Vectorize<ComplexDbl>& b,
      int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
    }
    return b;
  }

  static Vectorize<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const double*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const double*>(ptr))};
    }

    __at_align32__ value_type tmp_values[size()];
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
      __at_align32__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, reinterpret_cast<double*>(tmp_values));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<double*>(tmp_values));
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }

  const ComplexDbl& operator[](int idx) const = delete;
  ComplexDbl& operator[](int idx) = delete;

  Vectorize<ComplexDbl> map(ComplexDbl (*f)(ComplexDbl)) const {
    __at_align32__ ComplexDbl tmp[size()];
    store(tmp);
    for (int i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  Vectorize<ComplexDbl> map(ComplexDbl (*f)(const ComplexDbl&)) const {
    __at_align32__ ComplexDbl tmp[size()];
    store(tmp);
    for (int i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  Vectorize<ComplexDbl> el_swapped() const {
    vfloat64 v0 = vec_xxpermdi(_vec0, _vec0, 2);
    vfloat64 v1 = vec_xxpermdi(_vec1, _vec1, 2);
    return {v0, v1};
  }

  Vectorize<ComplexDbl> el_madd(
      const Vectorize<ComplexDbl>& multiplier,
      const Vectorize<ComplexDbl>& val) const {
    return {
        vec_madd(_vec0, multiplier._vec0, val._vec0),
        vec_madd(_vec1, multiplier._vec1, val._vec1)};
  }

  Vectorize<ComplexDbl> el_mergeo() const {
    vfloat64 v0 = vec_splat(_vec0, 1);
    vfloat64 v1 = vec_splat(_vec1, 1);
    return {v0, v1};
  }

  Vectorize<ComplexDbl> el_mergee() const {
    vfloat64 v0 = vec_splat(_vec0, 0);
    vfloat64 v1 = vec_splat(_vec1, 0);
    return {v0, v1};
  }

  static Vectorize<ComplexDbl> el_mergee(
      Vectorize<ComplexDbl>& first,
      Vectorize<ComplexDbl>& second) {
    // as mergee phased in , we can use vec_perm with mask
    return {
        vec_mergeh(first._vec0, second._vec0),
        vec_mergeh(first._vec1, second._vec1)};
  }

  Vectorize<ComplexDbl> abs_2_() const {
    auto a = (*this).elwise_mult(*this);
    auto permuted = a.el_swapped();
    a = a + permuted;
    return a;
  }

  Vectorize<ComplexDbl> abs_() const {
    auto ret = abs_2_();
    return ret.elwise_sqrt();
  }

  Vectorize<ComplexDbl> abs() const {
    return abs_() & vd_real_mask;
  }

  Vectorize<ComplexDbl> angle_() const {
    // angle = atan2(b/a)
    // auto b_a = _mm256_permute_pd(values, 0x05);     // b        a
    // return Sleef_atan2d4_u10(values, b_a);          // 90-angle angle
    auto ret = el_swapped();
    for (int i = 0; i < 2; i++) {
      ret._vec0[i] = std::atan2(_vec0[i], ret._vec0[i]);
      ret._vec1[i] = std::atan2(_vec1[i], ret._vec0[i]);
    }
    return ret;
  }

  Vectorize<ComplexDbl> angle() const {
    auto a = angle_().el_swapped();
    return a & vd_real_mask;
  }

  Vectorize<ComplexDbl> real_() const {
    return *this & vd_real_mask;
  }
  Vectorize<ComplexDbl> real() const {
    return *this & vd_real_mask;
  }
  Vectorize<ComplexDbl> imag_() const {
    return *this & vd_imag_mask;
  }
  Vectorize<ComplexDbl> imag() const {
    return imag_().el_swapped();
  }

  Vectorize<ComplexDbl> conj_() const {
    return *this ^ vd_isign_mask;
  }
  Vectorize<ComplexDbl> conj() const {
    return *this ^ vd_isign_mask;
  }

  Vectorize<ComplexDbl> log() const {
    // Most trigonomic ops use the log() op to improve complex number
    // performance.
    return map(std::log);
  }

  Vectorize<ComplexDbl> log2() const {
    // log2eB_inv
    auto ret = log();
    return ret.elwise_mult(vd_log2e_inv);
  }
  Vectorize<ComplexDbl> log10() const {
    auto ret = log();
    return ret.elwise_mult(vd_log10e_inv);
  }

  Vectorize<ComplexDbl> asin() const {
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
    re = Vectorize<ComplexDbl>(vd_one) - re;
    auto root = el_blend<0x0A>(re, im).sqrt();
    auto ln = (b_a + root).log();
    return ln.el_swapped().conj();
  }

  Vectorize<ComplexDbl> acos() const {
    // acos(x) = pi/2 - asin(x)
    return Vectorize(vd_pi_2) - asin();
  }

  Vectorize<ComplexDbl> atan() const {
    // atan(x) = i/2 * ln((i + z)/(i - z))
    auto ione = Vectorize(vd_imag_one);
    auto sum = ione + *this;
    auto sub = ione - *this;
    auto ln = (sum / sub).log(); // ln((i + z)/(i - z))
    return ln * vd_imag_half; // i/2*ln()
  }

  Vectorize<ComplexDbl> sin() const {
    return map(std::sin);
  }
  Vectorize<ComplexDbl> sinh() const {
    return map(std::sinh);
  }
  Vectorize<ComplexDbl> cos() const {
    return map(std::cos);
  }
  Vectorize<ComplexDbl> cosh() const {
    return map(std::cosh);
  }

  Vectorize<ComplexDbl> tan() const {
    return map(std::tan);
  }
  Vectorize<ComplexDbl> tanh() const {
    return map(std::tanh);
  }
  Vectorize<ComplexDbl> ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  Vectorize<ComplexDbl> floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  Vectorize<ComplexDbl> neg() const {
    auto z = Vectorize<ComplexDbl>(vd_zero);
    return z - *this;
  }
  Vectorize<ComplexDbl> round() const {
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }

  Vectorize<ComplexDbl> trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorize<ComplexDbl> elwise_sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }

  void dump() const {
    std::cout << _vec0[0] << "," << _vec0[1] << ",";
    std::cout << _vec1[0] << "," << _vec1[1] << std::endl;
  }

  Vectorize<ComplexDbl> sqrt() const {
    return map(std::sqrt);
  }

  Vectorize<ComplexDbl> reciprocal() const {
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2() = c/abs_2()
    // im = (bc - ad)/abs_2() = d/abs_2()
    auto c_d = *this ^ vd_isign_mask; // c       -d
    auto abs = abs_2_();
    return c_d.elwise_div(abs);
  }

  Vectorize<ComplexDbl> rsqrt() const {
    return sqrt().reciprocal();
  }

  static Vectorize<ComplexDbl> horizontal_add(
      Vectorize<ComplexDbl>& first,
      Vectorize<ComplexDbl>& second) {
    auto first_perm = first.el_swapped(); // 2perm
    auto second_perm = second.el_swapped(); // 2perm
    // summ
    auto first_ret = first + first_perm; // 2add
    auto second_ret = second + second_perm; // 2 add
    // now lets choose evens
    return el_mergee(first_ret, second_ret); // 2 mergee's
  }

  static Vectorize<ComplexDbl> horizontal_sub(
      Vectorize<ComplexDbl>& first,
      Vectorize<ComplexDbl>& second) {
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

  Vectorize<ComplexDbl> inline operator*(const Vectorize<ComplexDbl>& b) const {
    //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
#if 1
    // this is more vsx friendly than simulating horizontal from x86
    auto vi = b.el_mergeo();
    auto vr = b.el_mergee();
    vi = vi ^ vd_rsign_mask;
    auto ret = elwise_mult(vr);
    auto vx_swapped = el_swapped();
    ret = vx_swapped.el_madd(vi, ret);
#else
    auto ac_bd = elwise_mult(b);
    auto d_c = b.el_swapped();
    d_c = d_c ^ vd_isign_mask;
    auto ad_bc = elwise_mult(d_c);
    auto ret = horizontal_sub(ac_bd, ad_bc);
#endif
    return ret;
  }

  Vectorize<ComplexDbl> inline operator/(const Vectorize<ComplexDbl>& b) const {
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2()
    // im = (bc - ad)/abs_2()
#if 1
    auto vi = b.el_mergeo();
    auto vr = b.el_mergee();
    auto abs_b = b.abs_2_();
    vi = vi ^ vd_isign_mask;
    auto ret = elwise_mult(vr);
    auto vx_swapped = el_swapped();
    ret = vx_swapped.el_madd(vi, ret);
    ret = ret.elwise_div(abs_b);
#else
    // Vectorize x86 simulation
    auto ac_bd = elwise_mult(b);
    auto d_c = b.el_swapped();
    d_c = d_c ^ vd_rsign_mask;
    auto ad_bc = elwise_mult(d_c);
    auto abs_b = b.abs_2_();
    auto re_im = horizontal_add(ac_bd, ad_bc);
    auto ret = re_im.elwise_div(abs_b);
#endif
    return ret;
  }

  Vectorize<ComplexDbl> exp() const {
    return map(std::exp);
  }

  Vectorize<ComplexDbl> pow(const Vectorize<ComplexDbl>& exp) const {
    __at_align32__ ComplexDbl x_tmp[size()];
    __at_align32__ ComplexDbl y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (int i = 0; i < size(); i++) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }

  Vectorize<ComplexDbl> sgn() const {
    return map(at::native::sgn_impl);
  }

  Vectorize<ComplexDbl> hypot(const Vectorize<ComplexDbl>& b) const {
      TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<ComplexDbl> nextafter(const Vectorize<ComplexDbl>& b) const {
      TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<ComplexDbl> igamma(const Vectorize<ComplexDbl>& x) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<ComplexDbl> igammac(const Vectorize<ComplexDbl>& x) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<ComplexDbl> log1p() const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<ComplexDbl> atan2(const Vectorize<ComplexDbl>& b) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<ComplexDbl> erf() const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<ComplexDbl> erfc() const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<ComplexDbl> expm1() const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<ComplexDbl> operator<(const Vectorize<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<ComplexDbl> operator<=(const Vectorize<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<ComplexDbl> operator>(const Vectorize<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<ComplexDbl> operator>=(const Vectorize<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<ComplexDbl> eq(const Vectorize<ComplexDbl>& other) const {
    auto ret = (*this == other);
    return ret & vd_one;
  }
  Vectorize<ComplexDbl> ne(const Vectorize<ComplexDbl>& other) const {
    auto ret = (*this != other);
    return ret & vd_one;
  }

  Vectorize<ComplexDbl> lt(const Vectorize<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<ComplexDbl> le(const Vectorize<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<ComplexDbl> gt(const Vectorize<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<ComplexDbl> ge(const Vectorize<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  DEFINE_MEMBER_OP(operator==, ComplexDbl, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, ComplexDbl, vec_cmpne)

  DEFINE_MEMBER_OP(operator+, ComplexDbl, vec_add)
  DEFINE_MEMBER_OP(operator-, ComplexDbl, vec_sub)
  DEFINE_MEMBER_OP(operator&, ComplexDbl, vec_and)
  DEFINE_MEMBER_OP(operator|, ComplexDbl, vec_or)
  DEFINE_MEMBER_OP(operator^, ComplexDbl, vec_xor)
  // elelemtwise helpers
  DEFINE_MEMBER_OP(elwise_mult, ComplexDbl, vec_mul)
  DEFINE_MEMBER_OP(elwise_div, ComplexDbl, vec_div)
  DEFINE_MEMBER_OP(elwise_gt, ComplexDbl, vec_cmpgt)
  DEFINE_MEMBER_OP(elwise_ge, ComplexDbl, vec_cmpge)
  DEFINE_MEMBER_OP(elwise_lt, ComplexDbl, vec_cmplt)
  DEFINE_MEMBER_OP(elwise_le, ComplexDbl, vec_cmple)
};

template <>
Vectorize<ComplexDbl> inline maximum(
    const Vectorize<ComplexDbl>& a,
    const Vectorize<ComplexDbl>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_LT_OQ);
  // auto max = _mm256_blendv_ps(a, b, mask);
  auto mask = abs_a.elwise_lt(abs_b);
  auto max = Vectorize<ComplexDbl>::elwise_blendv(a, b, mask);

  return max;
  // Exploit the fact that all-ones is a NaN.
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(max, isnan);
}

template <>
Vectorize<ComplexDbl> inline minimum(
    const Vectorize<ComplexDbl>& a,
    const Vectorize<ComplexDbl>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_GT_OQ);
  // auto min = _mm256_blendv_ps(a, b, mask);
  auto mask = abs_a.elwise_gt(abs_b);
  auto min = Vectorize<ComplexDbl>::elwise_blendv(a, b, mask);
  return min;
  // Exploit the fact that all-ones is a NaN.
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(min, isnan);
}


} // namespace
} // namespace vec256
} // namespace at
