#pragma once
// Shared code for bfloat16 and float16.

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

namespace at::vec {
inline namespace CPU_CAPABILITY {

// Shared implementation between Vectorized<c10::Half> and
// Vectorized<c10::BFloat16>. Uses CRTP to allow derived class
// customization.
template <typename VecT, typename ValueT, template <int, bool> typename BlendRegs, typename Derived>
struct Vectorized16 {
 protected:
  VecT values;
 public:
  using value_type = ValueT;
  using size_type = int;
  static constexpr size_type size() {
    static_assert(sizeof(VecT) == 8 * sizeof(value_type));
    return 8;
  }

 protected:
  Derived map2(
      const Derived& second,
      value_type (*const f)(value_type, value_type)) const {
    __at_align__ value_type tmp_first[size()];
    __at_align__ value_type tmp_second[size()];
    static_cast<const Derived*>(this)->store(tmp_first); // store this to tmp_first
    second.store(tmp_second);
    for (const auto i : c10::irange(size())) {
      tmp_first[i] = f(tmp_first[i], tmp_second[i]);
    }
    return Derived::loadu(tmp_first);
  }

 public:
  Vectorized16() = default;
  Vectorized16(VecT v) : values(v) {}

  operator VecT() const {
    return values;
  }

  template <int64_t mask>
  static Derived blend(const Derived& a, const Derived& b) {
    Derived vec;
    vec.values = BlendRegs<0, (mask & 0x01) != 0>::impl(
        a.values, b.values, vec.values);
    vec.values = BlendRegs<1, (mask & 0x02) != 0>::impl(
        a.values, b.values, vec.values);
    vec.values = BlendRegs<2, (mask & 0x04) != 0>::impl(
        a.values, b.values, vec.values);
    vec.values = BlendRegs<3, (mask & 0x08) != 0>::impl(
        a.values, b.values, vec.values);

    vec.values = BlendRegs<4, (mask & 0x10) != 0>::impl(
        a.values, b.values, vec.values);
    vec.values = BlendRegs<5, (mask & 0x20) != 0>::impl(
        a.values, b.values, vec.values);
    vec.values = BlendRegs<6, (mask & 0x40) != 0>::impl(
        a.values, b.values, vec.values);
    vec.values = BlendRegs<7, (mask & 0x80) != 0>::impl(
        a.values, b.values, vec.values);

    return vec;
  }

  template <typename step_t>
  static Derived arange(
      value_type base = 0,
      step_t step = static_cast<step_t>(1)) {
    const Derived base_vec(base);
    const Derived step_vec(step);
    const Derived step_sizes(
        value_type(0),
        value_type(1),
        value_type(2),
        value_type(3),
        value_type(4),
        value_type(5),
        value_type(6),
        value_type(7));
    return fmadd(step_sizes, step_vec, base_vec);
  }

  // Very slow implementation of indexing.
  // Only required because vec256_qint refers to this.
  // Once we specialize that implementation for ARM
  // this should be removed. TODO (kimishpatel)
  value_type operator[](int idx) const {
    __at_align__ value_type tmp[size()];
    static_cast<const Derived*>(this)->store(tmp);
    return tmp[idx];
  }

  int zero_mask() const {
    __at_align__ value_type tmp[size()];
    static_cast<const Derived*>(this)->store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Derived map(value_type (*const f)(value_type)) const {
    __at_align__ value_type tmp[size()];
    static_cast<const Derived*>(this)->store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return Derived::loadu(tmp);
  }

  Derived angle() const {
    auto zero = Derived(0);
    auto pi = Derived(c10::pi<value_type>);
    auto tmp = Derived::blendv(zero, pi, *static_cast<const Derived*>(this) < zero);
    return Derived::blendv(tmp, *static_cast<const Derived*>(this), static_cast<const Derived*>(this)->isnan());
  }
  Derived real() const {
    return *this;
  }
  Derived imag() const {
    return Derived(0);
  }
  Derived conj() const {
    return *this;
  }

  // Sleef does not support FP16/BF16, so many math functions are applied by
  // converting to FP32, applying the math function, and then converting back to
  // FP16/BF16.
  Derived acos() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::acos);
  }
  Derived acosh() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::acosh);
  }
  Derived asin() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::asin);
  }
  Derived atan() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::atan);
  }
  Derived atanh() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::atanh);
  }
  Derived atan2(const Derived& exp) const {
    return static_cast<const Derived*>(this)->map2_with_vec_float_method(exp, &Vectorized<float>::atan2);
  }
  Derived copysign(const Derived& sign) const {
    return static_cast<const Derived*>(this)->map2_with_vec_float_method(sign, &Vectorized<float>::copysign);
  }
  Derived erf() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::erf);
  }
  Derived erfc() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::erfc);
  }
  Derived erfinv() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::erfinv);
  }
  Derived exp() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::exp);
  }
  Derived exp2() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::exp2);
  }
  Derived expm1() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::expm1);
  }
  Derived exp_u20() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::exp_u20);
  }
  Derived fmod(const Derived& q) const {
    // This function is questionable with a conversion, so we use map2
    return map2(q, std::fmod);
  }
  Derived hypot(const Derived& b) const {
    return static_cast<const Derived*>(this)->map2_with_vec_float_method(b, &Vectorized<float>::hypot);
  }
  Derived i0() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::i0);
  }
  Derived i0e() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::i0e);
  }
  Derived digamma() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::digamma);
  }
  Derived igamma(const Derived& x) const {
    return static_cast<const Derived*>(this)->map2_with_vec_float_method(x, &Vectorized<float>::igamma);
  }
  Derived igammac(const Derived& x) const {
    return static_cast<const Derived*>(this)->map2_with_vec_float_method(x, &Vectorized<float>::igammac);
  }
  Derived log() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::log);
  }
  Derived log10() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::log10);
  }
  Derived log1p() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::log1p);
  }
  Derived log2() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::log2);
  }
  Derived nextafter(const Derived& b) const {
    // This function does not make sense with conversion, so we use map2
    return map2(b, std::nextafter);
  }
  Derived sin() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::sin);
  }
  Derived sinh() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::sinh);
  }
  Derived cos() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::cos);
  }
  Derived cosh() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::cosh);
  }
  Derived ceil() const {
    // This function is questionable with a conversion, so we use map
    return map(at::native::ceil_impl);
  }
  Derived floor() const {
    // This function is questionable with a conversion, so we use map
    return map(at::native::floor_impl);
  }
  Derived round() const {
    // This function is questionable with a conversion, so we use map
    return map(at::native::round_impl);
  }
  Derived tan() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::tan);
  }
  Derived tanh() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::tanh);
  }
  Derived lgamma() const {
    return static_cast<const Derived*>(this)->map_with_vec_float_method(&Vectorized<float>::lgamma);
  }
  Derived rsqrt() const {
    return static_cast<const Derived*>(this)->sqrt().reciprocal();
  }
  Derived pow(const Derived& exp) const {
    return static_cast<const Derived*>(this)->map2_with_vec_float_method(exp, &Vectorized<float>::pow);
  }

};


} // namespace CPU_CAPABILITY
} // namespace at::vec
