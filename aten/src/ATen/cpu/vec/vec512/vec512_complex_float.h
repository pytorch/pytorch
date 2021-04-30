#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <c10/util/complex.h>
#include <ATen/cpu/vec/vec512/intrinsics.h>
#include <ATen/cpu/vec/vec512/vec512_base.h>
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

template <> class Vectorize<c10::complex<float>> {
private:
  __m512 values;
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
public:
  using value_type = c10::complex<float>;
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }
  Vectorize() {}
  Vectorize(__m512 v) : values(v) {}
  Vectorize(c10::complex<float> val) {
    float real_value = val.real();
    float imag_value = val.imag();
    values = _mm512_setr_ps(real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value);
  }
  Vectorize(c10::complex<float> val1, c10::complex<float> val2,
            c10::complex<float> val3, c10::complex<float> val4,
            c10::complex<float> val5, c10::complex<float> val6,
            c10::complex<float> val7, c10::complex<float> val8) {
    values = _mm512_setr_ps(val1.real(), val1.imag(),
                            val2.real(), val2.imag(),
                            val3.real(), val3.imag(),
                            val4.real(), val4.imag(),
                            val5.real(), val5.imag(),
                            val6.real(), val6.imag(),
                            val7.real(), val7.imag(),
                            val8.real(), val8.imag());
  }
  operator __m512() const {
    return values;
  }
  template <int64_t mask>
  static Vectorize<c10::complex<float>> blend(const Vectorize<c10::complex<float>>& a,
                                              const Vectorize<c10::complex<float>>& b) {
     // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    // NOLINTNEXTLINE(clang-diagnostic-warning)
    switch (mask) {
      case 0:
        return a;
      case 1:
        return _mm512_mask_blend_ps(0x03, a.values, b.values); //b0000 0001 = b0000 0011
      case 2:
        return _mm512_mask_blend_ps(0x0C, a.values, b.values); //b0000 0010 = b0000 1100
      case 3:
        return _mm512_mask_blend_ps(0x0F, a.values, b.values); //b0000 0011 = b0000 1111
      case 4:
        return _mm512_mask_blend_ps(0x30, a.values, b.values); //b0000 0100 = b0011 0000
      case 5:
        return _mm512_mask_blend_ps(0x33, a.values, b.values); //b0000 0101 = b0011 0011
      case 6:
        return _mm512_mask_blend_ps(0x3C, a.values, b.values); //b0000 0110 = b0011 1100
      case 7:
        return _mm512_mask_blend_ps(0x3F, a.values, b.values); //b0000 0111 = b0011 1111
      case 8:
        return _mm512_mask_blend_ps(0xC0, a.values, b.values); //b0000 1000 = b1100 0000
      case 9:
        return _mm512_mask_blend_ps(0xC3, a.values, b.values); //b0000 1001 = b1100 0011
      case 10:
        return _mm512_mask_blend_ps(0xCC, a.values, b.values); //b0000 1010 = b1100 1100
      case 11:
        return _mm512_mask_blend_ps(0xCF, a.values, b.values); //b0000 1011 = b1100 1111
      case 12:
        return _mm512_mask_blend_ps(0xF0, a.values, b.values); //b0000 1100 = b1111 0000
      case 13:
        return _mm512_mask_blend_ps(0xF3, a.values, b.values); //b0000 1101 = b1111 0011
      case 14:
        return _mm512_mask_blend_ps(0xFC, a.values, b.values); //b0000 1110 = b1111 1100
      case 15:
        return _mm512_mask_blend_ps(0xFF, a.values, b.values); //b0000 1111 = b1111 1111
      case 16:
        return _mm512_mask_blend_ps(0x300, a.values, b.values); //b0001 0000 = b11 0000 0000
      case 17:
        return _mm512_mask_blend_ps(0x30C, a.values, b.values); //b0001 0001 = b11 0000 0011
      case 18:        
        return _mm512_mask_blend_ps(0x30C, a.values, b.values); //b0001 0010 = b11 0000 1100
      case 19:
        return _mm512_mask_blend_ps(0x30F, a.values, b.values); //b0001 0011 = b11 0000 1111
      case 20:
        return _mm512_mask_blend_ps(0x330, a.values, b.values); //b0001 0100 = b11 0011 0000
      case 21:
        return _mm512_mask_blend_ps(0x333, a.values, b.values); //b0001 0101 = b11 0011 0011
      case 22:
        return _mm512_mask_blend_ps(0x33C, a.values, b.values); //b0001 0110 = b11 0011 1100
      case 23:
        return _mm512_mask_blend_ps(0x33F, a.values, b.values); //b0001 0111  = b11 0011 1111
      case 24:
        return _mm512_mask_blend_ps(0x3C0, a.values, b.values); //b0001 1000 = b11 1100 0000
      case 25:
        return _mm512_mask_blend_ps(0x3C3, a.values, b.values); //b0001 1001 = b111100 0011
      case 26:
        return _mm512_mask_blend_ps(0x3CC, a.values, b.values); //b0000 1010 = b11 1100 1100
      case 27:
        return _mm512_mask_blend_ps(0x3CF, a.values, b.values); //b0000 1011 = b11 1100 1111
      case 28:
        return _mm512_mask_blend_ps(0x3F0, a.values, b.values); //b000 1100 = b11 1111 0000
      case 29:
        return _mm512_mask_blend_ps(0x3F3, a.values, b.values); //b0000 1101 = b1 1111 0011
      case 30:
        return _mm512_mask_blend_ps(0x3FF, a.values, b.values); //b0001 1110 = b11 1111 1100
      case 31:
        return _mm512_mask_blend_ps(0x3FF, a.values, b.values); //b01 1111 = b11 1111 1111
    }
    return b;
  }
  static Vectorize<c10::complex<float>> blendv(const Vectorize<c10::complex<float>>& a,
                                               const Vectorize<c10::complex<float>>& b,
                                               const Vectorize<c10::complex<float>>& mask) {
    // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_ = _mm512_castps_si512(_mm512_unpacklo_ps(mask.values, mask.values));
    return _mm512_permutex2var_ps(a.values, mask_, b.values);

  }
  template<typename step_t>
  static Vectorize<c10::complex<float>> arange(c10::complex<float> base = 0., 
                                               step_t step = static_cast<step_t>(1)) {
    return Vectorize<c10::complex<float>>(base,
                                        base + step,
                                        base + c10::complex<float>(2)*step,
                                        base + c10::complex<float>(3)*step,
                                        base + c10::complex<float>(4)*step,
                                        base + c10::complex<float>(5)*step,
                                        base + c10::complex<float>(6)*step,
                                        base + c10::complex<float>(7)*step);
  }
  static Vectorize<c10::complex<float>> set(const Vectorize<c10::complex<float>>& a, 
                                            const Vectorize<c10::complex<float>>& b,
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
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }
    return b;
  }
  static Vectorize<c10::complex<float>> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));

    __at_align64__ float tmp_values[2*size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < 2*size(); ++i) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const float*>(ptr),
        count * sizeof(c10::complex<float>));
    return _mm512_load_ps(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[2*size()];
      _mm512_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<float>));
    }
  }
  // AVX512 doesn't have horizontal add & horizontal sub instructions.
  // TODO: hadd_ps() & hsub_ps() may have scope for improvement.
  // At https://stackoverflow.com/questions/26896432/horizontal-add-with-m512-avx512/26905830,
  // Peter Cordes recommends not using _mm256_hadd_ps because it has a high latency
  // and blocks port longer than other clever alternatives.
  static __m512 hadd_ps(__m512 a, __m512 b) {
    auto first_half_a = _mm512_extracti32x8_epi32(_mm512_castps_si512(a), 0);
    auto second_half_a = _mm512_extracti32x8_epi32(_mm512_castps_si512(a), 1);
    auto first_half_b = _mm512_extracti32x8_epi32(_mm512_castps_si512(b), 0);
    auto second_half_b = _mm512_extracti32x8_epi32(_mm512_castps_si512(b), 1);
    auto first_half_hadd = _mm256_hadd_ps(_mm256_castsi256_ps(first_half_a),
                                          _mm256_castsi256_ps(first_half_b));
    auto second_half_hadd = _mm256_hadd_ps(_mm256_castsi256_ps(second_half_a),
                                           _mm256_castsi256_ps(second_half_b));
    auto ret_val = _mm512_set1_ps(0.0);
    ret_val = _mm512_insertf32x8(ret_val, first_half_hadd, 0);
    ret_val = _mm512_insertf32x8(ret_val, second_half_hadd, 1);
    return ret_val;
  }
  static __m512 hsub_ps(__m512 a, __m512 b) {
    auto first_half_a = _mm512_extracti32x8_epi32(_mm512_castps_si512(a), 0);
    auto second_half_a = _mm512_extracti32x8_epi32(_mm512_castps_si512(a), 1);
    auto first_half_b = _mm512_extracti32x8_epi32(_mm512_castps_si512(b), 0);
    auto second_half_b = _mm512_extracti32x8_epi32(_mm512_castps_si512(b), 1);
    auto first_half_hsub = _mm256_hsub_ps(_mm256_castsi256_ps(first_half_a),
                                          _mm256_castsi256_ps(first_half_b));
    auto second_half_hsub = _mm256_hsub_ps(_mm256_castsi256_ps(second_half_a),
                                           _mm256_castsi256_ps(second_half_b));
    auto ret_val = _mm512_set1_ps(0.0);
    ret_val = _mm512_insertf32x8(ret_val, first_half_hsub, 0);
    ret_val = _mm512_insertf32x8(ret_val, second_half_hsub, 1);
    return ret_val;
  }
  const c10::complex<float>& operator[](int idx) const  = delete;
  c10::complex<float>& operator[](int idx) = delete;
  Vectorize<c10::complex<float>> map(c10::complex<float> (*f)(const c10::complex<float> &)) const {
    __at_align64__ c10::complex<float> tmp[size()];
    store(tmp);
    for (int i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  __m512 abs_2_() const {
    auto val_2 = _mm512_mul_ps(values, values);     // a*a     b*b
    auto ret = hadd_ps(val_2, val_2);        // a*a+b*b a*a+b*b
    return _mm512_permute_ps(ret, 0xD8);
  }
  __m512 abs_() const {
    return _mm512_sqrt_ps(abs_2_());                // abs     abs
  }
  Vectorize<c10::complex<float>> abs() const {
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    return _mm512_and_ps(abs_(), real_mask);        // abs     0
  }
  __m512 angle_() const {
    //angle = atan2(b/a)
    auto b_a = _mm512_permute_ps(values, 0xB1);     // b        a
    return Sleef_atan2f16_u10(values, b_a);          // 90-angle angle
  }
  Vectorize<c10::complex<float>> angle() const {
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    auto angle = _mm512_permute_ps(angle_(), 0xB1); // angle    90-angle
    return _mm512_and_ps(angle, real_mask);         // angle    0
  }
  Vectorize<c10::complex<float>> sgn() const {
    auto abs = abs_();
    auto zero = _mm512_setzero_ps();
    auto mask = _mm512_cmp_ps_mask(abs, zero, _CMP_EQ_OQ);
    auto abs_val = Vectorize(abs);

    auto div = values / abs_val.values;       // x / abs(x)

    return _mm512_mask_blend_ps(mask, div, zero);
  }
  __m512 real_() const {
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    return _mm512_and_ps(values, real_mask);
  }
  Vectorize<c10::complex<float>> real() const {
    return real_();
  }
  __m512 imag_() const {
    const __m512 imag_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF));
    return _mm512_and_ps(values, imag_mask);
  }
  Vectorize<c10::complex<float>> imag() const {
    return _mm512_permute_ps(imag_(), 0xB1);        //b        a
  }
  __m512 conj_() const {
    const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                            0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    return _mm512_xor_ps(values, sign_mask);        // a       -b
  }
  Vectorize<c10::complex<float>> conj() const {
    return conj_();
  }
  Vectorize<c10::complex<float>> log() const {
    // Most trigonomic ops use the log() op to improve complex number performance.
    return map(std::log);
  }
  Vectorize<c10::complex<float>> log2() const {
    const __m512 log2_ = _mm512_set1_ps(std::log(2));
    return _mm512_div_ps(log(), log2_);
  }
  Vectorize<c10::complex<float>> log10() const {
    const __m512 log10_ = _mm512_set1_ps(std::log(10));
    return _mm512_div_ps(log(), log10_);
  }
  Vectorize<c10::complex<float>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    const __m512 one = _mm512_set1_ps(1);

    auto conj = conj_();
    auto b_a = _mm512_permute_ps(conj, 0xB1);                         //-b        a
    auto ab = _mm512_mul_ps(conj, b_a);                               //-ab       -ab
    auto im = _mm512_add_ps(ab, ab);                                  //-2ab      -2ab

    auto val_2 = _mm512_mul_ps(values, values);                       // a*a      b*b
    auto re = hsub_ps(val_2, _mm512_permute_ps(val_2, 0xB1));  // a*a-b*b  b*b-a*a
    re = _mm512_permute_ps(re, 0xD8);
    re = _mm512_sub_ps(one, re);

    auto root = Vectorize(_mm512_mask_blend_ps(0xAAAA, re, im)).sqrt();         //sqrt(re + i*im)
    auto ln = Vectorize(_mm512_add_ps(b_a, root)).log();                 //ln(iz + sqrt())
    return Vectorize(_mm512_permute_ps(ln.values, 0xB1)).conj();         //-i*ln()
  }
  Vectorize<c10::complex<float>> acos() const {
    return map(std::acos);
  }
  Vectorize<c10::complex<float>> atan() const;
  Vectorize<c10::complex<float>> atan2(const Vectorize<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> erf() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> erfc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> exp() const {
    //exp(a + bi)
    // = exp(a)*(cos(b) + sin(b)i)
    auto exp = Sleef_expf16_u10(values);                               //exp(a)           exp(b)
    exp = _mm512_mask_blend_ps(0xAAAA, exp, _mm512_permute_ps(exp, 0xB1));   //exp(a)           exp(a)

    auto sin_cos = Sleef_sincosf16_u10(values);                        //[sin(a), cos(a)] [sin(b), cos(b)]
    auto cos_sin = _mm512_mask_blend_ps(0xAAAA, _mm512_permute_ps(sin_cos.y, 0xB1),
                                   sin_cos.x);                  //cos(b)           sin(b)
    return _mm512_mul_ps(exp, cos_sin);
  }
  Vectorize<c10::complex<float>> expm1() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> sin() const {
    return map(std::sin);
  }
  Vectorize<c10::complex<float>> sinh() const {
    return map(std::sinh);
  }
  Vectorize<c10::complex<float>> cos() const {
    return map(std::cos);
  }
  Vectorize<c10::complex<float>> cosh() const {
    return map(std::cosh);
  }
  Vectorize<c10::complex<float>> ceil() const {
    return _mm512_ceil_ps(values);
  }
  Vectorize<c10::complex<float>> floor() const {
    return _mm512_floor_ps(values);
  }
  Vectorize<c10::complex<float>> hypot(const Vectorize<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> igamma(const Vectorize<c10::complex<float>> &x) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> igammac(const Vectorize<c10::complex<float>> &x) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> neg() const {
    auto zero = _mm512_setzero_ps();
    return _mm512_sub_ps(zero, values);
  }
  Vectorize<c10::complex<float>> nextafter(const Vectorize<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> round() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorize<c10::complex<float>> tan() const {
    return map(std::tan);
  }
  Vectorize<c10::complex<float>> tanh() const {
    return map(std::tanh);
  }
  Vectorize<c10::complex<float>> trunc() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorize<c10::complex<float>> sqrt() const {
    return map(std::sqrt);
  }
  Vectorize<c10::complex<float>> reciprocal() const;
  Vectorize<c10::complex<float>> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vectorize<c10::complex<float>> pow(const Vectorize<c10::complex<float>> &exp) const {
    __at_align64__ c10::complex<float> x_tmp[size()];
    __at_align64__ c10::complex<float> y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (int i = 0; i < size(); i++) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorize<c10::complex<float>> operator==(const Vectorize<c10::complex<float>>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_EQ_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF));
  }
  Vectorize<c10::complex<float>> operator!=(const Vectorize<c10::complex<float>>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_NEQ_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF));
  }
  Vectorize<c10::complex<float>> operator<(const Vectorize<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> operator<=(const Vectorize<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> operator>(const Vectorize<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> operator>=(const Vectorize<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorize<c10::complex<float>> eq(const Vectorize<c10::complex<float>>& other) const;
  Vectorize<c10::complex<float>> ne(const Vectorize<c10::complex<float>>& other) const;
  Vectorize<c10::complex<float>> lt(const Vectorize<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> le(const Vectorize<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> gt(const Vectorize<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorize<c10::complex<float>> ge(const Vectorize<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
};

template <> Vectorize<c10::complex<float>> inline operator+(const Vectorize<c10::complex<float>> &a,
                                                            const Vectorize<c10::complex<float>> &b) {
  return _mm512_add_ps(a, b);
}

template <> Vectorize<c10::complex<float>> inline operator-(const Vectorize<c10::complex<float>> &a,
                                                            const Vectorize<c10::complex<float>> &b) {
  return _mm512_sub_ps(a, b);
}

template <> Vectorize<c10::complex<float>> inline operator*(const Vectorize<c10::complex<float>> &a,
                                                            const Vectorize<c10::complex<float>> &b) {
  //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
  const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                          0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm512_mul_ps(a, b);         //ac       bd

  auto d_c = _mm512_permute_ps(b, 0xB1);    //d        c
  d_c = _mm512_xor_ps(sign_mask, d_c);      //d       -c
  auto ad_bc = _mm512_mul_ps(a, d_c);       //ad      -bc

  auto ret = Vectorize<c10::complex<float>>::hsub_ps(ac_bd, ad_bc);  //ac - bd  ad + bc
  ret = _mm512_permute_ps(ret, 0xD8);
  return ret;
}

template <> Vectorize<c10::complex<float>> inline operator/(const Vectorize<c10::complex<float>> &a,
                                                            const Vectorize<c10::complex<float>> &b) {
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2()
  //im = (bc - ad)/abs_2()
  const __m512 sign_mask = _mm512_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0,
                                          -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
  auto ac_bd = _mm512_mul_ps(a, b);         //ac       bd

  auto d_c = _mm512_permute_ps(b, 0xB1);    //d        c
  d_c = _mm512_xor_ps(sign_mask, d_c);      //-d       c
  auto ad_bc = _mm512_mul_ps(a, d_c);       //-ad      bc

  auto re_im = Vectorize<c10::complex<float>>::hadd_ps(ac_bd, ad_bc);//ac + bd  bc - ad
  re_im = _mm512_permute_ps(re_im, 0xD8);
  return _mm512_div_ps(re_im, b.abs_2_());
}

// reciprocal. Implement this here so we can use multiplication.
Vectorize<c10::complex<float>> Vectorize<c10::complex<float>>::reciprocal() const {
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2() = c/abs_2()
  //im = (bc - ad)/abs_2() = d/abs_2()
  const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                          0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm512_xor_ps(sign_mask, values);    //c       -d
  return _mm512_div_ps(c_d, abs_2_());
}

Vectorize<c10::complex<float>> Vectorize<c10::complex<float>>::atan() const {
  // atan(x) = i/2 * ln((i + z)/(i - z))
  const __m512 i = _mm512_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                  0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  const Vectorize i_half = _mm512_setr_ps(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
                                          0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);

  auto sum = Vectorize(_mm512_add_ps(i, values));                      // a        1+b
  auto sub = Vectorize(_mm512_sub_ps(i, values));                      // -a       1-b
  auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
  return i_half*ln;                                                 // i/2*ln()
}

template <>
Vectorize<c10::complex<float>> inline maximum(const Vectorize<c10::complex<float>>& a,
                                              const Vectorize<c10::complex<float>>& b) {
  auto zero_vector = _mm512_set1_epi32(0);
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_LT_OQ);
  auto max = _mm512_mask_blend_ps(mask, a, b);
  // Exploit the fact that all-ones is a NaN.
  auto isnan_mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_UNORD_Q);
  auto isnan = _mm512_mask_set1_epi32(zero_vector, isnan_mask, 0xFFFFFFFF);
  return _mm512_or_ps(max, _mm512_castsi512_ps(isnan));
}

template <>
Vectorize<c10::complex<float>> inline minimum(const Vectorize<c10::complex<float>>& a,
                                              const Vectorize<c10::complex<float>>& b) {
  auto zero_vector = _mm512_set1_epi32(0);
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_GT_OQ);
  auto min = _mm512_mask_blend_ps(mask, a, b);
  // Exploit the fact that all-ones is a NaN.
  auto isnan_mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_UNORD_Q);
  auto isnan = _mm512_mask_set1_epi32(zero_vector, isnan_mask, 0xFFFFFFFF);
  return _mm512_or_ps(min, _mm512_castsi512_ps(isnan));
}

template <>
Vectorize<c10::complex<float>> inline operator&(const Vectorize<c10::complex<float>>& a,
                                                const Vectorize<c10::complex<float>>& b) {
  return _mm512_and_ps(a, b);
}

template <>
Vectorize<c10::complex<float>> inline operator|(const Vectorize<c10::complex<float>>& a,
                                                const Vectorize<c10::complex<float>>& b) {
  return _mm512_or_ps(a, b);
}

template <>
Vectorize<c10::complex<float>> inline operator^(const Vectorize<c10::complex<float>>& a,
                                                const Vectorize<c10::complex<float>>& b) {
  return _mm512_xor_ps(a, b);
}

Vectorize<c10::complex<float>> Vectorize<c10::complex<float>>::eq(
    const Vectorize<c10::complex<float>>& other) const {
  return (*this == other) & Vectorize<c10::complex<float>>(_mm512_set1_ps(1.0f));
}

Vectorize<c10::complex<float>> Vectorize<c10::complex<float>>::ne(
    const Vectorize<c10::complex<float>>& other) const {
  return (*this != other) & Vectorize<c10::complex<float>>(_mm512_set1_ps(1.0f));
}

#endif

}}}
