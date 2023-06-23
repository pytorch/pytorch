#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>

#include <c10/util/irange.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint8.h>

#include <array>
#include <cmath>

// This file defines Vectorized<> for the quantized types.
//
//
// Currently, we simply use these classes as efficient converters between
// the quantized types and Vectorized<float>, usually in bandwidth-bound cases
// where doing the arithmetic in full-precision is acceptable (e.g.
// elementwise operators).
//
//
// Conversions are as follows:
//  Vectorized<qint8> -> 4x Vectorized<float>
//  Vectorized<quint8> -> 4x Vectorized<float>
//  Vectorized<qint32> -> 1x Vectorized<float>
//
// The size of the returned float vector is specified by the special
// constexpr function float_num_vecs. The type of the value returned
// from dequantize (and expected as an argument to quantize) is
// specified by float_vec_return_type.
//
// When writing kernels with these vectors, it is expected that floating-
// point operations will be carried out in a loop over Vectorized<T>::float_num_vecs
// iterations.

namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

struct Vectorizedqi {
 protected:
  __m512i vals __attribute__((aligned(64)));

 public:
  Vectorizedqi() {}
  Vectorizedqi(__m512i v) : vals(v) {}
  operator __m512i() const {
    return vals;
  }
};


template <typename T>
__m512i pack_saturate_and_clamp(
    __m512i first,
    __m512i second,
    T min_val,
    T max_val);

template <>
inline __m512i pack_saturate_and_clamp<int32_t>(
    __m512i first,
    __m512i second,
    int32_t min_val,
    int32_t max_val) {
  // This function is for linkage only, will not be used
  AT_ERROR("pack_saturate_and_clamp<int32_t> is not supported");
}

template <>
inline __m512i pack_saturate_and_clamp<int8_t>(
    __m512i first,
    __m512i second,
    int8_t min_val,
    int8_t max_val) {
  __m512i packed_and_sat = _mm512_packs_epi16(first, second);
  return _mm512_max_epi8(
      _mm512_set1_epi8(min_val),
      _mm512_min_epi8(packed_and_sat, _mm512_set1_epi8(max_val)));
}

template <>
inline __m512i pack_saturate_and_clamp<uint8_t>(
    __m512i first,
    __m512i second,
    uint8_t min_val,
    uint8_t max_val) {
  __m512i packed_and_sat = _mm512_packus_epi16(first, second);
  return _mm512_max_epu8(
      _mm512_set1_epi8(min_val),
      _mm512_min_epu8(packed_and_sat, _mm512_set1_epi8(max_val)));
}

inline Vectorized<float> load_uint8_as_float(const uint8_t* src_data) {
  // Load 16*uint8
  __m128i input_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_data));
  // Convert from 16*u8 to 16*int32
  __m512i input_512_extended = _mm512_cvtepu8_epi32(input_128);
  // Convert from 16*int32 to 16*float32
  return _mm512_cvtepi32_ps(input_512_extended);
}

inline void store_float_as_uint8(at::vec::Vectorized<float> values, uint8_t* dst_data) {
  // Convert from float32 to int32
  __m512i x_values_int32 = _mm512_cvtps_epi32(values);

  // Convert from int32 to int16 using signed saturation
  __m512i xy_packed_v = _mm512_packs_epi32(x_values_int32, x_values_int32);

  constexpr auto min_val = std::numeric_limits<uint8_t>::min();
  constexpr auto max_val = std::numeric_limits<uint8_t>::max();

  // Convert from int16 to uint8 using unsigned saturation
  __m512i xyzw_clamped_v = pack_saturate_and_clamp<uint8_t>(
      xy_packed_v, xy_packed_v, min_val, max_val);
  __m512i permute_mask_v =
      _mm512_set_epi32(0x0f, 0x0b, 0x07, 0x03, 0x0e, 0x0a, 0x06, 0x02,
                      0x0d, 0x09, 0x05, 0x01, 0x0c, 0x08, 0x04, 0x00);
  xyzw_clamped_v = _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);

  // Store to dst
  _mm_storeu_si128(
    reinterpret_cast<__m128i*>(dst_data),
    _mm512_castsi512_si128(xyzw_clamped_v));
}

template <typename T>
inline void __attribute__((always_inline)) QuantizeAvx512(
    const float* src,
    typename T::underlying* dst,
    int len,
    float inverse_scale,
    int64_t zero_point) {
  constexpr int VLEN = 16;
  constexpr auto min_val = std::numeric_limits<typename T::underlying>::min();
  constexpr auto max_val = std::numeric_limits<typename T::underlying>::max();
  const __m512i min_v = _mm512_set1_epi32(min_val);
  const __m512i max_v = _mm512_set1_epi32(max_val);
  // This is the largest int32 value < int32_max exactly representable in float
  constexpr int32_t int32_float_max_val =
      std::numeric_limits<int32_t>::max() - 127;
  int i = 0;
  __m512 inverse_scale_v = _mm512_set1_ps(inverse_scale);
  // clang-format off
  static const __m512i shuffle_mask_v = _mm512_set_epi8(
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff,
      0x0c, 0x08, 0x04, 0x00);
  // clang-format on
  __m512i permute_mask_v =
      _mm512_set_epi32(0x0f, 0x0b, 0x07, 0x03, 0x0e, 0x0a, 0x06, 0x02,
                       0x0d, 0x09, 0x05, 0x01, 0x0c, 0x08, 0x04, 0x00);
  __m512i permute_mask_l8_v =
      _mm512_set_epi32(0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0c, 0x08,
                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);
  int len_aligned = len / (VLEN * 4) * (VLEN * 4);
  for (; i < len_aligned; i += 4 * VLEN) {
    // x
    __m512 x_vals = _mm512_load_ps(src + i);
    __m512 x_transformed_v = _mm512_mul_ps(x_vals, inverse_scale_v);
    // If the floating point value is greater than int32_max,
    // _mm512_cvtps_epi32 converts them to -ve. Clip at int32_float_max_val to
    // Clip at int32_float_max_val to avoid this.
    x_transformed_v =
        _mm512_min_ps(x_transformed_v, _mm512_set1_ps(int32_float_max_val));
    // y
    __m512 y_vals = _mm512_load_ps(src + i + VLEN);
    __m512 y_transformed_v = _mm512_mul_ps(y_vals, inverse_scale_v);
    y_transformed_v =
        _mm512_min_ps(y_transformed_v, _mm512_set1_ps(int32_float_max_val));
    // z
    __m512 z_vals = _mm512_load_ps(src + i + 2 * VLEN);
    __m512 z_transformed_v = _mm512_mul_ps(z_vals, inverse_scale_v);
    z_transformed_v =
        _mm512_min_ps(z_transformed_v, _mm512_set1_ps(int32_float_max_val));
    // w
    __m512 w_vals = _mm512_load_ps(src + i + 3 * VLEN);
    __m512 w_transformed_v = _mm512_mul_ps(w_vals, inverse_scale_v);
    w_transformed_v =
        _mm512_min_ps(w_transformed_v, _mm512_set1_ps(int32_float_max_val));

    __m512i x_rounded_v = _mm512_cvtps_epi32(x_transformed_v);
    __m512i y_rounded_v = _mm512_cvtps_epi32(y_transformed_v);
    __m512i z_rounded_v = _mm512_cvtps_epi32(z_transformed_v);
    __m512i w_rounded_v = _mm512_cvtps_epi32(w_transformed_v);

    // add zero point
    x_rounded_v = _mm512_add_epi32(x_rounded_v, _mm512_set1_epi32(zero_point));
    y_rounded_v = _mm512_add_epi32(y_rounded_v, _mm512_set1_epi32(zero_point));
    z_rounded_v = _mm512_add_epi32(z_rounded_v, _mm512_set1_epi32(zero_point));
    w_rounded_v = _mm512_add_epi32(w_rounded_v, _mm512_set1_epi32(zero_point));

    __m512i xy_packed_v = _mm512_packs_epi32(x_rounded_v, y_rounded_v);
    __m512i zw_packed_v = _mm512_packs_epi32(z_rounded_v, w_rounded_v);
    __m512i xyzw_clamped_v = pack_saturate_and_clamp<typename T::underlying>(
        xy_packed_v, zw_packed_v, min_val, max_val);

    xyzw_clamped_v =
        _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), xyzw_clamped_v);
  }

  // Additional 8-lane AVX512 version to take advantage when len is smaller
  // based on fbgemm::QuantizeAvx2 (https://github.com/pytorch/FBGEMM)
  for (; i < len / VLEN * VLEN; i += VLEN) {
    __m512 x_vals = _mm512_load_ps(src + i);
    __m512 x_transformed_v = _mm512_mul_ps(x_vals, inverse_scale_v);
    x_transformed_v =
        _mm512_min_ps(x_transformed_v, _mm512_set1_ps(int32_float_max_val));
    __m512i x_rounded_v = _mm512_cvtps_epi32(x_transformed_v);
    x_rounded_v = _mm512_add_epi32(x_rounded_v, _mm512_set1_epi32(zero_point));
    __m512i x_clipped_v =
        _mm512_max_epi32(min_v, _mm512_min_epi32(max_v, x_rounded_v));

    x_clipped_v = _mm512_shuffle_epi8(x_clipped_v, shuffle_mask_v);
    x_clipped_v = _mm512_permutexvar_epi32(permute_mask_l8_v, x_clipped_v);
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(dst + i),
        _mm512_castsi512_si128(x_clipped_v));
  }

  for (; i < len; ++i) {
    float transformed = src[i] * inverse_scale;

    // Not exactly the same behavior as the vectorized code.
    // The vectorized code above always rounds to even in halfway cases
    // (https://software.intel.com/en-us/node/523819), but std::nearbyint
    // does the same only when the current rounding mode is FE_TONEAREST.
    // However, in practice, this should not be a problem because most cases
    // use the default rounding mode FE_TONEAREST.
    // Note that we cannot implement the same behavior as the vectorized code
    // using std::round because it does rounding away from zero in halfway
    // cases.
    transformed = zero_point + std::nearbyint(transformed);
    float clipped =
        std::min(std::max(transformed, float(min_val)), float(max_val));
    dst[i] = clipped;
  }
}

template<>
struct Vectorized<c10::qint32> : public Vectorizedqi {
    using size_type = int;
    static constexpr size_type size() {
        return 16;
    }

    static constexpr int float_num_vecs() {
        return 1;
    }

    static constexpr int int_num_vecs() {
        return 1;
    }

    using float_vec_return_type = std::array<Vectorized<float>, 1>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 1>;
    using value_type = c10::qint32::underlying;

 public:
    using Vectorizedqi::Vectorizedqi;
    Vectorized() {}

    Vectorized(__m512i vals_) { vals = vals_;}

    // Broadcast constructor
    Vectorized(const c10::qint32& val) {
        value_type uw = val.val_;
        vals = _mm512_set1_epi32(uw);
    }

    void store(void* ptr, int count = size()) const {
      if (count != size()) {
        memcpy(ptr, &vals, count * sizeof(value_type));
      } else {
        _mm512_storeu_si512((__m512i*)ptr, vals);
      }
    }

    static Vectorized<c10::qint32> loadu(const void* ptr) {
        return Vectorized<c10::qint32>(ptr);
    }

    static Vectorized<c10::qint32> loadu(const void* ptr, int64_t count) {
        __at_align__ value_type tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        return loadu(tmp_values);
    }

    float_vec_return_type dequantize(
        Vectorized<float> scale,
        Vectorized<float> zero_point,
        Vectorized<float> scale_zp_premul) const {
      __m512 float_vals = _mm512_cvtepi32_ps(vals);
      return {vec::fmadd(scale, Vectorized<float>(float_vals), scale_zp_premul)};
    }

    static Vectorized<c10::qint32> quantize(
        const float_vec_return_type& rhs,
        float scale,
        int32_t zero_point,
        float inverse_scale) {
      Vectorized<c10::qint32> retval;
      auto rhs_data = (__m512)rhs[0];
      at::native::quantize_vec<c10::qint32, /*precision=*/32>(
          scale, zero_point, (float*)&rhs_data, (c10::qint32*)&retval.vals, 16);
      return retval;
    }

    Vectorized<c10::qint32> maximum(Vectorized<c10::qint32> b) const {
      return _mm512_max_epi32(vals, b.vals);
    }

    Vectorized<c10::qint32> minimum(Vectorized<c10::qint32> b) const {
      return _mm512_min_epi32(vals, b.vals);
    }

    Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const {
        return maximum(zero_point);
    }

    Vectorized<c10::qint32> relu6(
        Vectorized<c10::qint32> zero_point,
        Vectorized<c10::qint32> q_six) {
      return _mm512_min_epi32(
          _mm512_max_epi32(vals, zero_point.vals), q_six.vals);
    }

    int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
      return {_mm512_sub_epi32(vals, b)};
    }

    static Vectorized<c10::qint32> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
      __m512 multiplier_v = _mm512_set1_ps(multiplier);
      __m512i zero_point_v = _mm512_set1_epi32(zero_point);

      __m512 scaled = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[0]), multiplier_v);
      __m512i rounded = _mm512_cvtps_epi32(scaled);
      return _mm512_add_epi32(rounded, zero_point_v);
    }

 private:
    // Load from memory constructor
    Vectorized(const void* ptr) {
      vals = _mm512_loadu_si512((const __m512i*)ptr);
    }
};

template <>
Vectorized<c10::qint32> inline maximum(const Vectorized<c10::qint32>& a, const Vectorized<c10::qint32>& b) {
  return a.maximum(b);
}

template <>
Vectorized<c10::qint32> inline operator*(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  return _mm512_mullo_epi32(a, b);
}

template <>
Vectorized<c10::qint32> inline operator+(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  return _mm512_add_epi32(a, b);
}

/*
 * Convert values from int32 back to int8/uint8
 */
template <typename T>
__m512i RequantizeAvx512(
    const std::array<Vectorized<c10::qint32>, 4>& inp,
    __m512 multiplier,
    __m512i zp) {
  static_assert(
      std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
      "Only int8_t/uint8_t are supported");
  constexpr auto min_val = std::numeric_limits<T>::min();
  constexpr auto max_val = std::numeric_limits<T>::max();
  __m512i permute_mask_v =
      _mm512_set_epi32(0x0f, 0x0b, 0x07, 0x03, 0x0e, 0x0a, 0x06, 0x02,
                       0x0d, 0x09, 0x05, 0x01, 0x0c, 0x08, 0x04, 0x00);
  __m512 x_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[0]), multiplier);
  __m512 y_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[1]), multiplier);
  __m512 z_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[2]), multiplier);
  __m512 w_scaled_v = _mm512_mul_ps(_mm512_cvtepi32_ps(inp[3]), multiplier);

  __m512i x_rounded_v = _mm512_cvtps_epi32(x_scaled_v);
  __m512i y_rounded_v = _mm512_cvtps_epi32(y_scaled_v);
  __m512i z_rounded_v = _mm512_cvtps_epi32(z_scaled_v);
  __m512i w_rounded_v = _mm512_cvtps_epi32(w_scaled_v);

  /* Add zero point */
  __m512i x_v = _mm512_add_epi32(x_rounded_v, zp);
  __m512i y_v = _mm512_add_epi32(y_rounded_v, zp);
  __m512i z_v = _mm512_add_epi32(z_rounded_v, zp);
  __m512i w_v = _mm512_add_epi32(w_rounded_v, zp);

  /* Pack to int16_t and saturate */
  __m512i xy_packed_v = _mm512_packs_epi32(x_v, y_v);
  __m512i zw_packed_v = _mm512_packs_epi32(z_v, w_v);

  __m512i xyzw_clamped_v =
      pack_saturate_and_clamp<T>(xy_packed_v, zw_packed_v, min_val, max_val);

  /*
   * xyzw_clamped_v has results in the following layout so we need to
   * permute: x0-3 y0-3 z0-3 w0-3 x4-7 y4-7 z4-7 w4-7 x8-11 y8-11 z8-11 w8-11 x12-15 y12-15 z12-15 w12-15
   */
  xyzw_clamped_v = _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);
  return xyzw_clamped_v;
}

template<>
struct Vectorized<c10::qint8> : public Vectorizedqi {
    static constexpr int size() {
        return 64;
    }

    static constexpr int float_num_vecs() {
        return 4;
    }

    static constexpr int int_num_vecs() {
        return 4;
    }

    using float_vec_return_type = std::array<Vectorized<float>, 4>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;
    using value_type = typename c10::qint8::underlying;

 public:
    using Vectorizedqi::Vectorizedqi;

    Vectorized() {}
    Vectorized(__m512i vals_) { vals = vals_;}

    // Broadcast constructor
    Vectorized(const c10::qint8& val) {
        value_type uw = val.val_;
        vals = _mm512_set1_epi8(uw);
    }

    // This is needed because the compiler emits awful code for the default
    // constructor for moving the enum
    Vectorized(const Vectorized<c10::qint8>& other) : Vectorizedqi(other.vals) { }

    // This is added to avoid error: definition of implicit copy assignment operator
    // for 'Vectorized<c10::qint8>' is deprecated because it has a user-declared
    // copy constructor [-Werror,-Wdeprecated-copy]
    Vectorized& operator=(const Vectorized<c10::qint8>&) = default;

    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            _mm512_storeu_si512((__m512i*)ptr, vals);
        }
    }

    static Vectorized<c10::qint8> loadu(const void* ptr) {
        return Vectorized<c10::qint8>(ptr);
    }

    static Vectorized<c10::qint8> loadu(const void* ptr, int64_t count) {
        __at_align__ value_type tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        return loadu(tmp_values);
    }

 private:
    __m512i cvtepi8_epi32(__m128i epi8_vals) const {
        return _mm512_cvtepi8_epi32(epi8_vals);
    }

 public:
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_neg_zp_premul) const {
    __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
    __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
    __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
    __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);

    __m512 float_val0 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val0));
    __m512 float_val1 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val1));
    __m512 float_val2 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val2));
    __m512 float_val3 = _mm512_cvtepi32_ps(cvtepi8_epi32(int_val3));

    auto val0 =
        vec::fmadd(scale, Vectorized<float>(float_val0), scale_neg_zp_premul);
    auto val1 =
        vec::fmadd(scale, Vectorized<float>(float_val1), scale_neg_zp_premul);
    auto val2 =
        vec::fmadd(scale, Vectorized<float>(float_val2), scale_neg_zp_premul);
    auto val3 =
        vec::fmadd(scale, Vectorized<float>(float_val3), scale_neg_zp_premul);
    return {val0, val1, val2, val3};
  }

  static Vectorized<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    auto* rhs_data = (float*)rhs.data();
    int8_t quantized_values[64];
    QuantizeAvx512<c10::qint8>(
        rhs_data, quantized_values, 64, inverse_scale, zero_point);
    return Vectorized<c10::qint8>::loadu(quantized_values);
  }

  Vectorized<c10::qint8> maximum(Vectorized<c10::qint8> b) const {
      return _mm512_max_epi8(vals, b.vals);
    }

  Vectorized<c10::qint8> minimum(Vectorized<c10::qint8> b) const {
      return _mm512_min_epi8(vals, b.vals);
    }

    Vectorized<c10::qint8> relu(Vectorized<c10::qint8> zero_point) const {
        return maximum(zero_point);
    }

    Vectorized<c10::qint8> relu6(
        Vectorized<c10::qint8> zero_point,
        Vectorized<c10::qint8> q_six) {
      return _mm512_min_epi8(
          _mm512_max_epi8(vals, zero_point.vals), q_six.vals);
    }

    int_vec_return_type widening_subtract(Vectorized<c10::qint8> b) const {
      __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
      __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
      __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
      __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);

      __m512i int32_val0 = cvtepi8_epi32(int_val0);
      __m512i int32_val1 = cvtepi8_epi32(int_val1);
      __m512i int32_val2 = cvtepi8_epi32(int_val2);
      __m512i int32_val3 = cvtepi8_epi32(int_val3);

      __m128i int_b0 = _mm_set_epi64x(b.vals[1], b.vals[0]);
      __m128i int_b1 = _mm_set_epi64x(b.vals[3], b.vals[2]);
      __m128i int_b2 = _mm_set_epi64x(b.vals[5], b.vals[4]);
      __m128i int_b3 = _mm_set_epi64x(b.vals[7], b.vals[6]);

      __m512i int32_b0 = cvtepi8_epi32(int_b0);
      __m512i int32_b1 = cvtepi8_epi32(int_b1);
      __m512i int32_b2 = cvtepi8_epi32(int_b2);
      __m512i int32_b3 = cvtepi8_epi32(int_b3);

      __m512i res_0 = _mm512_sub_epi32(int32_val0, int32_b0);
      __m512i res_1 = _mm512_sub_epi32(int32_val1, int32_b1);
      __m512i res_2 = _mm512_sub_epi32(int32_val2, int32_b2);
      __m512i res_3 = _mm512_sub_epi32(int32_val3, int32_b3);

      return {Vectorized<c10::qint32>(res_0),
              Vectorized<c10::qint32>(res_1),
              Vectorized<c10::qint32>(res_2),
              Vectorized<c10::qint32>(res_3)};
    }

    static Vectorized<c10::qint8> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
      __m512 multiplier_v = _mm512_set1_ps(multiplier);
      __m512i zero_point_v = _mm512_set1_epi32(zero_point);
      return RequantizeAvx512<value_type>(inp, multiplier_v, zero_point_v);
    }

 private:
    // Load from memory constructor
    Vectorized(const void* ptr) {
        vals = _mm512_loadu_si512((const __m512i*)ptr);
    }
};

template <>
Vectorized<c10::qint8> inline maximum(const Vectorized<c10::qint8>& a, const Vectorized<c10::qint8>& b) {
  return a.maximum(b);
}

template<>
struct Vectorized<c10::quint8> : public Vectorizedqi {
    static constexpr int size() {
        return 64;
    }

    static constexpr int float_num_vecs() {
        return 4;
    }

    static constexpr int int_num_vecs() {
        return 4;
    }

    using float_vec_return_type = std::array<Vectorized<float>, 4>;
    using int_vec_return_type = std::array<Vectorized<c10::qint32>, 4>;
    using value_type = typename c10::quint8::underlying;

 public:
    using Vectorizedqi::Vectorizedqi;
    Vectorized() {}

    Vectorized(__m512i vals_) { vals = vals_;}

    // Broadcast constructor
    Vectorized(const c10::quint8& val) {
        value_type uw = val.val_;
        vals = _mm512_set1_epi8(uw);
    }

    Vectorized(const Vectorized<c10::quint8>& other) : Vectorizedqi(other.vals) { }

    // This is added to avoid error: definition of implicit copy assignment operator
    // for 'Vectorized<c10::quint8>' is deprecated because it has a user-declared
    // copy constructor [-Werror,-Wdeprecated-copy]
    Vectorized& operator=(const Vectorized<c10::quint8>&) = default;

    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            _mm512_storeu_si512((__m512i*)ptr, vals);
        }
    }

    static Vectorized<c10::quint8> loadu(const void* ptr) {
        return Vectorized<c10::quint8>(ptr);
    }

    static Vectorized<c10::quint8> loadu(const void* ptr, int64_t count) {
        __at_align__ value_type tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (const auto i : c10::irange(size())) {
          tmp_values[i] = 0;
        }
        std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
        return loadu(tmp_values);
    }

 private:
    __m512i cvtepu8_epi32(__m128i epu8_vals) const {
        return _mm512_cvtepu8_epi32(epu8_vals);
    }

 public:
  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
    __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
    __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
    __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);

    __m512 float_val0 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val0));
    __m512 float_val1 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val1));
    __m512 float_val2 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val2));
    __m512 float_val3 = _mm512_cvtepi32_ps(cvtepu8_epi32(int_val3));

    auto val0 =
        vec::fmadd(scale, Vectorized<float>(float_val0), scale_zp_premul);
    auto val1 =
        vec::fmadd(scale, Vectorized<float>(float_val1), scale_zp_premul);
    auto val2 =
        vec::fmadd(scale, Vectorized<float>(float_val2), scale_zp_premul);
    auto val3 =
        vec::fmadd(scale, Vectorized<float>(float_val3), scale_zp_premul);

    return {val0, val1, val2, val3};
  }

  static Vectorized<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    auto* rhs_data = (float*)rhs.data();
    uint8_t quantized_values[64];
    QuantizeAvx512<c10::quint8>(
        rhs_data, quantized_values, 64, inverse_scale, zero_point);
    return Vectorized<c10::quint8>::loadu(quantized_values);
  }

  Vectorized<c10::quint8> maximum(Vectorized<c10::quint8> b) const {
      return _mm512_max_epu8(vals, b.vals);
    }

  Vectorized<c10::quint8> minimum(Vectorized<c10::quint8> b) const {
      return _mm512_min_epu8(vals, b.vals);
    }

    Vectorized<c10::quint8> relu(Vectorized<c10::quint8> zero_point) const {
        return maximum(zero_point);
    }

    Vectorized<c10::quint8> relu6(
        Vectorized<c10::quint8> zero_point,
        Vectorized<c10::quint8> q_six) {
      return _mm512_min_epu8(
          _mm512_max_epu8(vals, zero_point.vals), q_six.vals);
    }

    int_vec_return_type widening_subtract(Vectorized<c10::quint8> b) const {
      __m128i int_val0 = _mm_set_epi64x(vals[1], vals[0]);
      __m128i int_val1 = _mm_set_epi64x(vals[3], vals[2]);
      __m128i int_val2 = _mm_set_epi64x(vals[5], vals[4]);
      __m128i int_val3 = _mm_set_epi64x(vals[7], vals[6]);

      __m512i int32_val0 = cvtepu8_epi32(int_val0);
      __m512i int32_val1 = cvtepu8_epi32(int_val1);
      __m512i int32_val2 = cvtepu8_epi32(int_val2);
      __m512i int32_val3 = cvtepu8_epi32(int_val3);

      __m128i int_b0 = _mm_set_epi64x(b.vals[1], b.vals[0]);
      __m128i int_b1 = _mm_set_epi64x(b.vals[3], b.vals[2]);
      __m128i int_b2 = _mm_set_epi64x(b.vals[5], b.vals[4]);
      __m128i int_b3 = _mm_set_epi64x(b.vals[7], b.vals[6]);

      __m512i int32_b0 = cvtepu8_epi32(int_b0);
      __m512i int32_b1 = cvtepu8_epi32(int_b1);
      __m512i int32_b2 = cvtepu8_epi32(int_b2);
      __m512i int32_b3 = cvtepu8_epi32(int_b3);

      __m512i res_0 = _mm512_sub_epi32(int32_val0, int32_b0);
      __m512i res_1 = _mm512_sub_epi32(int32_val1, int32_b1);
      __m512i res_2 = _mm512_sub_epi32(int32_val2, int32_b2);
      __m512i res_3 = _mm512_sub_epi32(int32_val3, int32_b3);
      return {Vectorized<c10::qint32>(res_0),
              Vectorized<c10::qint32>(res_1),
              Vectorized<c10::qint32>(res_2),
              Vectorized<c10::qint32>(res_3)};
    }

    static Vectorized<c10::quint8> requantize_from_int(
        const int_vec_return_type& inp,
        float multiplier,
        int32_t zero_point) {
      __m512 multiplier_v = _mm512_set1_ps(multiplier);
      __m512i zero_point_v = _mm512_set1_epi32(zero_point);
      return RequantizeAvx512<value_type>(inp, multiplier_v, zero_point_v);
    }

 private:

    // Load from memory constructor
    Vectorized(const void* ptr) {
        vals = _mm512_loadu_si512((const __m512i*)ptr);
    }
};

template <>
Vectorized<c10::quint8> inline maximum(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return a.maximum(b);
}

#else

// NOTE: These are low-performance implementations that we fall back on.

template <
    typename T,
    typename float_vec_return_type_,
    typename int_vec_return_type_,
    int size_>
struct VectorizedQuantizedConverter {
  static constexpr int size() {
    return size_;
  }

  static constexpr int float_num_vecs() {
    return size() / 8;
  }

  static constexpr int int_num_vecs() {
    return size() / 8;
  }

  using float_vec_return_type = float_vec_return_type_;
  using int_vec_return_type = int_vec_return_type_;

  using value_type = typename T::underlying;
  std::array<value_type, size_> vals;

  VectorizedQuantizedConverter(T val) {
    for (const auto i : c10::irange(size())) {
      vals[i] = val.val_;
    }
  }

  VectorizedQuantizedConverter(const void* ptr) {
    memcpy(vals.data(), ptr, sizeof(value_type) * size());
  }

  void store(void* ptr, int count = size()) const {
    memcpy(ptr, vals.data(), count * sizeof(value_type));
  }

  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    float_vec_return_type rv;
    for (const auto i : c10::irange(float_num_vecs())) {
      float tmp_vals[16];
      for (const auto j : c10::irange(16)) {
        tmp_vals[j] = at::native::dequantize_val<T>(
            scale[j], zero_point[j], T(vals[16 * i + j]));
      }
      rv[i] = Vectorized<float>(tmp_vals[0],
          tmp_vals[1],
          tmp_vals[2],
          tmp_vals[3],
          tmp_vals[4],
          tmp_vals[5],
          tmp_vals[6],
          tmp_vals[7],
          tmp_vals[8],
          tmp_vals[9],
          tmp_vals[10],
          tmp_vals[11],
          tmp_vals[12],
          tmp_vals[13],
          tmp_vals[14],
          tmp_vals[15]);
    }
    return rv;
  }

 protected:
  VectorizedQuantizedConverter() {}
};

template <>
struct Vectorized<c10::qint32> : public VectorizedQuantizedConverter<
                                 c10::qint32,
                                 std::array<Vectorized<float>, 1>,
                                 std::array<Vectorized<c10::qint32>, 1>,
                                 16> {
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            16>() {}
  Vectorized(c10::qint32 val)
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            16>(val) {}
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::qint32,
            std::array<Vectorized<float>, 1>,
            std::array<Vectorized<c10::qint32>, 1>,
            16>(ptr) {}

  static Vectorized<c10::qint32> loadu(const void* ptr) {
    return Vectorized<c10::qint32>(ptr);
  }

  static Vectorized<c10::qint32> loadu(const void* ptr, int64_t count) {
    __at_align__ value_type tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
    return loadu(tmp_values);
  }

  static Vectorized<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 16> float_vals;

    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 16], 16);
    }

    at::native::quantize_vec<c10::qint32, /*precision=*/32>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint32*)qvals.data(),
        16 * float_num_vecs());

    return Vectorized<c10::qint32>::loadu(qvals.data());
  }

  Vectorized<c10::qint32> maximum(Vectorized<c10::qint32> b) const {
    Vectorized<c10::qint32> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vectorized<c10::qint32> minimum(Vectorized<c10::qint32> b) const {
    Vectorized<c10::qint32> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const  {
    return maximum(zero_point);
  }


  Vectorized<c10::qint32> relu6(
      Vectorized<c10::qint32> zero_point,
      Vectorized<c10::qint32> q_six) {
    Vectorized<c10::qint32> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
    int_vec_return_type retval;
    for (const auto i : c10::irange(size())) {
      retval[0].vals[i] = vals[i] - b.vals[i];
    }
    return retval;
  }

  static Vectorized<c10::qint32> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    Vectorized<c10::qint32> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] =
          std::nearbyint(static_cast<float>(inp[0].vals[i]) * multiplier) +
          zero_point;
    }
    return retval;
  }
};

template <>
Vectorized<c10::qint32> inline maximum(const Vectorized<c10::qint32>& a, const Vectorized<c10::qint32>& b) {
  return a.maximum(b);
}

template <>
Vectorized<c10::qint32> inline operator*(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  Vectorized<c10::qint32> retval;
  for (const auto i : c10::irange(std::decay_t<decltype(a)>::size())) {
    retval.vals[i] = a.vals[i] * b.vals[i];
  }
  return retval;
}

template <>
Vectorized<c10::qint32> inline operator+(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  Vectorized<c10::qint32> retval;
  for (const auto i : c10::irange(std::decay_t<decltype(a)>::size())) {
    retval.vals[i] = a.vals[i] + b.vals[i];
  }
  return retval;
}

template <>
struct Vectorized<c10::qint8> : public VectorizedQuantizedConverter<
                                c10::qint8,
                                std::array<Vectorized<float>, 4>,
                                std::array<Vectorized<c10::qint32>, 4>,
                                64> {
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>() {}
  Vectorized(c10::qint8 val)
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>(val) {}
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::qint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>(ptr) {}

  static Vectorized<c10::qint8> loadu(const void* ptr) {
    return Vectorized<c10::qint8>(ptr);
  }

  static Vectorized<c10::qint8> loadu(const void* ptr, int64_t count) {
    __at_align__ value_type tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
    return loadu(tmp_values);
  }

  static Vectorized<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 16> float_vals;

    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 16], 16);
    }

    at::native::quantize_vec<c10::qint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint8*)qvals.data(),
        16 * float_num_vecs());

    return Vectorized<c10::qint8>::loadu(qvals.data());
  }

  Vectorized<c10::qint8> maximum(Vectorized<c10::qint8> b) const {
    Vectorized<c10::qint8> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vectorized<c10::qint8> minimum(Vectorized<c10::qint8> b) const {
    Vectorized<c10::qint8> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vectorized<c10::qint8> relu(Vectorized<c10::qint8> zero_point) const {
    return maximum(zero_point);
  }

  Vectorized<c10::qint8> relu6(
      Vectorized<c10::qint8> zero_point,
      Vectorized<c10::qint8> q_six) {
    Vectorized<c10::qint8> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vectorized<c10::qint8> b) const {
    int_vec_return_type retval;
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    for (const auto i : c10::irange(int_num_vecs())) {
      for (const auto j : c10::irange(elem_per_int_vec)) {
        retval[i].vals[j] =
            static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
            static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
      }
    }
    return retval;
  }
  static Vectorized<c10::qint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    constexpr auto min_val = std::numeric_limits<value_type>::min();
    constexpr auto max_val = std::numeric_limits<value_type>::max();
    Vectorized<c10::qint8> retval;
    for (const auto i : c10::irange(int_num_vecs())) {
      for (const auto j : c10::irange(elem_per_int_vec)) {
        int32_t rounded =
            std::nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
            zero_point;
        retval.vals[i * elem_per_int_vec + j] =
            std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
      }
    }
    return retval;
  }
};

template <>
Vectorized<c10::qint8> inline maximum(const Vectorized<c10::qint8>& a, const Vectorized<c10::qint8>& b) {
  return a.maximum(b);
}

template <>
struct Vectorized<c10::quint8> : public VectorizedQuantizedConverter<
                                 c10::quint8,
                                 std::array<Vectorized<float>, 4>,
                                 std::array<Vectorized<c10::qint32>, 4>,
                                 64> {
  Vectorized()
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>() {}
  Vectorized(c10::quint8 val)
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>(val) {}
  Vectorized(const void* ptr)
      : VectorizedQuantizedConverter<
            c10::quint8,
            std::array<Vectorized<float>, 4>,
            std::array<Vectorized<c10::qint32>, 4>,
            64>(ptr) {}

  static Vectorized<c10::quint8> loadu(const void* ptr) {
    return Vectorized<c10::quint8>(ptr);
  }

  static Vectorized<c10::quint8> loadu(const void* ptr, int64_t count) {
    __at_align__ value_type tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, reinterpret_cast<const value_type*>(ptr), count * sizeof(value_type));
    return loadu(tmp_values);
  }

  static Vectorized<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * 16> float_vals;

    for (const auto i : c10::irange(float_num_vecs())) {
      rhs[i].store(&float_vals[i * 16], 16);
    }

    at::native::quantize_vec<c10::quint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::quint8*)qvals.data(),
        16 * float_num_vecs());

    return Vectorized<c10::quint8>::loadu(qvals.data());
  }

  Vectorized<c10::quint8> maximum(Vectorized<c10::quint8> b) const {
    Vectorized<c10::quint8> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vectorized<c10::quint8> minimum(Vectorized<c10::quint8> b) const {
    Vectorized<c10::quint8> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vectorized<c10::quint8> relu(Vectorized<c10::quint8> zero_point) const {
    return maximum(zero_point);
  }


  Vectorized<c10::quint8> relu6(
      Vectorized<c10::quint8> zero_point,
      Vectorized<c10::quint8> q_six) {
    Vectorized<c10::quint8> retval;
    for (const auto i : c10::irange(size())) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vectorized<c10::quint8> b) const {
    int_vec_return_type retval;
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    for (const auto i : c10::irange(int_num_vecs())) {
      for (const auto j : c10::irange(elem_per_int_vec)) {
        retval[i].vals[j] =
            static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
            static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
      }
    }
    return retval;
  }
  static Vectorized<c10::quint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    constexpr auto min_val = std::numeric_limits<value_type>::min();
    constexpr auto max_val = std::numeric_limits<value_type>::max();
    Vectorized<c10::quint8> retval;
    for (const auto i : c10::irange(int_num_vecs())) {
      for (const auto j : c10::irange(elem_per_int_vec)) {
        int32_t rounded =
            std::nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
            zero_point;
        retval.vals[i * elem_per_int_vec + j] =
            std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
      }
    }
    return retval;
  }
};

template <>
Vectorized<c10::quint8> inline maximum(const Vectorized<c10::quint8>& a, const Vectorized<c10::quint8>& b) {
  return a.maximum(b);
}

#endif // defined(CPU_CAPABILITY_AVX512) && !defined(MSVC)

}}}
