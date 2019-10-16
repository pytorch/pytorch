#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/qint8.h>
#include <c10/util/quint8.h>
#include <c10/util/qint32.h>


#include <array>

// This file defines Vec256<> for the quantized types.
//
//
// Currently, we simply use these classes as efficient converters between
// the quantized types and Vec256<float>, usually in bandwidth-bound cases
// where doing the arithmetic in full-precision is acceptable (e.g.
// elementwise operators).
//
//
// Conversions are as follows:
//  Vec256<qint8> -> 4x Vec256<float>
//  Vec256<quint8> -> 4x Vec256<float>
//  Vec256<qint32> -> 1x Vec256<float>
//
// The size of the returned float vector is specified by the special
// constexpr function float_num_vecs. The type of the value returned
// from dequantize (and expected as an argument to quantize) is
// specified by float_vec_return_type.
//
// When writing kernels with these vectors, it is expected that floating-
// point operations will be carried out in a loop over Vec256<T>::float_num_vecs
// iterations.

namespace at {
namespace vec256 {
namespace {

#if defined(__AVX__) && !defined(_MSC_VER)

#if defined(__AVX2__) && defined(__FMA__)
template <typename T>
__m256i pack_saturate_and_clamp(
    __m256i first,
    __m256i second,
    T min_val,
    T max_val);

template <>
__m256i pack_saturate_and_clamp<int8_t>(
    __m256i first,
    __m256i second,
    int8_t min_val,
    int8_t max_val) {
  __m256i packed_and_sat = _mm256_packs_epi16(first, second);
  return _mm256_max_epi8(
      _mm256_set1_epi8(min_val),
      _mm256_min_epi8(packed_and_sat, _mm256_set1_epi8(max_val)));
}

template <>
__m256i pack_saturate_and_clamp<uint8_t>(
    __m256i first,
    __m256i second,
    uint8_t min_val,
    uint8_t max_val) {
  __m256i packed_and_sat = _mm256_packus_epi16(first, second);
  return _mm256_max_epu8(
      _mm256_set1_epi8(min_val),
      _mm256_min_epu8(packed_and_sat, _mm256_set1_epi8(max_val)));
}
#endif

template <typename T>
inline void __attribute__((always_inline)) QuantizeAvx2(
    const float* src,
    typename T::underlying* dst,
    int len,
    float inverse_scale,
    int64_t zero_point) {
#if defined(__AVX2__) && defined(__FMA__)
  constexpr int VLEN = 8;
  constexpr auto min_val = std::numeric_limits<typename T::underlying>::min();
  constexpr auto max_val = std::numeric_limits<typename T::underlying>::max();
  int i = 0;
  __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  int len_aligned = len / (VLEN * 4) * (VLEN * 4);
  for (; i < len_aligned; i += 4 * VLEN) {
    // x
    __m256 x_vals = _mm256_load_ps(src + i);
    __m256 x_transformed_v =
        _mm256_fmadd_ps(x_vals, inverse_scale_v, _mm256_set1_ps(zero_point));
    // y
    __m256 y_vals = _mm256_load_ps(src + i + VLEN);
    __m256 y_transformed_v =
        _mm256_fmadd_ps(y_vals, inverse_scale_v, _mm256_set1_ps(zero_point));
    // z
    __m256 z_vals = _mm256_load_ps(src + i + 2 * VLEN);
    __m256 z_transformed_v =
        _mm256_fmadd_ps(z_vals, inverse_scale_v, _mm256_set1_ps(zero_point));
    // w
    __m256 w_vals = _mm256_load_ps(src + i + 3 * VLEN);
    __m256 w_transformed_v =
        _mm256_fmadd_ps(w_vals, inverse_scale_v, _mm256_set1_ps(zero_point));

    __m256i x_rounded_v = _mm256_cvtps_epi32(x_transformed_v);
    __m256i y_rounded_v = _mm256_cvtps_epi32(y_transformed_v);
    __m256i z_rounded_v = _mm256_cvtps_epi32(z_transformed_v);
    __m256i w_rounded_v = _mm256_cvtps_epi32(w_transformed_v);

    __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
    __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
    __m256i xyzw_clamped_v = pack_saturate_and_clamp<typename T::underlying>(
        xy_packed_v, zw_packed_v, min_val, max_val);

    xyzw_clamped_v =
        _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), xyzw_clamped_v);
  }

  for (; i < len; ++i) {
    float transformed = zero_point + src[i] * inverse_scale;
    float clipped =
        std::min(std::max(transformed, float(min_val)), float(max_val));
    // Not exactly the same behavior as the vectorized code.
    // The vectorized code above always rounds to even in halfway cases
    // (https://software.intel.com/en-us/node/523819), but std::nearbyint
    // does the same only when the current rounding mode is FE_TONEAREST.
    // However, in practice, this should not be a problem because most cases
    // use the default rounding mode FE_TONEAREST.
    // Note that we cannot implement the same behavior as the vectorized code
    // using std::round because it does rounding away from zero in halfway
    // cases.
    dst[i] = nearbyint(clipped);
  }
#else
  at::quantize_vec<T>(
      1.0f / inverse_scale, zero_point, src, reinterpret_cast<T*>(dst), len);
#endif
}

template<>
struct Vec256<c10::qint8> {
    static constexpr int size() {
        return 32;
    }

    static constexpr int float_num_vecs() {
        return 4;
    }

    using float_vec_return_type = std::array<Vec256<float>, 4>;
    using value_type = typename c10::qint8::underlying;

 private:
    __m256i vals __attribute__((aligned(64)));
 public:

    // Broadcast constructor
    Vec256(const c10::qint8& val) {
        value_type uw = val.val_;
        vals = _mm256_set1_epi8(uw);
    }

    // This is needed because the compiler emits awful code for the default
    // constructor for moving the enum
    Vec256(const Vec256<c10::qint8>& other) {
        vals = other.vals;
    }

    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            _mm256_storeu_si256((__m256i*)ptr, vals);
        }
    }

    static Vec256<c10::qint8> loadu(const void* ptr) {
        return Vec256<c10::qint8>(ptr);
    }

 private:
    __m256i cvtepi8_epi32(__m128i epi8_vals) const {
#ifdef __AVX2__
        return _mm256_cvtepi8_epi32(epi8_vals);
#else  // __AVX2__
        __m128i result_data[2];
        __m128i unpacked1 = _mm_unpacklo_epi8(epi8_vals, epi8_vals);
        __m128i unpacked2 = _mm_unpacklo_epi16(unpacked1, unpacked1);
        __m128i shifted1 = _mm_srli_si128(epi8_vals, 4);
        __m128i shifted2 = _mm_srai_epi32(unpacked2, 24);
        result_data[0] = shifted2;
        __m128i unpacked3 = _mm_unpacklo_epi8(shifted1, shifted1);
        __m128i unpacked4 = _mm_unpacklo_epi16(unpacked3, unpacked3);
        __m128i shifted3 = _mm_srai_epi32(unpacked4, 24);
        result_data[1] = shifted3;
        return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_data));
#endif
    }

 public:
  float_vec_return_type dequantize(
      Vec256<float> scale,
      Vec256<float> zero_point,
      Vec256<float> scale_neg_zp_premul) const {
    __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
    __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
    __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
    __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

    __m256 float_val0 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val0));
    __m256 float_val1 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val1));
    __m256 float_val2 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val2));
    __m256 float_val3 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val3));

#if defined(__AVX2__) && defined(__FMA__)
    auto val0 =
        vec256::fmadd(scale, Vec256<float>(float_val0), scale_neg_zp_premul);
    auto val1 =
        vec256::fmadd(scale, Vec256<float>(float_val1), scale_neg_zp_premul);
    auto val2 =
        vec256::fmadd(scale, Vec256<float>(float_val2), scale_neg_zp_premul);
    auto val3 =
        vec256::fmadd(scale, Vec256<float>(float_val3), scale_neg_zp_premul);
#else
    auto val0 = scale * (Vec256<float>(float_val0) - zero_point);
    auto val1 = scale * (Vec256<float>(float_val1) - zero_point);
    auto val2 = scale * (Vec256<float>(float_val2) - zero_point);
    auto val3 = scale * (Vec256<float>(float_val3) - zero_point);
#endif
    return {val0, val1, val2, val3};
  }

  static Vec256<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    auto* rhs_data = (float*)rhs.data();
    int8_t quantized_values[32];
    QuantizeAvx2<c10::qint8>(
        rhs_data, quantized_values, 32, inverse_scale, zero_point);
    return Vec256<c10::qint8>::loadu(quantized_values);
  }

  Vec256<c10::qint8> maximum(Vec256<c10::qint8> b) const {
#ifdef __AVX2__
      return _mm256_max_epi8(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      int8_t int_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      int8_t b_vals[size()];
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&b_vals), b.vals);
      int8_t result_vals[size()];
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::max<int8_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    Vec256<c10::qint8> relu(Vec256<c10::qint8> zero_point) const {
        return maximum(zero_point);
    }

    Vec256<c10::qint8> relu6(
        Vec256<c10::qint8> zero_point,
        Vec256<c10::qint8> q_six) {
#ifdef __AVX2__
      return _mm256_min_epi8(
          _mm256_max_epi8(vals, zero_point.vals), q_six.vals);
#else
      // Pray the compiler can autovectorize this
      int8_t int_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      int8_t zero_point_vals[size()];
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
      int8_t q_six_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
      int8_t result_vals[size()];
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<int8_t>(
            std::max<int8_t>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    void dump() const {
        for (size_t i = 0; i < size(); ++i) {
            std::cout << (int)((value_type*)&vals)[i] << " ";
        }
        std::cout << std::endl;
    }
 private:
    Vec256() {}

    Vec256(__m256i vals_) : vals(vals_) {}

    // Load from memory constructor
    Vec256(const void* ptr) {
        vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
};

template <>
Vec256<c10::qint8> inline maximum(const Vec256<c10::qint8>& a, const Vec256<c10::qint8>& b) {
  return a.maximum(b);
}

template<>
struct Vec256<c10::quint8> {
    static constexpr int size() {
        return 32;
    }

    static constexpr int float_num_vecs() {
        return 4;
    }

    using float_vec_return_type = std::array<Vec256<float>, 4>;
    using value_type = typename c10::quint8::underlying;

 private:
    __m256i vals __attribute__((aligned(64)));

 public:
    // Broadcast constructor
    Vec256(const c10::quint8& val) {
        value_type uw = val.val_;
        vals = _mm256_set1_epi8(uw);
    }

    Vec256(const Vec256<c10::quint8>& other) {
        vals = other.vals;
    }

    void store(void* ptr, int count = size()) const {
        if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
        } else {
            _mm256_storeu_si256((__m256i*)ptr, vals);
        }
    }

    static Vec256<c10::quint8> loadu(const void* ptr) {
        return Vec256<c10::quint8>(ptr);
    }

 private:
    __m256i cvtepu8_epi32(__m128i epu8_vals) const {
#ifdef __AVX2__
        return _mm256_cvtepu8_epi32(epu8_vals);
#else  // __AVX2__
        __m128i result_data[2];
        __m128i zeros = _mm_setzero_si128();
        __m128i unpacked1 = _mm_unpacklo_epi8(epu8_vals, zeros);
        __m128i unpacked2 = _mm_unpacklo_epi16(unpacked1, zeros);
        result_data[0] = unpacked2;
        __m128i shifted = _mm_srli_si128(epu8_vals, 4);
        __m128i unpacked3 = _mm_unpacklo_epi8(shifted, zeros);
        __m128i unpacked4 = _mm_unpacklo_epi16(unpacked3, zeros);
        result_data[1] = unpacked4;
        return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_data));
#endif
    }

 public:
  float_vec_return_type dequantize(
      Vec256<float> scale,
      Vec256<float> zero_point,
      Vec256<float> scale_zp_premul) const {
    __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
    __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
    __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
    __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

    __m256 float_val0 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val0));
    __m256 float_val1 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val1));
    __m256 float_val2 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val2));
    __m256 float_val3 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val3));

#if defined(__AVX2__) && defined(__FMA__)
    auto val0 =
        vec256::fmadd(scale, Vec256<float>(float_val0), scale_zp_premul);
    auto val1 =
        vec256::fmadd(scale, Vec256<float>(float_val1), scale_zp_premul);
    auto val2 =
        vec256::fmadd(scale, Vec256<float>(float_val2), scale_zp_premul);
    auto val3 =
        vec256::fmadd(scale, Vec256<float>(float_val3), scale_zp_premul);
#else
    auto val0 = scale * (Vec256<float>(float_val0) - zero_point);
    auto val1 = scale * (Vec256<float>(float_val1) - zero_point);
    auto val2 = scale * (Vec256<float>(float_val2) - zero_point);
    auto val3 = scale * (Vec256<float>(float_val3) - zero_point);
#endif
    return {val0, val1, val2, val3};
  }

  static Vec256<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    auto* rhs_data = (float*)rhs.data();
    uint8_t quantized_values[32];
    QuantizeAvx2<c10::quint8>(
        rhs_data, quantized_values, 32, inverse_scale, zero_point);
    return Vec256<c10::quint8>::loadu(quantized_values);
  }

  Vec256<c10::quint8> maximum(Vec256<c10::quint8> b) const {
#ifdef __AVX2__
      return _mm256_max_epu8(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      uint8_t int_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      uint8_t b_vals[size()];
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&b_vals), b.vals);
      uint8_t result_vals[size()];
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::max<uint8_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    Vec256<c10::quint8> relu(Vec256<c10::quint8> zero_point) const {
        return maximum(zero_point);
    }

    Vec256<c10::quint8> relu6(
        Vec256<c10::quint8> zero_point,
        Vec256<c10::quint8> q_six) {
#ifdef __AVX2__
      return _mm256_min_epu8(
          _mm256_max_epu8(vals, zero_point.vals), q_six.vals);
#else
      // Pray the compiler can autovectorize this
      uint8_t int_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      uint8_t zero_point_vals[size()];
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
      uint8_t q_six_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
      uint8_t result_vals[size()];
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<uint8_t>(
            std::max<uint8_t>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    void dump() const {
        for (size_t i = 0; i < size(); ++i) {
            std::cout << (int)((value_type*)&vals)[i] << " ";
        }
        std::cout << std::endl;
    }
 private:
    Vec256() {}

    Vec256(__m256i vals_) : vals(vals_) {}

    // Load from memory constructor
    Vec256(const void* ptr) {
        vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
};

template <>
Vec256<c10::quint8> inline maximum(const Vec256<c10::quint8>& a, const Vec256<c10::quint8>& b) {
  return a.maximum(b);
}

template<>
struct Vec256<c10::qint32> {
    static constexpr int size() {
        return 8;
    }

    static constexpr int float_num_vecs() {
        return 1;
    }

    using float_vec_return_type = std::array<Vec256<float>, 1>;
    using value_type = c10::qint32::underlying;

 private:
    __m256i vals __attribute__((aligned(64)));
 public:

    // Broadcast constructor
    Vec256(const c10::qint32& val) {
        value_type uw = val.val_;
        vals = _mm256_set1_epi32(uw);
    }

    void store(void* ptr, int count = size()) const {
      if (count != size()) {
        memcpy(ptr, &vals, count * sizeof(value_type));
      } else {
        _mm256_storeu_si256((__m256i*)ptr, vals);
      }
    }

    static Vec256<c10::qint32> loadu(const void* ptr) {
        return Vec256<c10::qint32>(ptr);
    }

    float_vec_return_type dequantize(
        Vec256<float> scale,
        Vec256<float> zero_point,
        Vec256<float> scale_zp_premul) const {
      __m256 float_vals = _mm256_cvtepi32_ps(vals);
#if defined(__AVX2__) && defined(__FMA__)
      return {vec256::fmadd(scale, Vec256<float>(float_vals), scale_zp_premul)};
#else
      return {scale * (Vec256<float>(float_vals) - zero_point)};
#endif
    }

    static Vec256<c10::qint32> quantize(
        const float_vec_return_type& rhs,
        float scale,
        int32_t zero_point,
        float inverse_scale) {
      Vec256<c10::qint32> retval;
      auto rhs_data = (__m256)rhs[0];
      at::quantize_vec<c10::qint32, /*precision=*/32>(
          scale, zero_point, (float*)&rhs_data, (c10::qint32*)&retval.vals, 8);
      return retval;
    }

    Vec256<c10::qint32> maximum(Vec256<c10::qint32> b) const {
#ifdef __AVX2__
      return _mm256_max_epi32(vals, b.vals);
#else
      // Pray the compiler can autovectorize this
      int32_t int_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      int32_t b_vals[size()];
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&b_vals), b.vals);
      int32_t result_vals[size()];
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::max<int32_t>(int_vals[i], b_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    Vec256<c10::qint32> relu(Vec256<c10::qint32> zero_point) const {
        return maximum(zero_point);
    }

    Vec256<c10::qint32> relu6(
        Vec256<c10::qint32> zero_point,
        Vec256<c10::qint32> q_six) {
#ifdef __AVX2__
      return _mm256_min_epi32(
          _mm256_max_epi32(vals, zero_point.vals), q_six.vals);
#else
      // Pray the compiler can autovectorize this
      int32_t int_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
      int32_t zero_point_vals[size()];
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
      int32_t q_six_vals[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
      int32_t result_vals[size()];
      for (size_t i = 0; i < size(); ++i) {
        result_vals[i] = std::min<int32_t>(
            std::max<int32_t>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
      }
      return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
#endif
    }

    void dump() const {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << ((int32_t*)&vals)[i] << " ";
        }
        std::cout << std::endl;
    }
 private:
    Vec256() {}

    Vec256(__m256i vals_) : vals(vals_) {}

    // Load from memory constructor
    Vec256(const void* ptr) {
      vals = _mm256_loadu_si256((const __m256i*)ptr);
    }
};

template <>
Vec256<c10::qint32> inline maximum(const Vec256<c10::qint32>& a, const Vec256<c10::qint32>& b) {
  return a.maximum(b);
}

#else

// NOTE: These are low-performance implementations that we fall back on
// if we are not building with AVX2. This may not be an issue, because
// currently for quantization we assume the user has at least AVX512
// installed, so these can simply act as a reference implementation.
//
// If in the future we relax this requirement (AVX2+), we should probably
// revisit these implementations

template <typename T, typename float_vec_return_type_, int size_>
struct Vec256QuantizedConverter {
  static constexpr int size() {
    return size_;
  }

  static constexpr int float_num_vecs() {
    return size() / 8;
  }

  using float_vec_return_type = float_vec_return_type_;

  using value_type = typename T::underlying;
  value_type vals[size()];

  Vec256QuantizedConverter(T val) {
    for (size_t i = 0; i < size(); ++i) {
      vals[i] = val.val_;
    }
  }

  Vec256QuantizedConverter(const void* ptr) {
    memcpy(vals, ptr, sizeof(value_type) * size());
  }

  void store(void* ptr, int count = size()) const {
    memcpy(ptr, vals, count * sizeof(value_type));
  }

  float_vec_return_type dequantize(
      Vec256<float> scale,
      Vec256<float> zero_point,
      Vec256<float> scale_zp_premul) const {
    float_vec_return_type rv;
    for (int i = 0; i < float_num_vecs(); ++i) {
      for (int j = 0; j < 8; ++j) {
        rv[i][j] =
            at::dequantize_val<T>(scale[j], zero_point[j], T(vals[8 * i + j]));
      }
    }
    return rv;
  }

  void dump() const {
      for (int i = 0; i < size(); ++i) {
          std::cout << vals[i] << " ";
      }
      std::cout << std::endl;
  }

 protected:
  Vec256QuantizedConverter() {}
};

template <>
struct Vec256<c10::qint8> : public Vec256QuantizedConverter<
                                c10::qint8,
                                std::array<Vec256<float>, 4>,
                                32> {
  Vec256(c10::qint8 val)
      : Vec256QuantizedConverter<c10::qint8, std::array<Vec256<float>, 4>, 32>(
            val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<c10::qint8, std::array<Vec256<float>, 4>, 32>(
            ptr) {}

  static Vec256<c10::qint8> loadu(const void* ptr) {
    return Vec256<c10::qint8>(ptr);
  }

  static Vec256<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    value_type qvals[size()];
    float float_vals[float_num_vecs() * 8];

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(float_vals + i * 8, 8);
    }

    at::quantize_vec<c10::qint8>(
        scale,
        zero_point,
        float_vals,
        (c10::qint8*)qvals,
        8 * float_num_vecs());

    return Vec256<c10::qint8>::loadu(qvals);
  }

  Vec256<c10::qint8> maximum(Vec256<c10::qint8> b) const {
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint8> relu(Vec256<c10::qint8> zero_point) const {
    return maximum(zero_point);
  }

  Vec256<c10::qint8> relu6(
      Vec256<c10::qint8> zero_point,
      Vec256<c10::qint8> q_six) {
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

 private:
  Vec256() {}
};

template <>
Vec256<c10::qint8> inline maximum(const Vec256<c10::qint8>& a, const Vec256<c10::qint8>& b) {
  return a.maximum(b);
}

template <>
struct Vec256<c10::quint8> : public Vec256QuantizedConverter<
                                 c10::quint8,
                                 std::array<Vec256<float>, 4>,
                                 32> {
  Vec256(c10::quint8 val)
      : Vec256QuantizedConverter<c10::quint8, std::array<Vec256<float>, 4>, 32>(
            val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<c10::quint8, std::array<Vec256<float>, 4>, 32>(
            ptr) {}

  static Vec256<c10::quint8> loadu(const void* ptr) {
    return Vec256<c10::quint8>(ptr);
  }

  static Vec256<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    value_type qvals[size()];
    float float_vals[float_num_vecs() * 8];

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(float_vals + i * 8, 8);
    }

    at::quantize_vec<c10::quint8>(
        scale,
        zero_point,
        float_vals,
        (c10::quint8*)qvals,
        8 * float_num_vecs());

    return Vec256<c10::quint8>::loadu(qvals);
  }

  Vec256<c10::quint8> maximum(Vec256<c10::quint8> b) const {
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::quint8> relu(Vec256<c10::quint8> zero_point) const {
    return maximum(zero_point);
  }


  Vec256<c10::quint8> relu6(
      Vec256<c10::quint8> zero_point,
      Vec256<c10::quint8> q_six) {
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

 private:
  Vec256() {}
};

template <>
Vec256<c10::quint8> inline maximum(const Vec256<c10::quint8>& a, const Vec256<c10::quint8>& b) {
  return a.maximum(b);
}

template <>
struct Vec256<c10::qint32> : public Vec256QuantizedConverter<
                                 c10::qint32,
                                 std::array<Vec256<float>, 1>,
                                 8> {
  Vec256(c10::qint32 val)
      : Vec256QuantizedConverter<c10::qint32, std::array<Vec256<float>, 1>, 8>(
            val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<c10::qint32, std::array<Vec256<float>, 1>, 8>(
            ptr) {}

  static Vec256<c10::qint32> loadu(const void* ptr) {
    return Vec256<c10::qint32>(ptr);
  }

  static Vec256<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    value_type qvals[size()];
    float float_vals[float_num_vecs() * 8];

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(float_vals + i * 8, 8);
    }

    at::quantize_vec<c10::qint32, /*precision=*/32>(
        scale,
        zero_point,
        float_vals,
        (c10::qint32*)qvals,
        8 * float_num_vecs());

    return Vec256<c10::qint32>::loadu(qvals);
  }

  Vec256<c10::qint32> maximum(Vec256<c10::qint32> b) const {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint32> relu(Vec256<c10::qint32> zero_point) const  {
    return maximum(zero_point);
  }


  Vec256<c10::qint32> relu6(
      Vec256<c10::qint32> zero_point,
      Vec256<c10::qint32> q_six) {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

 private:
  Vec256() {}
};

template <>
Vec256<c10::qint32> inline maximum(const Vec256<c10::qint32>& a, const Vec256<c10::qint32>& b) {
  return a.maximum(b);
}

#endif // defined(__AVX__) && !defined(_MSC_VER)

}}}
