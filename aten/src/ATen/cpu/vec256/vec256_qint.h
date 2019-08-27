#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/qint8.h>
#include <c10/util/quint8.h>
#include <c10/util/qint32.h>

// This file defines Vec256<> for the quantized types.
//
//
// Currently, we simply use these classes as efficient converters between
// the quantized types and Vec256<float>, usually in bandwidth-bound cases
// where doing the arithmetic in full-precision is acceptable (e.g.
// elementwise operators).

namespace at {
namespace vec256 {
namespace {

#if defined(__AVX__) && !defined(_MSC_VER)

#include "immintrin.h"

template<>
struct Vec256<c10::qint8> {
    static constexpr int size() {
        return 8;
    }

    using value_type = int8_t;

    // Top 64 bits (starting at &as_vec + 8) are don't-care
    union {
        __m128i as_vec;
        value_type as_ints[16];
    } vals __attribute__((aligned(64)));

    // Broadcast constructor
    Vec256(const c10::qint8& val) {
        value_type uw = val.val_;
        for (int i = 0; i < 16; ++i) {
            vals.as_ints[i] = uw;
        }
    }

    // This is needed because the compiler emits awful code for the default
    // constructor for moving the enum
    Vec256(const Vec256<c10::qint8>& other) {
        vals.as_vec = other.vals.as_vec;
    }

    void store(void* ptr, int count = size()) const {
        memcpy(ptr, &vals.as_ints, count * sizeof(value_type));
    }

    static Vec256<c10::qint8> loadu(const void* ptr) {
        return Vec256<c10::qint8>(ptr);
    }

    Vec256<float> dequantize(Vec256<float> scale, Vec256<float> zero_point) const {
#ifdef __AVX2__
        __m256i int32_vals = _mm256_cvtepi8_epi32(vals.as_vec);
#else // __AVX2__
        __m256i int32_vals;
        for (int i = 0; i < 8; ++i) {
            ((int32_t*)&int32_vals)[i] = vals.as_vec[i];
        }
#endif
        __at_align32__ __m256 float_vals = _mm256_cvtepi32_ps(int32_vals);
        return scale * (Vec256<float>(float_vals) - zero_point);
    }

    static Vec256<c10::qint8> quantize(const Vec256<float>& rhs, float scale, int32_t zero_point) {
        Vec256<c10::qint8> retval;
        // This should hopefully stay in-register with the copy optimized away
        auto rhs_data = (__m256)rhs;
        at::quantize_vec<c10::qint8>(scale, zero_point, (float*)&rhs_data, (c10::qint8*)retval.vals.as_ints, 8);
        return retval;
    }

    void dump() const {
        for (size_t i = 0; i < 8; ++i) {
            std::cout << (int)vals.as_ints[i] << " ";
        }
        std::cout << std::endl;
    }
 private:
    Vec256() {}

    // Load from memory constructor
    Vec256(const void* ptr) {
        memcpy(&vals.as_ints, ptr, size() * sizeof(value_type));
    }
};

template<>
struct Vec256<c10::quint8> {
    static constexpr int size() {
        return 8;
    }

    using value_type = uint8_t;

    // Top 64 bits (starting at &as_vec + 8) are don't-care
    union {
        __m128i as_vec;
        value_type as_ints[16];
    } vals __attribute__((aligned(64)));

    // Broadcast constructor
    Vec256(const c10::quint8& val) {
        value_type uw = val.val_;
        for (int i = 0; i < 16; ++i) {
            vals.as_ints[i] = uw;
        }
    }

    // This is needed because the compiler emits awful code for the default
    // constructor for moving the enum
    Vec256(const Vec256<c10::quint8>& other) {
        vals.as_vec = other.vals.as_vec;
    }

    void store(void* ptr, int count = size()) const {
        memcpy(ptr, &vals.as_ints, count * sizeof(value_type));
    }

    static Vec256<c10::quint8> loadu(const void* ptr) {
        return Vec256<c10::quint8>(ptr);
    }

    Vec256<float> dequantize(Vec256<float> scale, Vec256<float> zero_point) const {
#ifdef __AVX2__
        __m256i int32_vals = _mm256_cvtepu8_epi32(vals.as_vec);
#else // __AVX2__
        __m256i int32_vals;
        for (int i = 0; i < 8; ++i) {
            ((uint32_t*)&int32_vals)[i] = vals.as_vec[i];
        }
#endif
        __m256 float_vals = _mm256_cvtepi32_ps(int32_vals);
        return scale * (Vec256<float>(float_vals) - zero_point);
    }

    static Vec256<c10::quint8> quantize(const Vec256<float>& rhs, float scale, int32_t zero_point) {
        Vec256<c10::quint8> retval;
        // This should hopefully stay in-register with the copy optimized away
        auto rhs_data = (__m256)rhs;
        at::quantize_vec<c10::quint8>(scale, zero_point, (float*)&rhs_data, (c10::quint8*)retval.vals.as_ints, 8);
        return retval;
    }

    void dump() const {
        for (size_t i = 0; i < 8; ++i) {
            std::cout << (int)vals.as_ints[i] << " ";
        }
        std::cout << std::endl;
    }
 private:
    Vec256() {}

    // Load from memory constructor
    Vec256(const void* ptr) {
        memcpy(&vals.as_ints, ptr, size() * sizeof(value_type));
    }
};

template<>
struct Vec256<c10::qint32> {
    static constexpr int size() {
        return 8;
    }

    using value_type = int32_t;

    union {
        __m256i as_vec;
        value_type as_ints[8];
    } vals __attribute__((aligned(64)));

    // Broadcast constructor
    Vec256(const c10::qint32& val) {
        value_type uw = val.val_;
        for (int i = 0; i < 8; ++i) {
            vals.as_ints[i] = uw;
        }
    }

    void store(void* ptr, int count = size()) const {
        memcpy(ptr, &vals.as_ints, count * sizeof(value_type));
    }

    static Vec256<c10::qint32> loadu(const void* ptr) {
        return Vec256<c10::qint32>(ptr);
    }

    Vec256<float> dequantize(Vec256<float> scale, Vec256<float> zero_point) const {
        __m256 float_vals = _mm256_cvtepi32_ps(vals.as_vec);
        return scale * (Vec256<float>(float_vals) - zero_point);
    }

    static Vec256<c10::qint32> quantize(const Vec256<float>& rhs, float scale, int32_t zero_point) {
        Vec256<c10::qint32> retval;
        auto rhs_data = (__m256)rhs;
        at::quantize_vec<c10::qint32, /*precision=*/32>(scale, zero_point, (float*)&rhs_data, (c10::qint32*)retval.vals.as_ints, 8);
        return retval;
    }

    void dump() const {
        for (size_t i = 0; i < 8; ++i) {
            std::cout << (int)vals.as_ints[i] << " ";
        }
        std::cout << std::endl;
    }
 private:
    Vec256() {}

    // Load from memory constructor
    Vec256(const void* ptr) {
        memcpy(&vals.as_ints, ptr, size() * sizeof(value_type));
    }
};

#else

// NOTE: These are low-performance implementations that we fall back on
// if we are not building with AVX2. This may not be an issue, because
// currently for quantization we assume the user has at least AVX512
// installed, so these can simply act as a reference implementation.
//
// If in the future we relax this requirement (AVX2+), we should probably
// revisit these implementations

template <typename T>
struct Vec256QuantizedConverter {
    static constexpr int size() {
        return 8;
    }

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

    Vec256<float> dequantize(Vec256<float> scale, Vec256<float> zero_point) const {
        float float_vals[size()];
        for (int i = 0; i < size(); ++i) {
            float_vals[i] = at::dequantize_val<T>(scale[i], zero_point[i], T(vals[i]));
        }
        return Vec256<float>::loadu(float_vals);
    }
};

template<>
struct Vec256<c10::qint8> : public Vec256QuantizedConverter<c10::qint8> {
    Vec256(c10::qint8 val) : Vec256QuantizedConverter<c10::qint8>(val) {}
    Vec256(const void* ptr) : Vec256QuantizedConverter<c10::qint8>(ptr) {}


    static Vec256<c10::qint8> loadu(const void* ptr) {
        return Vec256<c10::qint8>(ptr);
    }

    static Vec256<c10::qint8> quantize(const Vec256<float>& rhs, float scale, int32_t zero_point) {
        value_type qvals[size()];
        float float_vals[8];

        rhs.store(float_vals, 8);

        at::quantize_vec<c10::qint8>(scale, zero_point, float_vals, (c10::qint8*)qvals, 8);

        return Vec256<c10::qint8>::loadu(qvals);
    }
};

template<>
struct Vec256<c10::quint8> : public Vec256QuantizedConverter<c10::quint8> {
    Vec256(c10::quint8 val) : Vec256QuantizedConverter<c10::quint8>(val) {}
    Vec256(const void* ptr) : Vec256QuantizedConverter<c10::quint8>(ptr) {}

    static Vec256<c10::quint8> loadu(const void* ptr) {
        return Vec256<c10::quint8>(ptr);
    }

    static Vec256<c10::quint8> quantize(const Vec256<float>& rhs, float scale, int32_t zero_point) {
        value_type qvals[size()];
        float float_vals[8];

        rhs.store(float_vals, 8);

        at::quantize_vec<c10::quint8>(scale, zero_point, float_vals, (c10::quint8*)qvals, 8);

        return Vec256<c10::quint8>::loadu(qvals);
    }

};

template<>
struct Vec256<c10::qint32> : public Vec256QuantizedConverter<c10::qint32> {
    Vec256(c10::qint32 val) : Vec256QuantizedConverter<c10::qint32>(val) {}
    Vec256(const void* ptr) : Vec256QuantizedConverter<c10::qint32>(ptr) {}

    static Vec256<c10::qint32> loadu(const void* ptr) {
        return Vec256<c10::qint32>(ptr);
    }

    static Vec256<c10::qint32> quantize(const Vec256<float>& rhs, float scale, int32_t zero_point) {
        value_type qvals[size()];
        float float_vals[8];

        rhs.store(float_vals, 8);

        at::quantize_vec<c10::qint32, /*precision=*/32>(scale, zero_point, float_vals, (c10::qint32*)qvals, 8);

        return Vec256<c10::qint32>::loadu(qvals);
    }
};

#endif // defined(__AVX__) && !defined(_MSC_VER)

}}}