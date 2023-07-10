// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/math_v2.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct PassThrough
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<double, double>(double& y, const double& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<int32_t, int32_t>(int32_t& y, const int32_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, float>(bhalf_t& y, const float& x) const
    {
        y = type_convert<bhalf_t>(x);
    }

    template <>
    __host__ __device__ void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<int8_t, int32_t>(int8_t& y, const int32_t& x) const
    {
        y = type_convert<int8_t>(x);
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    __host__ __device__ void operator()<int4_t, int4_t>(int4_t& y, const int4_t& x) const
    {
        y = x;
    }
#endif
};

struct UnaryConvert
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
    }
};

struct Scale
{
    __host__ __device__ Scale(float scale) : scale_(scale) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = scale_ * x;
    };

    __host__ __device__ auto Value() const { return scale_; }

    float scale_;
};

struct ScaleAndResetNaNToMinusInfinity
{
    __host__ __device__ ScaleAndResetNaNToMinusInfinity(float scale) : scale_(scale) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = ck::math::isnan(x) ? -ck::NumericLimits<float>::Infinity() : scale_ * x;
    };

    float scale_;
};

struct UnaryDivide
{
    __host__ __device__ UnaryDivide(const int32_t divider = 1) : divider_(divider) {}

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = x / type_convert<T>(divider_);
    };

    int32_t divider_ = 1;
};

struct UnarySquare
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same_v<T, float> || is_same_v<T, double> || is_same_v<T, int32_t> ||
                          is_same_v<T, int8_t>
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                          || is_same_v<T, int4_t>
#endif
                      ,
                      "Data type is not supported by this operation!");
        y = x * x;
    };
};

struct UnaryAbs
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::abs(x);
    };
};

struct UnarySqrt
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::sqrt(x);
    };
};

struct Relu
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        y = x > 0 ? x : 0;
    }

    template <>
    __host__ __device__ void operator()(bhalf_t& y, const bhalf_t& x) const
    {
        float x_f32 = ck::type_convert<float>(x);
        float y_f32 = x_f32 > 0 ? x_f32 : 0;
        y           = ck::type_convert<bhalf_t>(y_f32);
    }
};

// Y = FastGelu(X)
struct FastGelu
{
    // Fast GeLU
    // https://paperswithcode.com/method/gelu
    // y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    __host__ __device__ static constexpr float GetFastGeLU(float x)
    {
        const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float emu = exp(-u);
        const float cdf = 0.5f + 0.5f * (2.f / (1.f + emu) - 1.f);
        return x * cdf;
    }

    template <typename T>
    static inline constexpr bool is_valid_param_type_v =
        std::is_same_v<T, float> || std::is_same_v<T, half_t> || std::is_same_v<T, bhalf_t> ||
        std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        || std::is_same_v<T, ck::int4_t>
#endif
        ;

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        static_assert(is_valid_param_type_v<Y> && is_valid_param_type_v<X>);

        const float tmp_y = GetFastGeLU(type_convert<float>(x));
        y                 = type_convert<Y>(tmp_y);
    }
};

// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+erf(x/sqrt(2)))
struct Gelu
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = 0.5f * x * (1.f + erf(float(0.70710678118f * x)));
    }

    template <>
    __host__ __device__ void operator()<ck::half_t, ck::half_t>(ck::half_t& y,
                                                                const ck::half_t& x) const
    {
        y = ck::half_t(0.5) * x * (ck::half_t(1) + ck::half_t(erf(float(0.70710678118f * x))));
    }
};

struct Sigmoid
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value,
                      "Data type is not supported by this operation!");

        y = 1 / (ck::type_convert<T>(1) + exp(-x));
    };

    int32_t divider_ = 1;
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
