// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct Add
{
    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const half_t& x1) const
    {
        y = x0 + type_convert<half_t>(x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        y = type_convert<half_t>(x0) + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp + x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        y = x0 + x1;
    };
};

struct Subtract
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 - x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 - x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 - x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp - x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        y = x0 - x1;
    };
};

struct Bilinear
{
    Bilinear(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& y, const float& x0, const float& x1) const
    {
        y = alpha_ * x0 + beta_ * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, half_t, half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = type_convert<half_t>(alpha_) * x0 + type_convert<half_t>(beta_) * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, float, half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        y = type_convert<half_t>(alpha_ * x0 + beta_ * ck::type_convert<float>(x1));
    };

    float alpha_;
    float beta_;
};

struct AddRelu
{
    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& y, const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        y             = a > 0.0f ? a : 0.0f;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double, double, double>(double& y, const double& x0, const double& x1) const
    {
        const double a = x0 + x1;
        y              = a > 0.0 ? a : 0.0;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, half_t, half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        const half_t a = x0 + x1;
        y              = a > type_convert<half_t>(0.0f) ? a : type_convert<half_t>(0.0f);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t, float, half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        const float a = x0 + x1;
        y             = a > type_convert<half_t>(0.0f) ? a : type_convert<half_t>(0.0f);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float, float, half_t>(float& y, const float& x0, const half_t& x1) const
    {
        const float a = x0 + type_convert<float>(x1);
        y             = a > 0.0f ? a : 0.0f;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<int, int, int8_t>(int& y, const int& x0, const int8_t& x1) const
    {
        const int8_t a = x0 + x1;
        y              = a > 0 ? a : 0;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t, int8_t, int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        const int8_t a = x0 + x1;
        y              = a > 0 ? a : 0;
    };
};

struct AddHardswish
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        double a = x0 + x1;
        double b = a + 3.0;
        double c = (b > 0) * (b > 6.0 ? 6.0 : b) * a * 0.166667;
        y        = c;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        float a = x0 + x1;
        float b = a + 3.0f;
        float c = (b > 0) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    };
};

// C = A * B
// E = FastGelu(C + D)
struct AddFastGelu
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
        std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>;

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const
    {
        static_assert(is_valid_param_type_v<E> && is_valid_param_type_v<C> &&
                      is_valid_param_type_v<D>);

        const float y = GetFastGeLU(type_convert<float>(c) + type_convert<float>(d));

        e = type_convert<E>(y);
    }

    template <typename D>
    __host__ __device__ constexpr void operator()(float& e, const float& c, const D& d) const
    {
        static_assert(is_valid_param_type_v<D>);

        e = GetFastGeLU(c + type_convert<float>(d));
    }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
