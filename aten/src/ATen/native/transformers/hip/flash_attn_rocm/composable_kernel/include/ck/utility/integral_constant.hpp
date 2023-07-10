// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
    __host__ __device__ constexpr operator value_type() const noexcept { return value; }
    __host__ __device__ constexpr value_type operator()() const noexcept { return value; }
};

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator+(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    return integral_constant<decltype(X + Y), X + Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator-(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y <= X, "wrong!");
    return integral_constant<decltype(X - Y), X - Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator*(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    return integral_constant<decltype(X * Y), X * Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator/(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y > 0, "wrong!");
    return integral_constant<decltype(X / Y), X / Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator%(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y > 0, "wrong!");
    return integral_constant<decltype(X % Y), X % Y>{};
}

} // namespace ck
