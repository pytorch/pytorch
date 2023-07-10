// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/reduction_enums.hpp"

namespace ck {

struct float_equal_one
{
    template <class T>
    __host__ __device__ inline bool operator()(T x)
    {
        return x <= static_cast<T>(1.0f) and x >= static_cast<T>(1.0f);
    };
};

struct float_equal_zero
{
    template <class T>
    __host__ __device__ inline bool operator()(T x)
    {
        return x <= static_cast<T>(0.0f) and x >= static_cast<T>(0.0f);
    };
};

template <index_t N>
static constexpr __device__ index_t get_shift()
{
    return (get_shift<N / 2>() + 1);
};

template <>
constexpr __device__ index_t get_shift<1>()
{
    return (0);
}

} // namespace ck
