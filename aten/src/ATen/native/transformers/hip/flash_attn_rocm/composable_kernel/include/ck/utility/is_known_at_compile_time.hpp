// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "integral_constant.hpp"
#include "sequence.hpp"
#include "tuple.hpp"

namespace ck {

template <typename T>
struct is_known_at_compile_time;

template <>
struct is_known_at_compile_time<index_t>
{
    static constexpr bool value = false;
};

template <>
struct is_known_at_compile_time<long_index_t>
{
    static constexpr bool value = false;
};

template <typename T, T X>
struct is_known_at_compile_time<integral_constant<T, X>>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_known_at_compile_time<Sequence<Is...>>
{
    static constexpr bool value = true;
};

template <typename... Ts>
struct is_known_at_compile_time<Tuple<Ts...>>
{
    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return container_reduce(
            Tuple<Ts...>{},
            [](auto x, bool r) {
                return is_known_at_compile_time<remove_cvref_t<decltype(x)>>::value & r;
            },
            true);
    }

    static constexpr bool value = IsKnownAtCompileTime();
};

} // namespace ck
