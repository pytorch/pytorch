// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_PRINT_HPP
#define CK_PRINT_HPP

#include "array.hpp"
#include "statically_indexed_array.hpp"
#include "container_helper.hpp"
#include "sequence.hpp"

namespace ck {

template <typename T>
__host__ __device__ void print_array(const char* s, T a)
{
    constexpr index_t nsize = a.Size();

    printf("%s size %d, {", s, nsize);
    static_for<0, nsize, 1>{}([&a](auto i) constexpr { printf("%d, ", int32_t{a[i]}); });
    printf("}\n");
}

} // namespace ck
#endif
