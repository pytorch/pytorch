// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "statically_indexed_array.hpp"
#include "data_type.hpp"

namespace ck {

template <typename S,
          index_t NX,
          index_t NY,
          typename enable_if<is_scalar_type<S>::value, bool>::type = false>
struct transpose_vectors;

// transpose fp16 2x2
__device__ void transpose_fp16_2x2(const half2_t& x0, const half2_t& x1, half2_t& y0, half2_t& y1)
{
#if 0
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    const vector_type<half_t, 2> vx0{x0}, vx1{x1};
    vector_type<half_t, 2> vy0, vy1;

    vy0.template AsType<half_t>()(I0) = vx0.template AsType<half_t>()[I0];
    vy0.template AsType<half_t>()(I1) = vx1.template AsType<half_t>()[I0];

    vy1.template AsType<half_t>()(I0) = vx0.template AsType<half_t>()[I1];
    vy1.template AsType<half_t>()(I1) = vx1.template AsType<half_t>()[I1];

    y0 = vy0.template AsType<half2_t>()[I0];
    y1 = vy1.template AsType<half2_t>()[I0];
#else
    constexpr int32_t m0 = 0x05040100;
    constexpr int32_t m1 = 0x07060302;

    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
    //                   -- -- -- --     -- -- -- --      -  -  -  -
    //             index  7  6  5  4      3  2  1  0     33 77 44 88
    // index is reversed because of little endianness (least significant bits first)
    y0 = bit_cast<half2_t>(__builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m0));
    y1 = bit_cast<half2_t>(__builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m1));
#endif
}

template <index_t NX, index_t NY>
struct transpose_vectors<half_t, NX, NY>
{
    // we got [NY * NX] amount of S data to be transposed
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S  = half_t;
    using VX = vector_type<half_t, s_per_x>;
    using VY = vector_type<half_t, s_per_y>;

    __device__ void operator()(const StaticallyIndexedArray<const VX&, NX>& vx_tuple,
                               StaticallyIndexedArray<VY&, NY>& vy_tuple)
    {
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};

        static_assert((NX % 2 == 0 && NY % 2 == 0), "wrong!");

        // loop over 2x2 tile and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 2>{}([&](auto iy) {
            static_for<0, NX, 2>{}([&](auto ix) {
                // reference to 2 half2_t data from vx_tuple
                const auto& x_s2_0 = vx_tuple[ix].template AsType<half2_t>()[iy / I2];
                const auto& x_s2_1 = vx_tuple[ix + I1].template AsType<half2_t>()[iy / I2];

                // reference to 2 half2_t data from vy_tuple
                auto& y_s2_0 = vy_tuple(iy).template AsType<half2_t>()(ix / I2);
                auto& y_s2_1 = vy_tuple(iy + I1).template AsType<half2_t>()(ix / I2);

                // transpose
                transpose_fp16_2x2(x_s2_0, x_s2_1, y_s2_0, y_s2_1);
            });
        });
    }
};

// transpose int8 4x4
__device__ void transpose_int8_4x4(const int8x4_t& x0,
                                   const int8x4_t& x1,
                                   const int8x4_t& x2,
                                   const int8x4_t& x3,
                                   int8x4_t& y0,
                                   int8x4_t& y1,
                                   int8x4_t& y2,
                                   int8x4_t& y3)
{
    int32_t t0, t1;
    int32_t z0, z1, z2, z3;
    constexpr int32_t m0 = 0x05010400;
    constexpr int32_t m1 = 0x05040100;
    constexpr int32_t m2 = 0x07060302;
    constexpr int32_t m3 = 0x07030602;

    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
    //                   -- -- -- --     -- -- -- --      -  -  -  -
    //             index  7  6  5  4      3  2  1  0     33 77 44 88
    // index is reversed because of little endianness (least significant bits first)
    t0 = __builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m0);
    t1 = __builtin_amdgcn_perm(bit_cast<int32_t>(x3), bit_cast<int32_t>(x2), m0);
    z0 = __builtin_amdgcn_perm(bit_cast<int32_t>(t1), bit_cast<int32_t>(t0), m1);
    z1 = __builtin_amdgcn_perm(bit_cast<int32_t>(t1), bit_cast<int32_t>(t0), m2);
    t0 = __builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m3);
    t1 = __builtin_amdgcn_perm(bit_cast<int32_t>(x3), bit_cast<int32_t>(x2), m3);
    z2 = __builtin_amdgcn_perm(bit_cast<int32_t>(t1), bit_cast<int32_t>(t0), m1);
    z3 = __builtin_amdgcn_perm(bit_cast<int32_t>(t1), bit_cast<int32_t>(t0), m2);

    y0 = bit_cast<int8x4_t>(z0);
    y1 = bit_cast<int8x4_t>(z1);
    y2 = bit_cast<int8x4_t>(z2);
    y3 = bit_cast<int8x4_t>(z3);
}

template <index_t NX, index_t NY>
struct transpose_vectors<int8_t, NX, NY>
{
    // we got [NY * NX] amount of S data to be transposed
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S  = int8_t;
    using VX = vector_type<int8_t, s_per_x>;
    using VY = vector_type<int8_t, s_per_y>;

    __device__ void operator()(const StaticallyIndexedArray<const VX&, NX>& vx_tuple,
                               StaticallyIndexedArray<VY&, NY>& vy_tuple)
    {
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};
        static constexpr auto I3 = Number<3>{};
        static constexpr auto I4 = Number<4>{};

        static_assert((NX % 4 == 0 && NY % 4 == 0), "wrong!");

        // loop over 4x4 tile and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 4>{}([&](auto iy) {
            static_for<0, NX, 4>{}([&](auto ix) {
                // reference to 4 int8 data from vx_tuple
                const auto& x_s4_0 = vx_tuple[ix].template AsType<int8x4_t>()[iy / I4];
                const auto& x_s4_1 = vx_tuple[ix + I1].template AsType<int8x4_t>()[iy / I4];
                const auto& x_s4_2 = vx_tuple[ix + I2].template AsType<int8x4_t>()[iy / I4];
                const auto& x_s4_3 = vx_tuple[ix + I3].template AsType<int8x4_t>()[iy / I4];

                // reference to 4 int8 data from vy_tuple
                auto& y_s4_0 = vy_tuple(iy).template AsType<int8x4_t>()(ix / I4);
                auto& y_s4_1 = vy_tuple(iy + I1).template AsType<int8x4_t>()(ix / I4);
                auto& y_s4_2 = vy_tuple(iy + I2).template AsType<int8x4_t>()(ix / I4);
                auto& y_s4_3 = vy_tuple(iy + I3).template AsType<int8x4_t>()(ix / I4);

                // transpose
                transpose_int8_4x4(x_s4_0, x_s4_1, x_s4_2, x_s4_3, y_s4_0, y_s4_1, y_s4_2, y_s4_3);
            });
        });
    }
};

} // namespace ck
