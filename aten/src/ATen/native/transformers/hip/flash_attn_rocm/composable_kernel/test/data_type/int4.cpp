// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <bitset>
#include <cinttypes>
#include <cstdint>
#include <iomanip>
#include "gtest/gtest.h"
#include <hip/hip_runtime.h>

#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/library/utility/device_memory.hpp"

using ck::int4_t;

TEST(Int4, BaseArithmetic)
{
    int4_t a{1};
    int4_t b{-2};
    EXPECT_EQ(a + a, int4_t{2});
    EXPECT_EQ(a - a, int4_t{0});
    EXPECT_EQ(a + b, int4_t{-1});
    EXPECT_EQ(a - b, int4_t{3});
    EXPECT_EQ(a * a, int4_t{1});
    EXPECT_EQ(a * b, int4_t{-2});
    EXPECT_EQ(b * b, int4_t{4});
    EXPECT_EQ(a / b, int4_t{0});
    a = int4_t{4};
    EXPECT_EQ(a / b, int4_t{-2});
    b = int4_t{2};
    EXPECT_EQ(a % b, int4_t{0});
}

TEST(Int4, NumericLimits)
{
    EXPECT_EQ(ck::NumericLimits<int4_t>::Min(), int4_t{-8});
    EXPECT_EQ(ck::NumericLimits<int4_t>::Max(), int4_t{7});
    EXPECT_EQ(ck::NumericLimits<int4_t>::Lowest(), int4_t{-8});
}

TEST(Int4, MathOpsV2)
{
    int4_t a{4};
    int4_t b{-5};

    EXPECT_EQ(ck::math::abs(a), int4_t{4});
    EXPECT_EQ(ck::math::abs(b), int4_t{5});
    EXPECT_FALSE(ck::math::isnan(b));
}

namespace {

__global__ void copy(const int4_t* src, std::int8_t* dst, ck::index_t N)
{
    ck::index_t tid = ck::get_thread_global_1d_id();

    const int8_t* src_i8 = reinterpret_cast<const int8_t*>(src);

    if(tid < N)
    {
        for(ck::index_t i = tid; i < N; i += ck::get_grid_size())
        {
            dst[i] = src_i8[i];
        }
    }
}

__global__ void copy_with_static_cast(const int4_t* src, std::int8_t* dst, ck::index_t N)
{
    ck::index_t tid = ck::get_thread_global_1d_id();

    if(tid < N)
    {
        for(ck::index_t i = tid; i < N; i += ck::get_grid_size())
        {
            dst[i] = static_cast<std::int8_t>(src[i]);
        }
    }
}

} // anonymous namespace

TEST(Int4, CopyAsI8PositiveValue)
{
    constexpr std::size_t SIZE = 100;
    std::vector<int4_t> h_src_i4(SIZE, 7);
    std::vector<std::int8_t> h_src_i8(SIZE, 7);
    std::vector<std::int8_t> h_dst_i8(SIZE, 0);

    DeviceMem d_src_i4(h_src_i4.size() * sizeof(int4_t));
    DeviceMem d_dst_i8(h_dst_i8.size() * sizeof(std::int8_t));

    d_src_i4.SetZero();
    d_dst_i8.SetZero();

    d_src_i4.ToDevice(h_src_i4.data());

    copy<<<1, 64>>>(reinterpret_cast<const int4_t*>(d_src_i4.GetDeviceBuffer()),
                    reinterpret_cast<std::int8_t*>(d_dst_i8.GetDeviceBuffer()),
                    SIZE);
    hip_check_error(hipDeviceSynchronize());
    d_dst_i8.FromDevice(h_dst_i8.data());

    for(std::size_t i = 0; i < SIZE; ++i)
    {
        EXPECT_EQ(h_src_i8[i], h_dst_i8[i]);
    }
}

TEST(Int4, DISABLED_CopyAsI8NegativeValue)
{
    constexpr std::size_t SIZE = 32;
    std::vector<int4_t> h_src_i4(SIZE, -8);
    std::vector<std::int8_t> h_src_i8(SIZE, -8);
    std::vector<std::int8_t> h_dst_i8(SIZE, 0);

    DeviceMem d_src_i4(h_src_i4.size() * sizeof(int4_t));
    DeviceMem d_dst_i8(h_dst_i8.size() * sizeof(std::int8_t));

    d_src_i4.SetZero();
    d_dst_i8.SetZero();

    d_src_i4.ToDevice(h_src_i4.data());

    copy<<<1, 64>>>(reinterpret_cast<const int4_t*>(d_src_i4.GetDeviceBuffer()),
                    reinterpret_cast<std::int8_t*>(d_dst_i8.GetDeviceBuffer()),
                    SIZE);
    hip_check_error(hipDeviceSynchronize());
    d_dst_i8.FromDevice(h_dst_i8.data());

    for(std::size_t i = 0; i < SIZE; ++i)
    {
        EXPECT_EQ(h_src_i8[i], h_dst_i8[i]);
    }
}

TEST(Int4, CopyAsI8NegativeValueStaticCast)
{
    constexpr std::size_t SIZE = 32;
    std::vector<int4_t> h_src_i4(SIZE, -8);
    std::vector<std::int8_t> h_src_i8(SIZE, -8);
    std::vector<std::int8_t> h_dst_i8(SIZE, 0);

    DeviceMem d_src_i4(h_src_i4.size() * sizeof(int4_t));
    DeviceMem d_dst_i8(h_dst_i8.size() * sizeof(std::int8_t));

    d_src_i4.SetZero();
    d_dst_i8.SetZero();

    d_src_i4.ToDevice(h_src_i4.data());

    copy_with_static_cast<<<1, 64>>>(reinterpret_cast<const int4_t*>(d_src_i4.GetDeviceBuffer()),
                                     reinterpret_cast<std::int8_t*>(d_dst_i8.GetDeviceBuffer()),
                                     SIZE);
    hip_check_error(hipDeviceSynchronize());
    d_dst_i8.FromDevice(h_dst_i8.data());

    for(std::size_t i = 0; i < SIZE; ++i)
    {
        EXPECT_EQ(h_src_i8[i], h_dst_i8[i]);
    }
}

TEST(Int4, DISABLED_BitwiseRepresentation)
{
    using bit8_t = std::bitset<8>;

    int4_t a_i4{3};
    std::int8_t a_i8 = *reinterpret_cast<std::int8_t*>(&a_i4);
    std::int8_t b_i8{3};
#if 0
    std::cout << std::hex << std::showbase << static_cast<int32_t>(a_i8)
              << ", " << static_cast<int32_t>(b_i8) << std::endl;
#endif
    EXPECT_EQ(bit8_t{static_cast<std::uint64_t>(a_i8)}, bit8_t{static_cast<std::uint64_t>(b_i8)});

    a_i4 = int4_t{-3};
    a_i8 = *reinterpret_cast<std::int8_t*>(&a_i4);
    b_i8 = std::int8_t{-3};
#if 0
    std::cout << std::hex << std::showbase << static_cast<int32_t>(a_i8)
              << ", " << static_cast<int32_t>(b_i8) << std::endl;
#endif
    EXPECT_EQ(bit8_t{static_cast<std::uint64_t>(a_i8)}, bit8_t{static_cast<std::uint64_t>(b_i8)});
}

TEST(Int4, BitwiseRepresentationStaticCast)
{
    using bit8_t = std::bitset<8>;

    int4_t a_i4{3};
    std::int8_t a_i8 = static_cast<std::int8_t>(a_i4);
    std::int8_t b_i8{3};
#if 0
    std::cout << std::hex << std::showbase << static_cast<int32_t>(a_i8)
              << ", " << static_cast<int32_t>(b_i8) << std::endl;
#endif
    EXPECT_EQ(bit8_t{static_cast<std::uint64_t>(a_i8)}, bit8_t{static_cast<std::uint64_t>(b_i8)});

    a_i4 = int4_t{-3};
    a_i8 = static_cast<std::int8_t>(a_i4);
    b_i8 = std::int8_t{-3};
#if 0
    std::cout << std::hex << std::showbase << static_cast<int32_t>(a_i8)
              << ", " << static_cast<int32_t>(b_i8) << std::endl;
#endif
    EXPECT_EQ(bit8_t{static_cast<std::uint64_t>(a_i8)}, bit8_t{static_cast<std::uint64_t>(b_i8)});
}
