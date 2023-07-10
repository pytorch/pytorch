// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cassert>
#include <vector>
#include <array>
#include <type_traits>

#include "ck/utility/data_type.hpp"

struct NormalizeInInfer
{
    NormalizeInInfer(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T1, typename T2, typename T3, typename T4>
    __host__ __device__ constexpr void operator()(T1& y,
                                                  const T1& x,
                                                  const T2& mean,
                                                  const T2& variance,
                                                  const T3& gamma,
                                                  const T4& beta) const
    {
        static_assert(std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;
        using ck::math::sqrt;

        T2 tmp_x, tmp_y;

        tmp_x = type_convert<T2>(x);

        tmp_y = ((tmp_x - mean) / sqrt(variance + type_convert<T2>(epsilon_))) *
                    type_convert<T2>(gamma) +
                type_convert<T2>(beta);
        y = type_convert<T1>(tmp_y);
    };

    double epsilon_;
};

template <int Rank, int NumReduceDim>
static inline std::array<int, Rank - NumReduceDim>
get_invariant_dims(const std::array<int, NumReduceDim>& reduceDims)
{
    int reduceFlag = 0;

    // flag the bits for the reduceDims
    for(int i = 0; i < NumReduceDim; i++)
    {
        reduceFlag |= 1 << reduceDims[i];
    };

    std::array<int, Rank - NumReduceDim> invariantDims;

    // collect invariant dimensions
    int dim = 0;
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) == 0)
        {
            invariantDims[dim] = i;
            dim++;
        };

    return invariantDims;
};
