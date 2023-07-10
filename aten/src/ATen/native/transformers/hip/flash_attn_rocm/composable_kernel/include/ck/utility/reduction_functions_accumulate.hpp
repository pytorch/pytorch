// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"

namespace ck {
namespace detail {

// Check for NaN; guarantee NaNs are NOT propagated to result (i.e., ignore NaNs)
template <typename ReduceOperation, typename AccDataType>
struct AccumulateWithNanIgnore
{
    __device__ static inline void Calculate(AccDataType& accuVal, AccDataType currVal)
    {
        if(!ck::math::isnan(currVal))
        {
            ReduceOperation{}(accuVal, currVal);
        }
    };
};

template <bool PropagateNan, typename ReduceOperation, typename AccDataType>
struct AccumulateWithNanCheck;

// Does not check for NaN; does not guarantee NaNs be propagated to result
// e.g., given that max(a, b) = a > b ? a : b
// then  max(NaN, 1) returns 1
//       max(1, NaN) returns NaN
// since any comparison involving NaNs returns false
template <typename ReduceOperation, typename AccDataType>
struct AccumulateWithNanCheck<false, ReduceOperation, AccDataType>
{
    // cppcheck-suppress constParameter
    __host__ __device__ static inline void Calculate(AccDataType& accuVal, AccDataType currVal)
    {
        ReduceOperation{}(accuVal, currVal);
    };
};

// Check for NaN; guarantees NaNs be propagated to result
template <typename ReduceOperation, typename AccDataType>
struct AccumulateWithNanCheck<true, ReduceOperation, AccDataType>
{
    __host__ __device__ static inline void Calculate(AccDataType& accuVal, AccDataType currVal)
    {
        using ck::math::isnan;

        if(isnan(currVal))
        {
            accuVal = currVal;
        }
        else
        {
            ReduceOperation{}(accuVal, currVal);
        };
    };
};

template <bool PropagateNan, typename ReduceOperation, typename AccDataType, typename IndexDataType>
struct AccumulateWithIndexAndNanCheck;

template <typename ReduceOperation, typename AccDataType, typename IndexDataType>
struct AccumulateWithIndexAndNanCheck<false, ReduceOperation, AccDataType, IndexDataType>
{
    __host__ __device__ static inline void
    // cppcheck-suppress constParameter
    Calculate(AccDataType& accuVal,
              AccDataType currVal,
              IndexDataType& accuIndex,
              IndexDataType currIndex)
    {
        bool changed = false;

        ReduceOperation{}(accuVal, currVal, changed);

        if(changed)
            accuIndex = currIndex;
    };
};

template <typename ReduceOperation, typename AccDataType, typename IndexDataType>
struct AccumulateWithIndexAndNanCheck<true, ReduceOperation, AccDataType, IndexDataType>
{
    // The method is called when the ReduceOperation is indexable and the user asked for indices
    __host__ __device__ static inline void Calculate(AccDataType& accuVal,
                                                     AccDataType currVal,
                                                     IndexDataType& accuIndex,
                                                     IndexDataType currIndex)
    {
        using ck::math::isnan;

        if(isnan(currVal))
        {
            accuVal   = currVal;
            accuIndex = currIndex;
        }
        else
        {
            bool changed = false;

            ReduceOperation{}(accuVal, currVal, changed);

            if(changed)
                accuIndex = currIndex;
        }
    };
};

} // namespace detail
} // namespace ck
