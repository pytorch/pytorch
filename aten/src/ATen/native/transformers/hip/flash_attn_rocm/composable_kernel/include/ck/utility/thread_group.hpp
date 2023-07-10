// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "get_id.hpp"

namespace ck {

template <index_t ThreadPerBlock>
struct ThisThreadBlock
{
    static constexpr index_t kNumThread_ = ThreadPerBlock;

    __device__ static constexpr index_t GetNumOfThread() { return kNumThread_; }

    __device__ static constexpr bool IsBelong() { return true; }

    __device__ static index_t GetThreadId() { return get_thread_local_1d_id(); }
};

template <index_t ThreadPerBlock>
struct SubThreadBlock
{
    static constexpr index_t kNumThread_ = ThreadPerBlock;

    __device__ SubThreadBlock(int mwave, int nwave) : mwave_(mwave), nwave_(nwave) {}

    __device__ static constexpr index_t GetNumOfThread() { return kNumThread_; }

    template <typename TupleArg1, typename TupleArg2>
    __device__ constexpr bool IsBelong(const TupleArg1& mwave_range, const TupleArg2& nwave_range)
    {
        // wave_range[I0] inclusive, wave_range[I1] exclusive
        if(mwave_ < mwave_range[I0])
            return false;
        else if(mwave_ >= mwave_range[I1])
            return false;
        else if(nwave_ < nwave_range[I0])
            return false;
        else if(nwave_ >= nwave_range[I1])
            return false;
        else
            return true;
    }

    __device__ static index_t GetThreadId() { return get_thread_local_1d_id(); }

    private:
    index_t mwave_, nwave_;
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
};

} // namespace ck
