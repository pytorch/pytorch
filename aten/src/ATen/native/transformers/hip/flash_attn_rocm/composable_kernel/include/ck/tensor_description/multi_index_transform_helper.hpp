// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform.hpp"

namespace ck {

template <typename LowLength>
__host__ __device__ constexpr auto make_pass_through_transform(const LowLength& low_length)
{
    return PassThrough<LowLength>{low_length};
}

template <typename LowLength, typename LeftPad, typename RightPad, bool SkipIsValidCheck = false>
__host__ __device__ constexpr auto
make_pad_transform(const LowLength& low_length,
                   const LeftPad& left_pad,
                   const RightPad& right_pad,
                   integral_constant<bool, SkipIsValidCheck> = integral_constant<bool, false>{})
{
    return Pad<LowLength, LeftPad, RightPad, SkipIsValidCheck>{low_length, left_pad, right_pad};
}

template <typename LowLength, typename LeftPadLength, bool SkipIsValidCheck = false>
__host__ __device__ constexpr auto make_left_pad_transform(
    const LowLength& low_length,
    const LeftPadLength& left_pad,
    integral_constant<bool, SkipIsValidCheck> = integral_constant<bool, false>{})
{
    return LeftPad<LowLength, LeftPadLength, SkipIsValidCheck>{low_length, left_pad};
}

template <typename LowLength, typename RightPadLength, bool SkipIsValidCheck = false>
__host__ __device__ constexpr auto make_right_pad_transform(
    const LowLength& low_length,
    const RightPadLength& right_pad,
    integral_constant<bool, SkipIsValidCheck> = integral_constant<bool, false>{})
{
    return RightPad<LowLength, RightPadLength, SkipIsValidCheck>{low_length, right_pad};
}

template <typename UpLengths,
          typename Coefficients,
          typename enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
__host__ __device__ constexpr auto make_embed_transform(const UpLengths& up_lengths,
                                                        const Coefficients& coefficients)
{
    return Embed<UpLengths, Coefficients>{up_lengths, coefficients};
}

template <typename LowLengths>
__host__ __device__ constexpr auto make_merge_transform(const LowLengths& low_lengths)
{
#if CK_EXPERIMENTAL_MERGE_USE_MAGIC_DIVISION
    return make_merge_transform_v2_magic_division(low_lengths);
#else
    return make_merge_transform_v1_carry_check(low_lengths);
#endif
}

template <typename LowLengths>
__host__ __device__ constexpr auto
make_merge_transform_v1_carry_check(const LowLengths& low_lengths)
{
    return Merge_v1_carry_check<LowLengths>{low_lengths};
}

template <typename LowLengths>
__host__ __device__ constexpr auto
make_merge_transform_v2_magic_division(const LowLengths& low_lengths)
{
#if 1
    return Merge_v2_magic_division<LowLengths>{low_lengths};
#else
    return Merge_v2r2_magic_division<LowLengths>{low_lengths};
#endif
}

template <typename LowLengths>
__host__ __device__ constexpr auto
make_merge_transform_v3_division_mod(const LowLengths& low_lengths)
{
    return Merge_v3_division_mod<LowLengths>{low_lengths};
}

template <typename UpLengths, bool Use24BitIntegerCalculation = false>
__host__ __device__ constexpr auto make_unmerge_transform(
    const UpLengths& up_lengths,
    integral_constant<bool, Use24BitIntegerCalculation> = integral_constant<bool, false>{})
{
    return UnMerge<UpLengths, Use24BitIntegerCalculation>{up_lengths};
}

template <typename LowerIndex>
__host__ __device__ constexpr auto make_freeze_transform(const LowerIndex& low_idx)
{
    return Freeze<LowerIndex>{low_idx};
}

template <typename UpperIndex>
__host__ __device__ constexpr auto make_insert_transform(const UpperIndex& up_idx)
{
    return Insert<UpperIndex>{up_idx};
}

template <typename LowLength, typename SliceBegin, typename SliceEnd>
__host__ __device__ constexpr auto make_slice_transform(const LowLength& low_length,
                                                        const SliceBegin& slice_begin,
                                                        const SliceEnd& slice_end)
{
    return Slice<LowLength, SliceBegin, SliceEnd>{low_length, slice_begin, slice_end};
}

template <typename VectorSize, typename UpLength>
__host__ __device__ constexpr auto make_vectorize_transform(const VectorSize& vector_size,
                                                            const UpLength& up_length)
{
    return Vectorize<VectorSize, UpLength>{vector_size, up_length};
}

template <typename Modulus, typename UpLength>
__host__ __device__ constexpr auto make_modulo_transform(const Modulus& modulus,
                                                         const UpLength& up_length)
{
    return Modulo<Modulus, UpLength>{modulus, up_length};
}
} // namespace ck
