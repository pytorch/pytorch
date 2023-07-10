// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <cassert>

#include "ck/utility/common_header.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/reduction_operator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// here, inLengths[] is already shuffled so that lengths of invariant dims are included before those
// of reduce dims
template <index_t Rank, int NumReduceDim>
std::pair<long_index_t, long_index_t> get_2d_lengths(const std::vector<index_t>& inLengths)
{
    static_assert(Rank <= 6, "bigger Rank size not supported!");

    long_index_t invariant_total_length = 1;
    long_index_t reduce_total_length    = 1;

    constexpr int NumInvariantDim = Rank - NumReduceDim;

    for(int i = NumInvariantDim; i < Rank; i++)
        reduce_total_length *= inLengths[i];

    for(int i = 0; i < NumInvariantDim; i++)
        invariant_total_length *= inLengths[i];

    return std::make_pair(invariant_total_length, reduce_total_length);
};

template <index_t Rank, int NumReduceDim>
std::pair<long_index_t, long_index_t> get_2d_lengths(const std::array<index_t, Rank>& inLengths)
{
    static_assert(Rank <= 6, "bigger Rank size not supported!");

    long_index_t invariant_total_length = 1;
    long_index_t reduce_total_length    = 1;

    constexpr int NumInvariantDim = Rank - NumReduceDim;

    for(int i = NumInvariantDim; i < Rank; i++)
        reduce_total_length *= inLengths[i];

    for(int i = 0; i < NumInvariantDim; i++)
        invariant_total_length *= inLengths[i];

    return std::make_pair(invariant_total_length, reduce_total_length);
};

// helper functions using variadic template arguments
template <index_t... Ns>
auto make_tuple_from_array_and_index_seq(const std::vector<index_t>& lengths, Sequence<Ns...>)
{
    return make_tuple(static_cast<index_t>(lengths[Ns])...);
};

template <index_t arraySize>
auto make_tuple_from_array(const std::vector<index_t>& lengths, Number<arraySize>)
{
    static_assert(arraySize >= 1 && arraySize <= 6, "The tensor should have 1 to 6 dimensions");

    constexpr auto index_seq = typename arithmetic_sequence_gen<0, arraySize, 1>::type{};

    return make_tuple_from_array_and_index_seq(lengths, index_seq);
};

template <index_t Rank, index_t NumReduceDim>
std::vector<index_t> shuffle_tensor_dimensions(const std::vector<index_t>& origLengthsStrides,
                                               const std::vector<int>& reduceDims)
{
    std::vector<index_t> newLengthsStrides;

    assert(Rank == origLengthsStrides.size() && NumReduceDim == reduceDims.size());

    int reduceFlag = 0;

    // flag the bits for the reduceDims
    for(int i = 0; i < NumReduceDim; i++)
    {
        reduceFlag |= 1 << reduceDims[i];
    };

    // collect invariant dimensions
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) == 0)
        {
            newLengthsStrides.push_back(origLengthsStrides[i]);
        };

    // collect reduce dimensions
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) > 0)
        {
            newLengthsStrides.push_back(origLengthsStrides[i]);
        };

    return newLengthsStrides;
};

template <index_t Rank, index_t NumReduceDim>
std::array<index_t, Rank>
shuffle_tensor_dimensions(const std::array<index_t, Rank>& origLengthsStrides,
                          const std::array<int, NumReduceDim>& reduceDims)
{
    std::array<index_t, Rank> newLengthsStrides;

    int reduceFlag = 0;

    // flag the bits for the reduceDims
    for(int i = 0; i < NumReduceDim; i++)
    {
        reduceFlag |= 1 << reduceDims[i];
    };

    // collect invariant dimensions
    int pos = 0;
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) == 0)
        {
            newLengthsStrides[pos++] = origLengthsStrides[i];
        };

    // collect reduce dimensions
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) > 0)
        {
            newLengthsStrides[pos++] = origLengthsStrides[i];
        };

    return newLengthsStrides;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
