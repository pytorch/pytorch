// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <numeric>

namespace ck {
template <typename T, typename ForwardIterator, typename Size, typename BinaryOperation>
auto accumulate_n(ForwardIterator first, Size count, T init, BinaryOperation op)
    -> decltype(std::accumulate(first, std::next(first, count), init, op))
{
    return std::accumulate(first, std::next(first, count), init, op);
}
} // namespace ck
