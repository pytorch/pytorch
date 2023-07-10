// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace ck {
namespace ranges {
template <typename InputRange, typename OutputIterator>
auto copy(InputRange&& range, OutputIterator iter)
    -> decltype(std::copy(std::begin(std::forward<InputRange>(range)),
                          std::end(std::forward<InputRange>(range)),
                          iter))
{
    return std::copy(std::begin(std::forward<InputRange>(range)),
                     std::end(std::forward<InputRange>(range)),
                     iter);
}

template <typename T, typename OutputRange>
auto fill(OutputRange&& range, const T& init)
    -> std::void_t<decltype(std::fill(std::begin(std::forward<OutputRange>(range)),
                                      std::end(std::forward<OutputRange>(range)),
                                      init))>
{
    std::fill(std::begin(std::forward<OutputRange>(range)),
              std::end(std::forward<OutputRange>(range)),
              init);
}

template <typename InputRange, typename OutputIterator, typename UnaryOperation>
auto transform(InputRange&& range, OutputIterator iter, UnaryOperation unary_op)
    -> decltype(std::transform(std::begin(range), std::end(range), iter, unary_op))
{
    return std::transform(std::begin(range), std::end(range), iter, unary_op);
}

} // namespace ranges
} // namespace ck
