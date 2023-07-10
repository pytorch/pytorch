// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <iterator>

#include "ck/tensor_description/tensor_descriptor.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(os, " "));
    return os;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& v)
{
    std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(os, " "));
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const ck::TensorDescriptor<Ts...>& desc)
{
    constexpr ck::index_t nDim = ck::remove_cvref_t<decltype(desc)>::GetNumOfDimension();

    os << "{";

    ck::static_for<0, nDim - 1, 1>{}([&](auto i) { os << desc.GetLength(i) << ", "; });

    os << desc.GetLength(ck::Number<nDim - 1>{});

    os << "}";

    return os;
}
