// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iterator>
#include <utility>

#include "ck/utility/type.hpp"

namespace ck {

template <typename T>
using iter_value_t = typename std::iterator_traits<remove_cvref_t<T>>::value_type;

template <typename T>
using iter_reference_t = decltype(*std::declval<T&>());

template <typename T>
using iter_difference_t = typename std::iterator_traits<remove_cvref_t<T>>::difference_type;

} // namespace ck
