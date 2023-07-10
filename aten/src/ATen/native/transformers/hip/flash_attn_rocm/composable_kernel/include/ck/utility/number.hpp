// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_NUMBER_HPP
#define CK_NUMBER_HPP

#include "integral_constant.hpp"

namespace ck {

template <index_t N>
using Number = integral_constant<index_t, N>;

template <index_t N>
using LongNumber = integral_constant<long_index_t, N>;

} // namespace ck
#endif
