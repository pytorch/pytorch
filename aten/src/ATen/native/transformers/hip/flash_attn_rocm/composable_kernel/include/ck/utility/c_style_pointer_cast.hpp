// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_C_STYLE_POINTER_CAST_HPP
#define CK_C_STYLE_POINTER_CAST_HPP

#include "type.hpp"
#include "enable_if.hpp"

namespace ck {

template <typename PY,
          typename PX,
          typename enable_if<is_pointer_v<PY> && is_pointer_v<PX>, bool>::type = false>
__host__ __device__ PY c_style_pointer_cast(PX p_x)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"
    return (PY)p_x; // NOLINT(old-style-cast, cast-align)
#pragma clang diagnostic pop
}

} // namespace ck
#endif
