// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

enum struct ReduceTensorOp
{
    ADD   = 0,
    MUL   = 1,
    MIN   = 2,
    MAX   = 3,
    AMAX  = 4,
    AVG   = 5,
    NORM1 = 6,
    NORM2 = 7,
    // MUL_NO_ZEROS = 8,
};

enum struct NanPropagation
{
    NOT_PROPAGATE_NAN = 0,
    PROPAGATE_NAN     = 1,
};

enum struct ReduceTensorIndices
{
    NO_INDICES        = 0,
    FLATTENED_INDICES = 1,
};

enum struct IndicesType
{
    INDICES_32BIT = 0,
    INDICES_64BIT = 1,
    INDICES_16BIT = 2,
    INDICES_8BIT  = 3,
};

} // namespace ck
