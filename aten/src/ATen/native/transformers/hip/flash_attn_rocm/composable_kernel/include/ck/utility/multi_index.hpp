// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "common_header.hpp"

#if CK_EXPERIMENTAL_USE_DYNAMICALLY_INDEXED_MULTI_INDEX
#include "array_multi_index.hpp"
#else
#include "statically_indexed_array_multi_index.hpp"
#endif
