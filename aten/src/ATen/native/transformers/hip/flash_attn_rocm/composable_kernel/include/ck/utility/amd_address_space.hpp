// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "c_style_pointer_cast.hpp"

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

namespace ck {

enum struct AddressSpaceEnum
{
    Generic,
    Global,
    Lds,
    Sgpr,
    Vgpr,
};

template <typename T>
__device__ T* cast_pointer_to_generic_address_space(T CK_CONSTANT_ADDRESS_SPACE* p)
{
    // cast a pointer in "Constant" address space (4) to "Generic" address space (0)
    // only c-style pointer cast seems be able to be compiled
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T*)p; // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

template <typename T>
__host__ __device__ T CK_CONSTANT_ADDRESS_SPACE* cast_pointer_to_constant_address_space(T* p)
{
    // cast a pointer in "Generic" address space (0) to "Constant" address space (4)
    // only c-style pointer cast seems be able to be compiled
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T CK_CONSTANT_ADDRESS_SPACE*)p; // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

} // namespace ck
