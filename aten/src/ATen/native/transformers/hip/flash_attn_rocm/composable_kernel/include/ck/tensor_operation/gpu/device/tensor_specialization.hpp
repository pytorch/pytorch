// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct TensorSpecialization
{
    Default,
    Packed
};

inline std::string getTensorSpecializationString(const TensorSpecialization& s)
{
    switch(s)
    {
    case TensorSpecialization::Default: return "Default";
    case TensorSpecialization::Packed: return "Packed";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
