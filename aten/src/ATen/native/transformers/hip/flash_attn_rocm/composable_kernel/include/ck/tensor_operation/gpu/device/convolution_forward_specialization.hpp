// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

namespace ck {
namespace tensor_operation {
namespace device {

enum struct ConvolutionForwardSpecialization
{
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    OddC,
};

inline std::string getConvForwardSpecializationString(const ConvolutionForwardSpecialization& s)
{
    switch(s)
    {
    case ConvolutionForwardSpecialization::Default: return "Default";
    case ConvolutionForwardSpecialization::Filter1x1Pad0: return "Filter1x1Pad0";
    case ConvolutionForwardSpecialization::Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
    case ConvolutionForwardSpecialization::OddC: return "OddC";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
