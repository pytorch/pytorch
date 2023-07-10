// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct ConvolutionBackwardDataSpecialization
{
    Default,
    Filter1x1Stride1Pad0,
};

inline std::string
getConvBackwardDataSpecializationString(const ConvolutionBackwardDataSpecialization& s)
{
    switch(s)
    {
    case ConvolutionBackwardDataSpecialization::Default: return "Default";
    case ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0:
        return "FFilter1x1Stride1Pad0";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
