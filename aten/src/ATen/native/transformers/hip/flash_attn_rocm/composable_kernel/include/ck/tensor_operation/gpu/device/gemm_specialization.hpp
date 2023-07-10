// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct GemmSpecialization
{
    // Gemm
    Default,
    MPadding,
    NPadding,
    KPadding,
    MNPadding,
    MKPadding,
    NKPadding,
    MNKPadding,
    // Gemm + Gemm
    OPadding,
    MOPadding,
    NOPadding,
    KOPadding,
    MNOPadding,
    MKOPadding,
    NKOPadding,
    MNKOPadding,
};

inline std::string getGemmSpecializationString(const GemmSpecialization& s)
{
    switch(s)
    {
    case GemmSpecialization::Default: return "Default";
    case GemmSpecialization::MPadding: return "MPadding";
    case GemmSpecialization::NPadding: return "NPadding";
    case GemmSpecialization::KPadding: return "KPadding";
    case GemmSpecialization::MNPadding: return "MNPadding";
    case GemmSpecialization::MKPadding: return "MKPadding";
    case GemmSpecialization::NKPadding: return "NKPadding";
    case GemmSpecialization::MNKPadding: return "MNKPadding";
    case GemmSpecialization::OPadding: return "OPadding";
    case GemmSpecialization::MOPadding: return "MOPadding";
    case GemmSpecialization::NOPadding: return "NOPadding";
    case GemmSpecialization::KOPadding: return "KOPadding";
    case GemmSpecialization::MNOPadding: return "MNOPadding";
    case GemmSpecialization::MKOPadding: return "MKOPadding";
    case GemmSpecialization::NKOPadding: return "NKOPadding";
    case GemmSpecialization::MNKOPadding: return "MNKOPadding";
    default: return "Unrecognized specialization!";
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
