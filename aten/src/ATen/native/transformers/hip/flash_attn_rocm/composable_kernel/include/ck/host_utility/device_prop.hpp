// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <map>
#include <hip/hip_runtime.h>

namespace ck {

inline std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }

    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    const std::string raw_name(props.gcnArchName);

    // https://github.com/ROCmSoftwarePlatform/MIOpen/blob/8498875aef84878e04c1eabefdf6571514891086/src/target_properties.cpp#L40
    static std::map<std::string, std::string> device_name_map = {
        {"Ellesmere", "gfx803"},
        {"Baffin", "gfx803"},
        {"RacerX", "gfx803"},
        {"Polaris10", "gfx803"},
        {"Polaris11", "gfx803"},
        {"Tonga", "gfx803"},
        {"Fiji", "gfx803"},
        {"gfx800", "gfx803"},
        {"gfx802", "gfx803"},
        {"gfx804", "gfx803"},
        {"Vega10", "gfx900"},
        {"gfx901", "gfx900"},
        {"10.3.0 Sienna_Cichlid 18", "gfx1030"},
    };

    const auto name = raw_name.substr(0, raw_name.find(':')); // str.substr(0, npos) returns str.

    auto match = device_name_map.find(name);
    if(match != device_name_map.end())
        return match->second;
    return name;
}

} // namespace ck
