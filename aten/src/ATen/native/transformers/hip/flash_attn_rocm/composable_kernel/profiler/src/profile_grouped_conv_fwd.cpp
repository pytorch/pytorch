// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_grouped_conv_fwd_impl.hpp"
#include "profiler_operation_registry.hpp"

namespace {

enum struct ConvLayout
{
    GNHWC_GKYXC_GNHWK, // 0
    NHWGC_GKYXC_NHWGK, // 1
};

enum struct ConvDataType
{
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
};

#define OP_NAME "grouped_conv_fwd"
#define OP_DESC "Grouped Convolution Forward"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
        << "                 1: Input fp16, Weight fp16, Output fp16\n"
        << "                 2: Input bf16, Weight bf16, Output bf16\n"
        << "                 3: Input int8, Weight int8, Output int8)\n"
        << "arg3: tensor layout (0: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, N, Ho, Wo, K]\n"
        << "                     1: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K])\n"
        << "arg4: verification (0: no, 1: yes)\n"
        << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg6: print tensor value (0: no; 1: yes)\n"
        << "arg7: time kernel (0: no, 1: yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format on
}

} // namespace

int profile_grouped_conv_fwd(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 9)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);
    const int num_dim_spatial  = std::stoi(argv[8]);

    // 8 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial
    if(argc != 8 + 1 + 4 + 6 * num_dim_spatial)
    {
        print_helper_msg();
        return 1;
    }

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 9, argv);

    using F32  = float;
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using INT8 = int8_t;

    //
    using GNWC   = ck::tensor_layout::convolution::GNWC;
    using GNHWC  = ck::tensor_layout::convolution::GNHWC;
    using GNDHWC = ck::tensor_layout::convolution::GNDHWC;

    using GKXC   = ck::tensor_layout::convolution::GKXC;
    using GKYXC  = ck::tensor_layout::convolution::GKYXC;
    using GKZYXC = ck::tensor_layout::convolution::GKZYXC;

    using GNWK   = ck::tensor_layout::convolution::GNWK;
    using GNHWK  = ck::tensor_layout::convolution::GNHWK;
    using GNDHWK = ck::tensor_layout::convolution::GNDHWK;

    //
    using NWGC   = ck::tensor_layout::convolution::NWGC;
    using NHWGC  = ck::tensor_layout::convolution::NHWGC;
    using NDHWGC = ck::tensor_layout::convolution::NDHWGC;

    using NWGK   = ck::tensor_layout::convolution::NWGK;
    using NHWGK  = ck::tensor_layout::convolution::NHWGK;
    using NDHWGK = ck::tensor_layout::convolution::NDHWGK;

    constexpr auto I1 = ck::Number<1>{};
    constexpr auto I2 = ck::Number<2>{};
    constexpr auto I3 = ck::Number<3>{};

    auto profile = [&](auto num_dim_spatial_tmp,
                       auto in_layout,
                       auto wei_layout,
                       auto out_layout,
                       auto in_type,
                       auto wei_type,
                       auto out_type) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using OutLayout = decltype(out_layout);

        using InDataType  = decltype(in_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

        bool pass = ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                InLayout,
                                                                WeiLayout,
                                                                OutLayout,
                                                                InDataType,
                                                                WeiDataType,
                                                                OutDataType>(
            do_verification, init_method, do_log, time_kernel, params);

        return pass ? 0 : 1;
    };

    // GNHWC_GKYXC_GNHWK
    if(num_dim_spatial == 1 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, INT8{}, INT8{}, INT8{});
        }
    }
    else if(num_dim_spatial == 2 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, INT8{}, INT8{}, INT8{});
        }
    }
    else if(num_dim_spatial == 3 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, INT8{}, INT8{}, INT8{});
        }
    }
    // NHWGC_GKYXC_NHWGK
    else if(num_dim_spatial == 1 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, INT8{}, INT8{}, INT8{});
        }
    }
    else if(num_dim_spatial == 2 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, INT8{}, INT8{}, INT8{});
        }
    }
    else if(num_dim_spatial == 3 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, INT8{}, INT8{}, INT8{});
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_fwd);
