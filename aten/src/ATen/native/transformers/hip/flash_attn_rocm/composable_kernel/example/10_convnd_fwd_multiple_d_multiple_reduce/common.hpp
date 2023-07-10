// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_multiple_r_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using BF16 = ck::bhalf_t;
using FP16 = ck::half_t;
using FP32 = float;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using I4 = ck::int4_t;
#endif
using I8  = std::int8_t;
using I32 = std::int32_t;

template <typename ALay, typename BLay, typename DELay, typename RLay>
struct LayoutSetting
{
    using ALayout  = ALay;
    using BLayout  = BLay;
    using DELayout = DELay;
    using RLayout  = RLay;
};

template <ck::index_t NDimSpatial>
struct LayoutSettingSelector;

namespace ctl = ck::tensor_layout::convolution;

template <>
struct LayoutSettingSelector<1> final : LayoutSetting<ctl::GNWC, ctl::GKXC, ctl::GNWK, ctl::GNW>
{
};

template <>
struct LayoutSettingSelector<2> final : LayoutSetting<ctl::GNHWC, ctl::GKYXC, ctl::GNHWK, ctl::GNHW>
{
};

template <>
struct LayoutSettingSelector<3> final
    : LayoutSetting<ctl::GNDHWC, ctl::GKZYXC, ctl::GNDHWK, ctl::GNDHW>
{
};

template <ck::index_t NDimSpatial>
using ALayout = typename LayoutSettingSelector<NDimSpatial>::ALayout;

template <ck::index_t NDimSpatial>
using BLayout = typename LayoutSettingSelector<NDimSpatial>::BLayout;

template <ck::index_t NDimSpatial>
using DELayout = typename LayoutSettingSelector<NDimSpatial>::DELayout;

template <ck::index_t NDimSpatial>
using RLayout = typename LayoutSettingSelector<NDimSpatial>::RLayout;

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

inline void print_help_msg()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ck::utils::conv::ConvParam& problem_size,
                           ExecutionConfig& config)
{
    constexpr int num_execution_config_args =
        3; // arguments for do_verification, init_method, time_kernel
    constexpr int num_conv_param_leading_args = 5; // arguments for num_dim_spatial_, G_, N_, K_, C_

    constexpr int threshold_to_catch_partial_args = 1 + num_execution_config_args;
    constexpr int threshold_to_catch_all_args =
        threshold_to_catch_partial_args + num_conv_param_leading_args;

    if(argc == 1)
    {
        // use default
    }
    // catch only ExecutionConfig arguments
    else if(argc == threshold_to_catch_partial_args)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    // catch both ExecutionConfig & ConvParam arguments
    else if(threshold_to_catch_all_args < argc && ((argc - threshold_to_catch_all_args) % 3 == 0))
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        const ck::index_t num_dim_spatial = std::stoi(argv[4]);
        problem_size                      = ck::utils::conv::parse_conv_param(
            num_dim_spatial, threshold_to_catch_partial_args, argv);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

inline HostTensorDescriptor
make_r0_host_tensor_descriptor(const ck::utils::conv::ConvParam& problem_size)
{
    std::vector<ck::index_t> dimensions{problem_size.G_, problem_size.N_};

    ck::ranges::copy(problem_size.output_spatial_lengths_, std::back_inserter(dimensions));

    return HostTensorDescriptor(dimensions);
}

template <typename Lengths, typename Strides>
void unpack_host_tensor_descriptor(const HostTensorDescriptor& descriptor,
                                   Lengths& lengths,
                                   Strides& strides)
{
    assert(size(descriptor.GetLengths()) == size(lengths));
    std::copy_n(begin(descriptor.GetLengths()), size(descriptor.GetLengths()), begin(lengths));

    assert(size(descriptor.GetStrides()) == size(strides));
    std::copy_n(begin(descriptor.GetStrides()), size(descriptor.GetStrides()), begin(strides));
}
