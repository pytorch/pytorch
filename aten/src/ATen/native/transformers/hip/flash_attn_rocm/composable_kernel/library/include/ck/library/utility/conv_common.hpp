// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/tensor_descriptor.hpp"

template <typename... InDesc,
          typename... WeiDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
constexpr auto get_convolution_output_default_4d_tensor_descriptor(
    const ck::TensorDescriptor<InDesc...>& in_desc,
    const ck::TensorDescriptor<WeiDesc...>& wei_desc,
    const ConvStrides& conv_strides,
    const ConvDilations conv_dilations,
    const LeftPads& left_pads,
    const RightPads& right_pads)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    assert(in_desc.GetNumOfDimension() == 4);
    assert(wei_desc.GetNumOfDimension() == 4);
    assert(in_desc.GetLength(I1) == wei_desc.GetLength(I1));

    const auto N  = in_desc.GetLength(I0);
    const auto Hi = in_desc.GetLength(I2);
    const auto Wi = in_desc.GetLength(I3);

    const auto K = wei_desc.GetLength(I0);
    const auto Y = wei_desc.GetLength(I2);
    const auto X = wei_desc.GetLength(I3);

    const auto LeftPadH = left_pads[I0];
    const auto LeftPadW = left_pads[I1];

    const auto RightPadH = right_pads[I0];
    const auto RightPadW = right_pads[I1];

    const auto YEff = (Y - I1) * conv_dilations[I0] + I1;
    const auto XEff = (X - I1) * conv_dilations[I1] + I1;

    const auto Ho = (Hi + LeftPadH + RightPadH - YEff) / conv_strides[I0] + I1;
    const auto Wo = (Wi + LeftPadW + RightPadW - XEff) / conv_strides[I1] + I1;

    return make_naive_tensor_descriptor_packed(make_tuple(N, K, Ho, Wo));
}

template <class InDesc, class WeiDesc, class OutDesc>
constexpr std::size_t
calculate_convolution_flops(const InDesc&, const WeiDesc& wei_desc, const OutDesc& out_desc)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const index_t N  = out_desc.GetLength(I0);
    const index_t K  = out_desc.GetLength(I1);
    const index_t Ho = out_desc.GetLength(I2);
    const index_t Wo = out_desc.GetLength(I3);

    const index_t C = wei_desc.GetLength(I1);
    const index_t Y = wei_desc.GetLength(I2);
    const index_t X = wei_desc.GetLength(I3);

    return std::size_t(2) * N * K * Ho * Wo * C * Y * X;
}
