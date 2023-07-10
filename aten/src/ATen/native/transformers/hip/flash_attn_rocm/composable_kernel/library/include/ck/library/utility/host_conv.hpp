// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "host_tensor.hpp"
#include "conv_common.hpp"

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_conv_nchw_kcyx_nkhw(const Tensor<TIn>& in,
                              const Tensor<TWei>& wei,
                              Tensor<TOut>& out,
                              const ConvStrides& conv_strides,
                              const ConvDilations& conv_dilations,
                              const InLeftPads& in_left_pads,
                              const InRightPads&)
{
    constexpr auto I0 = ck::Number<0>{};
    constexpr auto I1 = ck::Number<1>{};

    auto f_nchw = [&](auto n, auto k, auto ho, auto wo) {
        float v = 0;
        for(int c = 0; c < wei.mDesc.GetLengths()[1]; ++c)
        {
            for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[3])
                    {
                        v += ck::type_convert<float>(in(n, c, hi, wi)) *
                             ck::type_convert<float>(wei(k, c, y, x));
                    }
                }
            }
        }
        out(n, k, ho, wo) = ck::type_convert<TOut>(v);
    };

    make_ParallelTensorFunctor(f_nchw,
                               out.mDesc.GetLengths()[0],
                               out.mDesc.GetLengths()[1],
                               out.mDesc.GetLengths()[2],
                               out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
}

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_conv3d_ndhwc_kzyxc_ndhwk(const Tensor<TIn>& in,
                                   const Tensor<TWei>& wei,
                                   Tensor<TOut>& out,
                                   const ConvStrides& conv_strides,
                                   const ConvDilations& conv_dilations,
                                   const InLeftPads& in_left_pads,
                                   const InRightPads&)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    const auto Di     = in.mDesc.GetLengths()[1];
    const auto Hi     = in.mDesc.GetLengths()[2];
    const auto Wi     = in.mDesc.GetLengths()[3];
    const auto Z      = wei.mDesc.GetLengths()[1];
    const auto Y      = wei.mDesc.GetLengths()[2];
    const auto X      = wei.mDesc.GetLengths()[3];
    const auto C      = wei.mDesc.GetLengths()[4];

    auto f_ndhwc = [&](auto n, auto do_tmp, auto ho_tmp, auto wo_tmp, auto k) {
        // do__ must be converted to signed integer, otherwise zmin might be wrong in cases
        // negative values.
        const int do_ = static_cast<int>(do_tmp);
        const int ho  = static_cast<int>(ho_tmp);
        const int wo  = static_cast<int>(wo_tmp);
        const int zmin =
            std::max(0,
                     (in_left_pads[I0] - do_ * conv_strides[I0] + conv_dilations[I0] - 1) /
                         conv_dilations[I0]);
        const int ymin =
            std::max(0,
                     (in_left_pads[I1] - ho * conv_strides[I1] + conv_dilations[I1] - 1) /
                         conv_dilations[I1]);
        const int xmin =
            std::max(0,
                     (in_left_pads[I2] - wo * conv_strides[I2] + conv_dilations[I2] - 1) /
                         conv_dilations[I2]);
        const int zmax =
            std::min(Z, (in_left_pads[I0] - do_ * conv_strides[I0] + Di) / conv_dilations[I0]);
        const int ymax =
            std::min(Y, (in_left_pads[I1] - ho * conv_strides[I1] + Hi) / conv_dilations[I1]);
        const int xmax =
            std::min(X, (in_left_pads[I2] - wo * conv_strides[I2] + Wi) / conv_dilations[I2]);
        const int di_min = do_ * conv_strides[I0] + zmin * conv_dilations[I0] - in_left_pads[I0];
        const int hi_min = ho * conv_strides[I1] + ymin * conv_dilations[I1] - in_left_pads[I1];
        const int wi_min = wo * conv_strides[I2] + xmin * conv_dilations[I2] - in_left_pads[I2];

        double v = 0;

        const TIn* in_n   = in.mData.data() + n * Di * Hi * Wi * C;
        const TWei* wei_k = wei.mData.data() + k * Z * Y * X * C;

        int di = di_min;
        for(int z = zmin; z < zmax; ++z, di += conv_dilations[I0])
        {
            const TIn* in_n_di  = in_n + di * Hi * Wi * C;
            const TWei* wei_k_z = wei_k + z * Y * X * C;
            int hi              = hi_min;

            for(int y = ymin; y < ymax; ++y, hi += conv_dilations[I1])
            {
                const TIn* in_n_di_hi = in_n_di + hi * Wi * C;
                const TWei* wei_k_z_y = wei_k_z + y * X * C;
                int wi                = wi_min;

                for(int x = xmin; x < xmax; ++x, wi += conv_dilations[I2])
                {
                    const TIn* in_n_di_hi_wi = in_n_di_hi + wi * C;
                    const TWei* wei_k_z_y_x  = wei_k_z_y + x * C;

                    for(int c = 0; c < C; ++c)
                    {
                        v += static_cast<const double>(in_n_di_hi_wi[c]) *
                             static_cast<const double>(wei_k_z_y_x[c]);
                    }
                }
            }
        }

        out(n, do_, ho, wo, k) = v;
    };

    make_ParallelTensorFunctor(f_ndhwc,
                               out.mDesc.GetLengths()[0],
                               out.mDesc.GetLengths()[1],
                               out.mDesc.GetLengths()[2],
                               out.mDesc.GetLengths()[3],
                               out.mDesc.GetLengths()[4])(std::thread::hardware_concurrency() - 4);
}
