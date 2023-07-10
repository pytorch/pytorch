// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef NAIVE_CONV_FWD_HPP
#define NAIVE_CONV_FWD_HPP

namespace ck {
namespace ref {

/*
 * \brief naive implementation of 3D convolution. Layout is (NDHWC, KZYXC, NDHWK).
 *
 * \param N number of batches
 * \param K number of filters
 * \param C number of channels of weight
 * \param (Di, Hi, Wi) depth, height and width dimension of data
 * \param (Z, Y, X) depth, height and width dimensions of weights
 * \param (Do, Ho, Wo) depth, height and width dimension of output
 * \param (stride_z, stride_y, stride_x) strides
 * \param (dilation_z, dilation_y, dilation_x) dilations
 * \param (pad_z, pad_y, pad_x) pads
 */
template <typename TIn,
          typename TWei,
          typename TOut,
          typename TAcc,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
__global__ void naive_conv_fwd_ndhwc_kzyxc_ndhwk(const TIn* __restrict__ p_in,
                                                 const TWei* __restrict__ p_wei,
                                                 TOut* __restrict__ p_out,
                                                 index_t N,
                                                 index_t K,
                                                 index_t C,
                                                 index_t Di,
                                                 index_t Hi,
                                                 index_t Wi,
                                                 index_t Z,
                                                 index_t Y,
                                                 index_t X,
                                                 index_t Do,
                                                 index_t Ho,
                                                 index_t Wo,
                                                 index_t stride_z,
                                                 index_t stride_y,
                                                 index_t stride_x,
                                                 index_t dilation_z,
                                                 index_t dilation_y,
                                                 index_t dilation_x,
                                                 index_t pad_z,
                                                 index_t pad_y,
                                                 index_t pad_x)
{
    const index_t tid                = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t num_threads        = blockDim.x * gridDim.x;
    const long_index_t output_length = N * Do * Ho * Wo * K;

    const index_t out_strides[] = {Do * Ho * Wo * K, Ho * Wo * K, Wo * K, K};
    const index_t in_strides[]  = {Di * Hi * Wi * C, Hi * Wi * C, Wi * C, C};
    const index_t wei_strides[] = {Z * Y * X * C, Y * X * C, X * C, C};

    constexpr auto in_op  = InElementwiseOperation{};
    constexpr auto wei_op = WeiElementwiseOperation{};
    constexpr auto out_op = OutElementwiseOperation{};

    TIn in_val;
    TWei wei_val;
    TOut out_val;

    for(long_index_t ii = tid; ii < output_length; ii += num_threads)
    {
        const index_t n  = ii / out_strides[0];
        index_t k        = ii - n * out_strides[0];
        const index_t dO = k / out_strides[1];
        k -= dO * out_strides[1];
        const index_t ho = k / out_strides[2];
        k -= ho * out_strides[2];
        const index_t wo = k / out_strides[3];
        k -= wo * out_strides[3];

        TAcc acc = static_cast<TAcc>(0);

        const TIn* in_n   = p_in + static_cast<long_index_t>(n) * in_strides[0];
        const TWei* wei_k = p_wei + static_cast<long_index_t>(k) * wei_strides[0];

        for(index_t z = 0; z < Z; ++z)
        {
            index_t di          = stride_z * dO - pad_z + dilation_z * z;
            const TIn* in_n_di  = in_n + di * in_strides[1];
            const TWei* wei_k_z = wei_k + z * wei_strides[1];

            for(index_t y = 0; y < Y; ++y)
            {
                index_t hi            = stride_y * ho - pad_y + dilation_y * y;
                const TIn* in_n_di_hi = in_n_di + hi * in_strides[2];
                const TWei* wei_k_z_y = wei_k_z + y * wei_strides[2];

                for(index_t x = 0; x < X; ++x)
                {
                    index_t wi               = stride_x * wo - pad_x + dilation_x * x;
                    const TIn* in_n_di_hi_wi = in_n_di_hi + wi * in_strides[3];
                    const TWei* wei_k_z_y_x  = wei_k_z_y + x * wei_strides[3];

                    if(di >= 0 && di < Di && hi >= 0 && hi < Hi && wi >= 0 && wi < Wi)
                    {
                        for(index_t c = 0; c < C; ++c)
                        {
                            in_op(in_val, in_n_di_hi_wi[c]);
                            wei_op(wei_val, wei_k_z_y_x[c]);
                            acc += in_val * wei_val;
                        }
                    }
                }
            }
        }

        out_op(out_val, static_cast<TOut>(acc));
        p_out[ii] = out_val;
    }
}
} // namespace ref
} // namespace ck

#endif
