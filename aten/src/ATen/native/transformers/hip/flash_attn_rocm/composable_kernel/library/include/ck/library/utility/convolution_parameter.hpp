// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <numeric>
#include <iterator>
#include <vector>

#include "ck/ck.hpp"

#include "ck/library/utility/numeric.hpp"

namespace ck {
namespace utils {
namespace conv {

struct ConvParam
{
    ConvParam();
    ConvParam(ck::index_t n_dim,
              ck::index_t group_count,
              ck::index_t n_batch,
              ck::index_t n_out_channels,
              ck::index_t n_in_channels,
              const std::vector<ck::index_t>& filters_len,
              const std::vector<ck::index_t>& input_len,
              const std::vector<ck::index_t>& strides,
              const std::vector<ck::index_t>& dilations,
              const std::vector<ck::index_t>& left_pads,
              const std::vector<ck::index_t>& right_pads);

    ck::index_t num_dim_spatial_;
    ck::index_t G_;
    ck::index_t N_;
    ck::index_t K_;
    ck::index_t C_;

    std::vector<ck::index_t> filter_spatial_lengths_;
    std::vector<ck::index_t> input_spatial_lengths_;
    std::vector<ck::index_t> output_spatial_lengths_;

    std::vector<ck::index_t> conv_filter_strides_;
    std::vector<ck::index_t> conv_filter_dilations_;

    std::vector<ck::index_t> input_left_pads_;
    std::vector<ck::index_t> input_right_pads_;

    std::vector<ck::index_t> GetOutputSpatialLengths() const;

    std::size_t GetFlops() const;

    template <typename InDataType>
    std::size_t GetInputByte() const
    {
        // sizeof(InDataType) * (G * N * C * <input spatial lengths product>) +
        return sizeof(InDataType) *
               (G_ * N_ * C_ *
                ck::accumulate_n<std::size_t>(
                    std::begin(input_spatial_lengths_), num_dim_spatial_, 1, std::multiplies<>()));
    }

    template <typename WeiDataType>
    std::size_t GetWeightByte() const
    {
        // sizeof(WeiDataType) * (G * K * C * <filter spatial lengths product>) +
        return sizeof(WeiDataType) *
               (G_ * K_ * C_ *
                ck::accumulate_n<std::size_t>(
                    std::begin(filter_spatial_lengths_), num_dim_spatial_, 1, std::multiplies<>()));
    }

    template <typename OutDataType>
    std::size_t GetOutputByte() const
    {
        // sizeof(OutDataType) * (G * N * K * <output spatial lengths product>);
        return sizeof(OutDataType) * (G_ * N_ * K_ *
                                      std::accumulate(std::begin(output_spatial_lengths_),
                                                      std::end(output_spatial_lengths_),
                                                      static_cast<std::size_t>(1),
                                                      std::multiplies<std::size_t>()));
    }

    template <typename InDataType, typename WeiDataType, typename OutDataType>
    std::size_t GetByte() const
    {
        return GetInputByte<InDataType>() + GetWeightByte<WeiDataType>() +
               GetOutputByte<OutDataType>();
    }
};

std::string get_conv_param_parser_helper_msg();

ConvParam parse_conv_param(int num_dim_spatial, int arg_idx, char* const argv[]);

} // namespace conv
} // namespace utils
} // namespace ck

std::ostream& operator<<(std::ostream& os, const ck::utils::conv::ConvParam& p);
