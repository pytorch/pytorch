// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host_utility/io.hpp"

#include "ck/library/utility/convolution_parameter.hpp"

namespace ck {
namespace utils {
namespace conv {

ConvParam::ConvParam(ck::index_t n_dim,
                     ck::index_t group_count,
                     ck::index_t n_batch,
                     ck::index_t n_out_channels,
                     ck::index_t n_in_channels,
                     const std::vector<ck::index_t>& filters_len,
                     const std::vector<ck::index_t>& input_len,
                     const std::vector<ck::index_t>& strides,
                     const std::vector<ck::index_t>& dilations,
                     const std::vector<ck::index_t>& left_pads,
                     const std::vector<ck::index_t>& right_pads)
    : num_dim_spatial_(n_dim),
      G_(group_count),
      N_(n_batch),
      K_(n_out_channels),
      C_(n_in_channels),
      filter_spatial_lengths_(filters_len),
      input_spatial_lengths_(input_len),
      output_spatial_lengths_(num_dim_spatial_),
      conv_filter_strides_(strides),
      conv_filter_dilations_(dilations),
      input_left_pads_(left_pads),
      input_right_pads_(right_pads)
{
    if(static_cast<ck::index_t>(filter_spatial_lengths_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(input_spatial_lengths_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(conv_filter_strides_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(conv_filter_dilations_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(input_left_pads_.size()) != num_dim_spatial_ ||
       static_cast<ck::index_t>(input_right_pads_.size()) != num_dim_spatial_)
    {
        throw(
            std::runtime_error("ConvParam::ConvParam: "
                               "parameter size is different from number of declared dimensions!"));
    }

    for(ck::index_t i = 0; i < num_dim_spatial_; ++i)
    {
        // XEff = (X - 1) * conv_dilation_w + 1;
        // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
        const ck::index_t x_eff = (filter_spatial_lengths_[i] - 1) * conv_filter_dilations_[i] + 1;

        output_spatial_lengths_[i] =
            (input_spatial_lengths_[i] + input_left_pads_[i] + input_right_pads_[i] - x_eff) /
                conv_filter_strides_[i] +
            1;
    }
}

ConvParam::ConvParam()
    : ConvParam::ConvParam(2, 1, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1})
{
}

std::vector<ck::index_t> ConvParam::GetOutputSpatialLengths() const
{
    return output_spatial_lengths_;
}

std::size_t ConvParam::GetFlops() const
{
    // 2 * G * N * K * C * <output spatial lengths product> * <filter spatial lengths product>
    return static_cast<std::size_t>(2) * G_ * N_ * K_ * C_ *
           ck::accumulate_n<std::size_t>(
               std::begin(output_spatial_lengths_), num_dim_spatial_, 1, std::multiplies<>()) *
           ck::accumulate_n<std::size_t>(
               std::begin(filter_spatial_lengths_), num_dim_spatial_, 1, std::multiplies<>());
}

std::string get_conv_param_parser_helper_msg()
{
    std::string msg;

    msg += "Following arguments (depending on number of spatial dims):\n"
           " Number of spatial dimensions (1=Conv1d, 2=Conv2d, 3=Conv3d)\n"
           " G, N, K, C, \n"
           " <filter spatial dimensions>, (ie Y, X for 2D)\n"
           " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
           " <strides>, (ie Sy, Sx for 2D)\n"
           " <dilations>, (ie Dy, Dx for 2D)\n"
           " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
           " <right padding>, (ie RightPy, RightPx for 2D)\n";

    return msg;
}

ck::utils::conv::ConvParam parse_conv_param(int num_dim_spatial, int arg_idx, char* const argv[])
{
    const ck::index_t G = std::stoi(argv[arg_idx++]);
    const ck::index_t N = std::stoi(argv[arg_idx++]);
    const ck::index_t K = std::stoi(argv[arg_idx++]);
    const ck::index_t C = std::stoi(argv[arg_idx++]);

    std::vector<ck::index_t> filter_spatial_lengths(num_dim_spatial);
    std::vector<ck::index_t> input_spatial_lengths(num_dim_spatial);
    std::vector<ck::index_t> conv_filter_strides(num_dim_spatial);
    std::vector<ck::index_t> conv_filter_dilations(num_dim_spatial);
    std::vector<ck::index_t> input_left_pads(num_dim_spatial);
    std::vector<ck::index_t> input_right_pads(num_dim_spatial);

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        filter_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_strides[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_dilations[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_left_pads[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_right_pads[i] = std::stoi(argv[arg_idx++]);
    }

    return ck::utils::conv::ConvParam{num_dim_spatial,
                                      G,
                                      N,
                                      K,
                                      C,
                                      filter_spatial_lengths,
                                      input_spatial_lengths,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads};
}
} // namespace conv
} // namespace utils
} // namespace ck

std::ostream& operator<<(std::ostream& os, const ck::utils::conv::ConvParam& p)
{
    os << "ConvParam {"
       << "\nnum_dim_spatial: " << p.num_dim_spatial_ << "\nG: " << p.G_ << "\nN: " << p.N_
       << "\nK: " << p.K_ << "\nC: " << p.C_
       << "\nfilter_spatial_lengths: " << p.filter_spatial_lengths_
       << "\ninput_spatial_lengths: " << p.input_spatial_lengths_
       << "\nconv_filter_strides: " << p.conv_filter_strides_
       << "\nconv_filter_dilations: " << p.conv_filter_dilations_
       << "\ninput_left_pads: " << p.input_left_pads_
       << "\ninput_right_pads: " << p.input_right_pads_ << "}\n";

    return os;
}
