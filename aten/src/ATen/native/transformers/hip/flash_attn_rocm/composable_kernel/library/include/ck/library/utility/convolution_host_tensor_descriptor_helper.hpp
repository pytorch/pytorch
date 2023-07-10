// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace utils {
namespace conv {

namespace detail {

template <typename OldLayout>
std::vector<std::size_t> get_layout_transpose_gnchw_to_old()
{
    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NWC> ||
                 ck::is_same_v<OldLayout, ck::tensor_layout::convolution::KXC> ||
                 ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NWK>)
    {
        return {0, 1, 3, 2};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NHWC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::KYXC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NHWK>)
    {
        return {0, 1, 4, 2, 3};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NDHWC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::KZYXC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NDHWK>)
    {
        return {0, 1, 5, 2, 3, 4};
    }
    // separate from legacy code above
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNCW> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GKCX> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNKW>)
    {
        return {0, 1, 2, 3};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNCHW> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GKCYX> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNKHW>)
    {
        return {0, 1, 2, 3, 4};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNCDHW> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GKCZYX> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNKDHW>)
    {
        return {0, 1, 2, 3, 4, 5};
    }
    if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNWC> ||
                 ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GKXC> ||
                 ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNWK>)
    {
        return {0, 1, 3, 2};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNHWC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GKYXC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNHWK>)
    {
        return {0, 1, 4, 2, 3};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNDHWC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GKZYXC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::GNDHWK>)
    {
        return {0, 1, 5, 2, 3, 4};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NWGC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::KXGC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NWGK>)
    {
        return {2, 0, 3, 1};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NHWGC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::KYXGC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NHWGK>)
    {
        return {3, 0, 4, 1, 2};
    }
    else if constexpr(ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NDHWGC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::KZYXGC> ||
                      ck::is_same_v<OldLayout, ck::tensor_layout::convolution::NDHWGK>)
    {
        return {4, 0, 5, 1, 2, 3};
    }
    else
    {
        printf("%s\n", __func__);
        throw std::runtime_error("wrong! unsupported layout");
    }
}

} // namespace detail

// make tensor descriptor for packed input tensor, and order the dimension in the order of GNCHW
// regardless of physical layout
template <typename InLayout>
HostTensorDescriptor
make_input_host_tensor_descriptor_g_n_c_wis_packed(const ck::utils::conv::ConvParam& param)
{
    std::vector<std::size_t> physical_lengths;

    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NWC> ||
                 ck::is_same_v<InLayout, ck::tensor_layout::convolution::NHWC> ||
                 ck::is_same_v<InLayout, ck::tensor_layout::convolution::NDHWC>)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::GNCW> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::GNCHW> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::GNCDHW>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::GNWC> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::GNHWC> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::GNDHWC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck::is_same_v<InLayout, ck::tensor_layout::convolution::NWGC> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::NHWGC> ||
                      ck::is_same_v<InLayout, ck::tensor_layout::convolution::NDHWGC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.input_spatial_lengths_.begin(),
                                param.input_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", InLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<InLayout>());
}

// make tensor descriptor for packed weight tensor, and order the dimension in the order of GKCYX
// regardless of physical layout
template <typename WeiLayout>
HostTensorDescriptor
make_weight_host_tensor_descriptor_g_k_c_xs_packed(const ck::utils::conv::ConvParam& param)
{
    std::vector<std::size_t> physical_lengths;

    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KXC> ||
                 ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KYXC> ||
                 ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KZYXC>)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KXC> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KYXC> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KZYXC>)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::GKCX> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::GKCYX> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::GKCZYX>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::GKXC> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::GKYXC> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::GKZYXC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KXGC> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KYXGC> ||
                      ck::is_same_v<WeiLayout, ck::tensor_layout::convolution::KZYXGC>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.K_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.C_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.filter_spatial_lengths_.begin(),
                                param.filter_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", WeiLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<WeiLayout>());
}

// make tensor descriptor for packed output tensor, and order the dimension in the order of GNKHW
// regardless of physical layout
template <typename OutLayout>
HostTensorDescriptor
make_output_host_tensor_descriptor_g_n_k_wos_packed(const ck::utils::conv::ConvParam& param)
{
    std::vector<std::size_t> physical_lengths;

    // HACK: NHWC/KYXC/NHWK, which is treated as GNHWC/GKYXC/GNHWK by this function,
    // is used by some legacy kernel. New kernel should use GNHWK/GKYXC/GNHWK
    // TODO: remove this branch after removing legacy kernel
    if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NWK> ||
                 ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NHWK> ||
                 ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NDHWK>)
    {
        if(param.G_ != 1)
        {
            throw std::runtime_error("wrong! G != 1");
        }

        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    // separate from legacy code above
    else if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::GNKW> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::GNKHW> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::GNKDHW>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.end(),
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::GNWK> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::GNHWK> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::GNDHWK>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 2,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else if constexpr(ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NWGK> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NHWGK> ||
                      ck::is_same_v<OutLayout, ck::tensor_layout::convolution::NDHWGK>)
    {
        physical_lengths = std::vector<std::size_t>{static_cast<std::size_t>(param.N_),
                                                    static_cast<std::size_t>(param.G_),
                                                    static_cast<std::size_t>(param.K_)};

        physical_lengths.insert(physical_lengths.begin() + 1,
                                param.output_spatial_lengths_.begin(),
                                param.output_spatial_lengths_.begin() + param.num_dim_spatial_);
    }
    else
    {
        printf("%s\n", __func__);
        printf("%s\n", OutLayout::name);
        throw std::runtime_error("wrong! unsupported layout");
    }

    return transpose_host_tensor_descriptor_given_new2old(
        HostTensorDescriptor(physical_lengths),
        detail::get_layout_transpose_gnchw_to_old<OutLayout>());
}

} // namespace conv
} // namespace utils
} // namespace ck
