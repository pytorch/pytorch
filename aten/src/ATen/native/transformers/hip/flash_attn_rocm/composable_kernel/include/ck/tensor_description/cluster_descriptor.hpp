// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

template <typename Lengths,
          typename ArrangeOrder = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type>
__host__ __device__ constexpr auto make_cluster_descriptor(
    const Lengths& lengths,
    ArrangeOrder order = typename arithmetic_sequence_gen<0, Lengths::Size(), 1>::type{})
{
    constexpr index_t ndim_low = Lengths::Size();

    const auto reordered_lengths = container_reorder_given_new2old(lengths, order);

    const auto low_lengths = generate_tuple(
        [&](auto idim_low) { return reordered_lengths[idim_low]; }, Number<ndim_low>{});

    const auto transform = make_merge_transform(low_lengths);

    constexpr auto low_dim_old_top_ids = ArrangeOrder{};

    constexpr auto up_dim_new_top_ids = Sequence<0>{};

    return make_single_stage_tensor_adaptor(
        make_tuple(transform), make_tuple(low_dim_old_top_ids), make_tuple(up_dim_new_top_ids));
}

} // namespace ck
