// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

namespace ck {

// Assume:
//   1. Desc is known at compile-time
//   2. Buffer is StaticBuffer
//   3. OriginIdx is known at compile-time
//   4. use #-step
template <typename Data,
          typename Desc,
          typename SliceLengths,
          typename enable_if<Desc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceSet_v1
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    template <typename OriginIdx, typename Buffer>
    __device__ void Run(const Desc&, const OriginIdx&, Buffer& buf, const Data& initial_value) const
    {
        static_assert(Desc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        static_assert(Buffer::IsStaticBuffer(), "wrong! DstBuffer need to be StaticBuffer");

        static_assert(is_known_at_compile_time<remove_cvref_t<OriginIdx>>::value,
                      "wrong! OriginIdx need to be known at compile-time");

        // Desc is known at compile-time
        constexpr auto desc = remove_cvref_t<Desc>{};

        // OriginIdx is known at compile-time
        constexpr auto origin_idx = to_multi_index(OriginIdx{});

        static_ford<SliceLengths>{}([&](auto access_idx) {
            constexpr auto coord = make_tensor_coordinate(desc, origin_idx + access_idx);

            constexpr bool is_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(desc, coord);

            constexpr index_t offset = coord.GetOffset();

            if constexpr(is_valid)
            {
                buf(Number<offset>{}) = initial_value;
            }
        });
    }
};

} // namespace ck
