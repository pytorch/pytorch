// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

namespace ck {
// Assume:
//   1. src:
//     1. SrcDesc is known at compile-time
//     2. SrcBuffer is DynamicBuffer
//     3. src_ref_idx is known at run-time
//     4. SrcRefToOriginDisplacement is known at compile-time
//     5. use #-step
//   2. dst:
//     1. DstDesc is known at compile-time
//     2. DstBuffer is StaticBuffer
//     3. DstOriginIdx is known at compile-time
//     4. use direct address calculation
//   3. vector access on src
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename DimAccessOrder,
          typename SrcVectorTensorLengths,
          typename SrcVectorTensorContiguousDimOrder,
          typename enable_if<SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v4r1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v4r1(const Index& src_ref_idx)
        : src_ref_coord_(make_tensor_coordinate(SrcDesc{}, src_ref_idx))
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        static_for<0, nDim, 1>{}([](auto i) {
            static_assert(SliceLengths::At(i) % SrcVectorTensorLengths::At(i) == 0, "wrong!");
        });
    }

    template <typename SrcRefToOriginDisplacement,
              typename DstOriginIdx,
              typename SrcBuffer,
              typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcRefToOriginDisplacement&,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstOriginIdx&,
                        DstBuffer& dst_buf) const
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>::value &&
                is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value,
            "wrong! SrcBuffer or DstBuffer data type is wrong");

        static_assert(DstBuffer::IsStaticBuffer(), "wrong! DstBuffer need to be StaticBuffer");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcRefToOriginDisplacement>>::value &&
                          is_known_at_compile_time<remove_cvref_t<DstOriginIdx>>::value,
                      "wrong! SrcOriginToRefDistance and DstOriginToRefDistance need to be known "
                      "at compile-time");

        // SrcDesc and DstDesc are known at compile-time
        constexpr auto src_desc = remove_cvref_t<SrcDesc>{};
        constexpr auto dst_desc = remove_cvref_t<DstDesc>{};

        // SrcOriginToRefDisttance and DstOriginToRefDistance are known at compile-time
        constexpr auto src_ref_to_origin_disp_idx = to_multi_index(SrcRefToOriginDisplacement{});
        constexpr auto dst_origin_idx             = to_multi_index(DstOriginIdx{});

        // tensor descriptor for src_vector
        constexpr auto src_vector_tensor_lengths = SrcVectorTensorLengths{};

        constexpr auto src_vector_tensor_strides = container_reorder_given_old2new(
            container_reverse_exclusive_scan(
                container_reorder_given_new2old(src_vector_tensor_lengths,
                                                SrcVectorTensorContiguousDimOrder{}),
                math::multiplies{},
                I1),
            SrcVectorTensorContiguousDimOrder{});

        constexpr auto src_vector_desc =
            make_naive_tensor_descriptor(sequence_to_tuple_of_number(src_vector_tensor_lengths),
                                         sequence_to_tuple_of_number(src_vector_tensor_strides));

        // access order and lengths
        constexpr auto access_lengths = SliceLengths{} / src_vector_tensor_lengths;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_access_idx) {
            // position in slice window
            constexpr auto data_to_origin_disp_idx =
                ordered_access_idx.ReorderGivenOld2New(dim_access_order) *
                src_vector_tensor_lengths;

            // src coordinate at starting point of src_vector
            constexpr auto src_ref_to_data_disp_idx =
                src_ref_to_origin_disp_idx + data_to_origin_disp_idx;

            constexpr auto src_ref_to_data_disp_coord_step =
                make_tensor_coordinate_step(src_desc, src_ref_to_data_disp_idx);

            auto src_data_coord = src_ref_coord_;

            move_tensor_coordinate(src_desc, src_data_coord, src_ref_to_data_disp_coord_step);

            vector_type_maker_t<SrcData, src_vector_desc.GetElementSpaceSize()> src_vector;

            using src_vector_t = typename decltype(src_vector)::type;

            const bool is_src_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                src_desc, src_data_coord);

            // copy data from src_buf into src_vector
            src_vector.template AsType<src_vector_t>()(I0) =
                src_buf.template Get<src_vector_t>(src_data_coord.GetOffset(), is_src_valid);

            // copy data from src_vector into dst_buf (also cast from SrcData to DstData)
            static_ford<SrcVectorTensorLengths>{}([&](auto src_vector_idx_) {
                constexpr auto src_vector_idx = to_multi_index(src_vector_idx_);

                constexpr index_t src_vector_offset =
                    src_vector_desc.CalculateOffset(src_vector_idx);

                constexpr index_t dst_offset = dst_desc.CalculateOffset(
                    dst_origin_idx + data_to_origin_disp_idx + src_vector_idx);

                dst_buf(Number<dst_offset>{}) = type_convert<DstData>(
                    src_vector.template AsType<DstData>()[Number<src_vector_offset>{}]);
            });
        });
    }

    template <typename SrcSliceMoveStepIdx>
    __device__ void MoveSrcSliceWindow(const SrcDesc&,
                                       const SrcSliceMoveStepIdx& src_slice_move_step_idx)
    {
        constexpr auto src_desc = SrcDesc{};

        const auto src_slice_move_step_iter =
            make_tensor_coordinate_step(src_desc, to_multi_index(src_slice_move_step_idx));

        move_tensor_coordinate(SrcDesc{}, src_ref_coord_, src_slice_move_step_iter);
    }

    private:
    SrcCoord src_ref_coord_;
};

} // namespace ck
