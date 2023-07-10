// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

namespace ck {

// Assume:
//   1. src_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
//   4. Use thread buffer
template <typename SliceLengths,
          InMemoryDataOperationEnum DstInMemOp,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          typename SrcVectorTensorLengths,
          typename DstVectorTensorLengths,
          typename SrcVectorTensorContiguousDimOrder,
          typename DstVectorTensorContiguousDimOrder,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // RunRead(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun> // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
struct ThreadwiseTensorSliceTransfer_v5r1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v5r1(const SrcDesc& src_desc,
                                                            const Index& src_slice_origin,
                                                            const DstDesc& dst_desc,
                                                            const Index& dst_slice_origin)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin))
    {
        // TODO: fix this
        static_assert(is_same<SrcData, DstData>::value,
                      "wrong! current implementation assume SrcData and DstData are same type");

        static_for<0, nDim, 1>{}([](auto i) {
            static_assert(SliceLengths::At(i) % SrcVectorTensorLengths::At(i) == 0 &&
                              SliceLengths::At(i) % DstVectorTensorLengths::At(i) == 0,
                          "wrong!");
        });
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcBuffer, typename SrcStepHacks>
    __device__ void
    RunRead(const SrcDesc& src_desc, const SrcBuffer& src_buf, const SrcStepHacks& src_step_hacks)
    {
        static_assert(SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>::value,
            "wrong! SrcBuffer and SrcData data type are inconsistent");

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
        constexpr auto src_access_lengths = SliceLengths{} / src_vector_tensor_lengths;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // make forward steps
        const auto src_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? src_vector_tensor_lengths[i] : 0;
                });

                return make_tensor_coordinate_step(
                    src_desc, forward_step_idx, src_step_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward steps
        const auto src_backward_steps = generate_tuple(
            [&](auto i) {
                Index backward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -src_vector_tensor_lengths[i] : 0;
                });

                return make_tensor_coordinate_step(
                    src_desc, backward_step_idx, src_step_hacks[I1][i]);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_src_access_lengths)>{}([&](auto ordered_src_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_src_access_idx[I0];

                    static_for<0, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_src_access_lengths[j] + ordered_src_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            // calculate src data index
            constexpr auto src_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_src_access_idx[i]
                                                      : ordered_src_access_lengths[i] - 1 -
                                                            ordered_src_access_idx[i];
                });

                return container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                       src_vector_tensor_lengths;
            }();

            vector_type_maker_t<SrcData, src_vector_desc.GetElementSpaceSize()> src_vector;

            using src_vector_t = typename decltype(src_vector)::type;

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            // copy data from src_buf to src_vector
            src_vector.template AsType<src_vector_t>()(I0) =
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid);

            // copy data from src_vector to buffer_
            static_ford<SrcVectorTensorLengths>{}([&](auto src_vector_idx_) {
                constexpr auto src_vector_idx = to_multi_index(src_vector_idx_);

                constexpr index_t src_vector_offset =
                    src_vector_desc.CalculateOffset(src_vector_idx);

                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(src_data_idx + src_vector_idx);

                buffer_(Number<buffer_offset>{}) =
                    src_vector.template AsType<SrcData>()[Number<src_vector_offset>{}];
            });

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_src_access_idx[i] < ordered_src_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &=
                            ordered_src_access_idx[j] == ordered_src_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }
            ();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            src_desc, src_coord_, src_forward_steps[src_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            src_desc, src_coord_, src_backward_steps[src_dim_access_order[i]]);
                    }
                }
            });
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step =
                make_tensor_coordinate_step(src_desc, GetSrcCoordinateResetStep());

            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }
    }

    template <typename DstBuffer, typename DstStepHacks>
    __device__ void
    RunWrite(const DstDesc& dst_desc, DstBuffer& dst_buf, const DstStepHacks& dst_step_hacks)
    {
        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value,
            "wrong! SrcBuffer or DstBuffer data type is wrong");

        // tensor descriptor for dst_vector
        constexpr auto dst_vector_tensor_lengths = DstVectorTensorLengths{};

        constexpr auto dst_vector_tensor_strides = container_reorder_given_old2new(
            container_reverse_exclusive_scan(
                container_reorder_given_new2old(dst_vector_tensor_lengths,
                                                DstVectorTensorContiguousDimOrder{}),
                math::multiplies{},
                I1),
            DstVectorTensorContiguousDimOrder{});

        constexpr auto dst_vector_desc =
            make_naive_tensor_descriptor(sequence_to_tuple_of_number(dst_vector_tensor_lengths),
                                         sequence_to_tuple_of_number(dst_vector_tensor_strides));

        // dst access order and lengths
        constexpr auto dst_access_lengths = SliceLengths{} / dst_vector_tensor_lengths;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // make forward steps
        const auto dst_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? dst_vector_tensor_lengths[i] : 0;
                });

                return make_tensor_coordinate_step(
                    dst_desc, forward_step_idx, dst_step_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward steps
        const auto dst_backward_steps = generate_tuple(
            [&](auto i) {
                Index backward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -dst_vector_tensor_lengths[i] : 0;
                });

                return make_tensor_coordinate_step(
                    dst_desc, backward_step_idx, dst_step_hacks[I1][i]);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_dst_access_lengths)>{}([&](auto ordered_dst_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_dst_access_idx[I0];

                    static_for<0, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_dst_access_lengths[j] + ordered_dst_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            // calculate dst data index
            constexpr auto dst_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_dst_access_idx[i]
                                                      : ordered_dst_access_lengths[i] - 1 -
                                                            ordered_dst_access_idx[i];
                });

                return container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                       dst_vector_tensor_lengths;
            }();

            vector_type_maker_t<DstData, dst_vector_desc.GetElementSpaceSize()> dst_vector;

            // copy data from buffer_ to dst_vector (also cast from SrcData to DstData)
            static_ford<DstVectorTensorLengths>{}([&](auto dst_vector_idx_) {
                constexpr auto dst_vector_idx = to_multi_index(dst_vector_idx_);

                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(dst_data_idx + dst_vector_idx);

                constexpr index_t dst_vector_offset =
                    dst_vector_desc.CalculateOffset(dst_vector_idx);

                dst_vector.template AsType<DstData>()(Number<dst_vector_offset>{}) =
                    type_convert<DstData>(buffer_[Number<buffer_offset>{}]);
            });

            using dst_vector_t = typename decltype(dst_vector)::type;

            // copy data from dst_vector to dst_buf
            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            dst_buf.template Set<dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector.template AsType<dst_vector_t>()[Number<0>{}]);

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_dst_access_idx[i] < ordered_dst_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &=
                            ordered_dst_access_idx[j] == ordered_dst_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }
            ();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_forward_steps[dst_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_backward_steps[dst_dim_access_order[i]]);
                    }
                }
            });
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    template <typename SrcBuffer>
    __device__ void RunRead(const SrcDesc& src_desc, const SrcBuffer& src_buf)
    {
        constexpr index_t ntransform_src = SrcDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_src, 0>::type{};

        constexpr auto src_step_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        RunRead(src_desc, src_buf, src_step_hacks);
    }

    template <typename DstBuffer>
    __device__ void RunWrite(const DstDesc& dst_desc, DstBuffer& dst_buf)
    {
        constexpr index_t ntransform_dst = DstDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_dst, 0>::type{};

        constexpr auto dst_step_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        RunWrite(dst_desc, dst_buf, dst_step_hacks);
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto src_vector_tensor_lengths = SrcVectorTensorLengths{};

        constexpr auto src_access_lengths = SliceLengths{} / src_vector_tensor_lengths;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_src_access_lengths[I0] - 1;

                static_for<0, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_src_access_lengths[j] + ordered_src_access_lengths[j] - 1;
                });

                forward_sweep_(i) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate src data index after last iteration in RunRead(), if it has not being reset by
        // RunRead()
        constexpr auto src_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_src_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                   src_vector_tensor_lengths;
        }();

        //
        constexpr auto reset_src_data_step = [&]() {
            Index reset_src_data_step_;

            static_for<0, nDim, 1>{}([&](auto i) { reset_src_data_step_(i) = -src_data_idx[i]; });

            return reset_src_data_step_;
        }();

        return reset_src_data_step;
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto dst_vector_tensor_lengths = DstVectorTensorLengths{};

        constexpr auto dst_access_lengths = SliceLengths{} / dst_vector_tensor_lengths;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_dst_access_lengths[I0] - 1;

                static_for<0, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_dst_access_lengths[j] + ordered_dst_access_lengths[j] - 1;
                });

                forward_sweep_(i) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate dst data index after last iteration in RunWrite(), if it has not being reset by
        // RunWrite()
        constexpr auto dst_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_dst_access_lengths[i] - 1 : 0;
            });

            return container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                   dst_vector_tensor_lengths;
        }();

        //
        constexpr auto reset_dst_data_step = [&]() {
            Index reset_dst_data_step_;

            static_for<0, nDim, 1>{}([&](auto i) { reset_dst_data_step_(i) = -dst_data_idx[i]; });

            return reset_dst_data_step_;
        }();

        return reset_dst_data_step;
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src_desc, adjusted_step_idx);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <typename SrcMoveSliceWindowStepHack>
    __device__ void
    MoveSrcSliceWindow(const SrcDesc& src_desc,
                       const Index& src_slice_origin_step_idx,
                       const SrcMoveSliceWindowStepHack& src_move_slice_window_step_hack)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(
            src_desc, adjusted_step_idx, src_move_slice_window_step_hack);

        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }
    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    private:
    static constexpr auto buffer_desc_ =
        make_naive_tensor_descriptor_packed(sequence_to_tuple_of_number(SliceLengths{}));

    static constexpr auto buffer_size_ = buffer_desc_.GetElementSpaceSize();

    StaticBuffer<AddressSpaceEnum::Vgpr, SrcData, buffer_size_, true> buffer_;

    SrcCoord src_coord_;
    DstCoord dst_coord_;
};

} // namespace ck
