// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

namespace ck {

// Do following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions:
//   1. Don't save a reference to tensor descriptor in class, pass in tensor descriptor as argument
//   instead
//   2. Don't construct a new tensor coordinate everytime when using it, update and reuse the same
//   tensor coordinate instead
//   3. Don't use a pointer to VGPR buffer, use vector instead

namespace detail {
// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor
template <index_t VectorDim, index_t ScalarPerVector>
struct lambda_scalar_per_access
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? ScalarPerVector : 1;
    }
};

template <index_t VectorDim>
struct lambda_scalar_step_in_vector
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? 1 : 0;
    }
};
} // namespace detail

// Assume:
//   1. src:
//     1. SrcDesc is known at compile-time
//     2. SrcBuffer is StaticBuffer
//     3. SrcSliceOrginIdx is known at compile-time
//   2. dst:
//     1. DstDesc is not known at compile-time
//     2. DstBuffer is DynamicBuffer
//     3. DstSliceOrginIdx is not known at compile time
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          InMemoryDataOperationEnum DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          typename enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v1r3
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v1r3(const DstDesc& dst_desc,
                                                            const Index& dst_slice_origin_idx,
                                                            const ElementwiseOperation& element_op)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin_idx)),
          element_op_{element_op}
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx, typename SrcBuffer, typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                      "wrong! SrcSliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer(), "wrong! SrcBuffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>>;

        // TODO: Use SpaceFillingCurve::ScalarsPerAccess instread of DstScalarPerVector?
        static_assert(DstScalarPerVector == SpaceFillingCurve::ScalarPerVector,
                      "wrong!DstScalarPerVector != SpaceFillingCurve::ScalarPerVector");
        typename vector_type_maker<DstData, DstScalarPerVector>::type dst_vector;
        using dst_vector_t = typename vector_type_maker<DstData, DstScalarPerVector>::type::type;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);

            // copy data from src_buf into dst_vector
            // TODO: It's a hack here to use \p dst_scalar_step_in_vector. Use SpaceFillingCurve?
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t src_offset = src_desc.CalculateOffset(
                    src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                SrcData v;

                // apply element-wise operation
                element_op_(v, src_buf[Number<src_offset>{}]);

                // apply type convert
                dst_vector.template AsType<DstData>()(i) = type_convert<DstData>(v);
            });

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            // copy data from dst_vector into dst_buf
            dst_buf.template Update<DstInMemOp, dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector.template AsType<dst_vector_t>()[Number<0>{}]);

            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);

                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step));
            }
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    private:
    DstCoord dst_coord_;
    const ElementwiseOperation element_op_;
}; // namespace ThreadwiseTensorSliceTransfer_v1r3

// Assume:
//   1. src:
//     1. SrcDesc is not known at compile-time
//     2. SrcBuffer is DynamicBuffer
//     3. src_slice_origin_idx is not known at compile-time
//   2. dst:
//     1. DstDesc is known at compile-time
//     2. DstBuffer is StaticBuffer
//     3. dst_slice_origin_idx is known at compile-time
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t SrcScalarStrideInVector,
          bool SrcResetCoordinateAfterRun,
          bool InvalidElementAsNaN                                        = false,
          typename enable_if<DstDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v2
{
    static_assert((InvalidElementAsNaN && !std::is_integral<DstData>::value) ||
                      (!InvalidElementAsNaN),
                  "Filling invalid element as NaN is only for floating point types");

    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v2(const SrcDesc& src_desc,
                                                          const Index& src_slice_origin_idx)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin_idx))
    {
        static_assert(DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
        static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % SrcScalarPerVector == 0,
                      "wrong! Not divisible");
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    template <typename SrcBuffer, typename DstBuffer, typename DstSliceOriginIdx>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstSliceOriginIdx&,
                        DstBuffer& dst_buf)
    {
        static_assert(DstDesc::IsKnownAtCompileTime(),
                      "wrong! DstDesc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<DstSliceOriginIdx>>::value,
                      "wrong! DstSliceOrigin need to known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value &&
            "wrong! inconsistent type");

        // DstDesc and dst_slice_origin_idx are known at compile-time
        constexpr auto dst_desc             = remove_cvref_t<DstDesc>{};
        constexpr auto dst_slice_origin_idx = DstSliceOriginIdx{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(src_scalar_per_access)>>;

        // loop over tensor and copy
        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            typename vector_type_maker<SrcData, SrcScalarPerVector>::type src_vector;

            using src_vector_t =
                typename vector_type_maker<SrcData, SrcScalarPerVector>::type::type;
            constexpr auto src_data_idx = SpaceFillingCurve::GetIndex(idx_1d);

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            // copy data from src_buf into src_vector
            src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid);

            // copy data from src_vector into dst_buf
            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t dst_offset =
                    dst_desc.CalculateOffset(to_multi_index(dst_slice_origin_idx) + src_data_idx +
                                             i * src_scalar_step_in_vector);

                if constexpr(InvalidElementAsNaN)
                {
                    dst_buf(Number<dst_offset>{}) =
                        is_src_valid
                            ? type_convert<DstData>(src_vector.template AsType<SrcData>()[i])
                            : NumericLimits<DstData>::QuietNaN();
                }
                else
                {
                    dst_buf(Number<dst_offset>{}) =
                        type_convert<DstData>(src_vector.template AsType<SrcData>()[i]);
                }
            });

            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);

                move_tensor_coordinate(
                    src_desc, src_coord_, make_tensor_coordinate_step(src_desc, forward_step));
            }
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step =
                make_tensor_coordinate_step(src_desc, GetSrcCoordinateResetStep());

            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(src_scalar_per_access)>>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SpaceFillingCurve::Index{};
        }
        else
        {
            constexpr auto reset_step =
                SpaceFillingCurve::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            return reset_step;
        }
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by Run(), then need to adjust the step here
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

    private:
    SrcCoord src_coord_;
}; // namespace ck

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
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // RunRead(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun> // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
struct ThreadwiseTensorSliceTransfer_v3
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v3(const SrcDesc& src_desc,
                                                          const Index& src_slice_origin,
                                                          const DstDesc& dst_desc,
                                                          const Index& dst_slice_origin)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin))
    {
        static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % SrcScalarPerVector == 0,
                      "wrong! Not divisible");
        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");
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

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // make forward steps
        const auto src_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? src_scalar_per_access[i] : 0;
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
                    backward_step_idx(j) = (i.value == j.value) ? -src_scalar_per_access[i] : 0;
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

                    static_for<1, i, 1>{}([&](auto j) {
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
                       src_scalar_per_access;
            }();

            vector_type_maker_t<SrcData, SrcScalarPerVector> src_tmp_vector;

            using src_vector_t = typename decltype(src_tmp_vector)::type;

            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            // copy data from src_buf to src_tmp_vector
            src_tmp_vector.template AsType<src_vector_t>()(Number<0>{}) =
                src_buf.template Get<src_vector_t>(src_coord_.GetOffset(), is_src_valid);

            // copy data from src_tmp_vector to buffer_
            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(src_data_idx + i * src_scalar_step_in_vector);

                buffer_(Number<buffer_offset>{}) = src_tmp_vector.template AsType<SrcData>()[i];
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

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // src scalar per access on each dim
        // TODO: don't use this
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // make forward steps
        const auto dst_forward_steps = generate_tuple(
            [&](auto i) {
                Index forward_step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? dst_scalar_per_access[i] : 0;
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
                    backward_step_idx(j) = (i.value == j.value) ? -dst_scalar_per_access[i] : 0;
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

                    static_for<1, i, 1>{}([&](auto j) {
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
                       dst_scalar_per_access;
            }();

            vector_type_maker_t<DstData, DstScalarPerVector> dst_tmp_vector;

            // copy data from buffer_ to dst_tmp_vector
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(dst_data_idx + i * dst_scalar_step_in_vector);

                dst_tmp_vector.template AsType<DstData>()(i) =
                    type_convert<DstData>(buffer_[Number<buffer_offset>{}]);
            });

            using dst_vector_t = typename decltype(dst_tmp_vector)::type;

            // copy data from dst_tmp_vector to dst_buf
            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            dst_buf.template Set<dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_tmp_vector.template AsType<dst_vector_t>()[Number<0>{}]);

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
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_src_access_lengths[I0] - 1;

                static_for<1, i, 1>{}([&](auto j) {
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
                   src_scalar_per_access;
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
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_dst_access_lengths[I0] - 1;

                static_for<1, i, 1>{}([&](auto j) {
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
                   dst_scalar_per_access;
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
          index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t SrcScalarStrideInVector,
          typename enable_if<SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseTensorSliceTransfer_v4
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));

    __device__ constexpr ThreadwiseTensorSliceTransfer_v4(const Index& src_ref_idx)
        : src_ref_coord_(make_tensor_coordinate(SrcDesc{}, src_ref_idx))
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc and DstDesc need to known at compile-time");

        static_assert(SliceLengths::At(Number<SrcVectorDim>{}) % SrcScalarPerVector == 0,
                      "wrong! Not divisible");
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

        // scalar per access of each dim
        constexpr auto src_scalar_per_access = generate_sequence_v2(
            [&](auto i) constexpr {
                if constexpr(i == SrcVectorDim)
                {
                    return Number<SrcScalarPerVector>{};
                }
                else
                {
                    return Number<1>{};
                }
            },
            Number<nDim>{});

        // scalar step (if steping on SrcVectorDim) of each dim
        constexpr auto src_scalar_step_in_vector = generate_sequence_v2(
            [&](auto i) constexpr {
                if constexpr(i == SrcVectorDim)
                {
                    return Number<1>{};
                }
                else
                {
                    return Number<0>{};
                }
            },
            Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_access_idx) {
#if 0
            // TODO: unable to compile
            // position in slice window
            constexpr auto data_to_origin_disp_idx =
                container_reorder_given_old2new(ordered_access_idx, dim_access_order) *
                src_scalar_per_access;
#else
            // position in slice window
            constexpr auto data_to_origin_disp_idx =
                ordered_access_idx.ReorderGivenOld2New(dim_access_order) * src_scalar_per_access;
#endif
            // src coordinate
            constexpr auto src_ref_to_data_disp_idx =
                src_ref_to_origin_disp_idx + data_to_origin_disp_idx;

            constexpr auto src_ref_to_data_disp_coord_step =
                make_tensor_coordinate_step(src_desc, src_ref_to_data_disp_idx);

            auto src_data_coord = src_ref_coord_;

            move_tensor_coordinate(src_desc, src_data_coord, src_ref_to_data_disp_coord_step);

            vector_type_maker_t<SrcData, SrcScalarPerVector> src_tmp_vector;

            using src_vector_t = typename decltype(src_tmp_vector)::type;

            const bool is_src_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                src_desc, src_data_coord);

            // copy data from src_buf into src_tmp_vector
            if constexpr(SrcBuffer::IsDynamicBuffer())
            {
                src_tmp_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    src_buf.template Get<src_vector_t>(src_data_coord.GetOffset(), is_src_valid);
            }
            else if constexpr(SrcBuffer::IsStaticBuffer())
            {
                static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                    constexpr index_t src_offset = src_desc.CalculateOffset(
                        src_ref_to_origin_disp_idx + data_to_origin_disp_idx +
                        i * src_scalar_step_in_vector);

                    // apply type convert
                    src_tmp_vector.template AsType<SrcData>()(i) = src_buf[Number<src_offset>{}];
                });
            }
            // copy data from src_tmp_vector to dst_tmp_vector (data cast data from SrcData to
            // DstData)
            vector_type_maker_t<DstData, SrcScalarPerVector> dst_tmp_vector;

            // TODO: if SrcData and DstData are vetor type, then static_cast may not compile
            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                dst_tmp_vector.template AsType<DstData>()(i) =
                    type_convert<DstData>(src_tmp_vector.template AsType<SrcData>()[i]);
            });

            // copy data from dst_tmp_vector into dst_buf
            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t dst_offset = dst_desc.CalculateOffset(
                    dst_origin_idx + data_to_origin_disp_idx + i * src_scalar_step_in_vector);

                dst_buf(Number<dst_offset>{}) = dst_tmp_vector.template AsType<DstData>()[i];
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
    __device__ void SetSrcCoord(const Index& src_ref_idx)
    {
        src_ref_coord_ = make_tensor_coordinate(SrcDesc{}, src_ref_idx);
    }

    private:
    SrcCoord src_ref_coord_;
};

// Do NOT involve any tensor coordinates with StaticBuffer
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          typename enable_if<SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseTensorSliceTransfer_StaticToStatic
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    __device__ constexpr ThreadwiseTensorSliceTransfer_StaticToStatic(
        const ElementwiseOperation& element_op)
        : element_op_{element_op}
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc need to known at compile-time");

        static_assert(SliceLengths::At(Number<DstVectorDim>{}) % DstScalarPerVector == 0,
                      "wrong! Not divisible");
    }

    template <typename SrcSliceOriginIdx,
              typename DstSliceOriginIdx,
              typename SrcBuffer,
              typename DstBuffer>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcBuffer& src_buf,
                        const DstDesc&,
                        const DstSliceOriginIdx&,
                        DstBuffer& dst_buf)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime() && DstDesc::IsKnownAtCompileTime(),
                      "wrong! Desc need to known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<DstSliceOriginIdx>>::value,
                      "wrong! SliceOrigin need to known at compile-time");

        static_assert(SrcBuffer::IsStaticBuffer() && DstBuffer::IsStaticBuffer(),
                      "wrong! Buffer need to be StaticBuffer");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
        constexpr auto dst_desc             = remove_cvref_t<DstDesc>{};
        constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});
        constexpr auto dst_slice_origin_idx = to_multi_index(DstSliceOriginIdx{});

        // scalar per access on each dim
        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(dst_scalar_per_access)>>;

        static_assert(DstScalarPerVector == SpaceFillingCurve::ScalarPerVector,
                      "wrong!DstScalarPerVector != SpaceFillingCurve::ScalarPerVector");

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            constexpr auto idx_md = SpaceFillingCurve::GetIndex(idx_1d);

            // copy data from src_buf into dst_vector
            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t src_offset = src_desc.CalculateOffset(
                    src_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                constexpr index_t dst_offset = dst_desc.CalculateOffset(
                    dst_slice_origin_idx + idx_md + i * dst_scalar_step_in_vector);

                SrcData v;

                // apply element-wise operation
                element_op_(v, src_buf[Number<src_offset>{}]);

                // apply type convert
                dst_buf(Number<dst_offset>{}) = type_convert<DstData>(v);
            });
        });
    }

    ElementwiseOperation element_op_;
};

} // namespace ck
