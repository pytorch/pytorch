// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

namespace ck {

// Do following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions:
//   1. Don't save a reference to tensor descriptor in class, pass in tensor descriptor as argument
//   instead
//   2. Don't construct a new tensor coordinate everytime when using it, update and reuse the same
//   tensor coordinate instead
//   3. Don't use a pointer to VGPR buffer, use vector instead

// Assume:
//   1. src0_desc and dst_desc are not known at compile-time
//   2. SrcBuffer and DstBuffer are DynamicBuffer
//   3. src_slice_origin and dst_slice_origin are not known at compile-time,
template <typename Src0Data,
          typename Src1Data,
          typename Src2Data,
          typename DstData,
          typename Src0Desc,
          typename Src1Desc,
          typename Src2Desc,
          typename DstDesc,
          typename ElementwiseOperation,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t VectorDim,
          index_t ScalarPerVector,
          InMemoryDataOperationEnum DstInMemOp,
          bool Src0ResetCoordinateAfterRun,
          bool Src1ResetCoordinateAfterRun,
          bool Src2ResetCoordinateAfterRun,
          bool DstResetCoordinateAfterRun>
struct ThreadwiseTensorSliceTransfer_v6r3
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using Src0Coord = decltype(make_tensor_coordinate(Src0Desc{}, Index{}));
    using Src1Coord = decltype(make_tensor_coordinate(Src1Desc{}, Index{}));
    using Src2Coord = decltype(make_tensor_coordinate(Src2Desc{}, Index{}));
    using DstCoord  = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    static constexpr auto I0 = Number<0>{};

    __device__ constexpr ThreadwiseTensorSliceTransfer_v6r3(const Src0Desc& src0_desc,
                                                            const Index& src0_slice_origin,
                                                            const Src1Desc& src1_desc,
                                                            const Index& src1_slice_origin,
                                                            const Src2Desc& src2_desc,
                                                            const Index& src2_slice_origin,
                                                            const DstDesc& dst_desc,
                                                            const Index& dst_slice_origin,
                                                            const ElementwiseOperation& element_op)
        : src0_coord_(make_tensor_coordinate(src0_desc, src0_slice_origin)),
          src1_coord_(make_tensor_coordinate(src1_desc, src1_slice_origin)),
          src2_coord_(make_tensor_coordinate(src2_desc, src2_slice_origin)),
          dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin)),
          element_op_(element_op)
    {
        static_assert(SliceLengths::At(Number<VectorDim>{}) % ScalarPerVector == 0,
                      "wrong! cannot evenly divide");
    }

    __device__ void SetSrc0SliceOrigin(const Src0Desc& src0_desc,
                                       const Index& src0_slice_origin_idx)
    {
        src0_coord_ = make_tensor_coordinate(src0_desc, src0_slice_origin_idx);
    }

    __device__ void SetSrc1SliceOrigin(const Src1Desc& src1_desc,
                                       const Index& src1_slice_origin_idx)
    {
        src1_coord_ = make_tensor_coordinate(src1_desc, src1_slice_origin_idx);
    }

    __device__ void SetSrc2SliceOrigin(const Src2Desc& src2_desc,
                                       const Index& src2_slice_origin_idx)
    {
        src2_coord_ = make_tensor_coordinate(src2_desc, src2_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename Src0Buffer, typename Src1Buffer, typename Src2Buffer, typename DstBuffer>
    __device__ void Run(const Src0Desc& src0_desc,
                        const Src0Buffer& src0_buf,
                        const Src1Desc& src1_desc,
                        const Src1Buffer& src1_buf,
                        const Src2Desc& src2_desc,
                        const Src2Buffer& src2_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(scalar_per_access)>>;

        constexpr auto num_access = SpaceFillingCurve::GetNumOfAccess();

        // loop over space-filling curve
        static_for<0, num_access, 1>{}([&](auto idx_1d) {
            using src0_vector_type = vector_type_maker_t<Src0Data, ScalarPerVector>;
            using src0_vector_t    = typename src0_vector_type::type;

            using src1_vector_type = vector_type_maker_t<Src1Data, ScalarPerVector>;
            using src1_vector_t    = typename src1_vector_type::type;

            using src2_vector_type = vector_type_maker_t<Src2Data, ScalarPerVector>;
            using src2_vector_t    = typename src2_vector_type::type;

            using dst_vector_type = vector_type_maker_t<DstData, ScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;

            const bool is_src0_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src0_desc, src0_coord_);

            const bool is_src1_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src1_desc, src1_coord_);

            const bool is_src2_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src2_desc, src2_coord_);

            // copy data from src0_buf into src0_vector_container
            auto src0_vector_container = src0_vector_type{
                src0_buf.template Get<src0_vector_t>(src0_coord_.GetOffset(), is_src0_valid)};

            auto src1_vector_container = src1_vector_type{
                src1_buf.template Get<src1_vector_t>(src1_coord_.GetOffset(), is_src1_valid)};

            auto src2_vector_container = src2_vector_type{
                src2_buf.template Get<src2_vector_t>(src2_coord_.GetOffset(), is_src2_valid)};

            auto dst_vector_container = dst_vector_type{};

            // apply pointwise operation
            static_for<0, ScalarPerVector, 1>{}([&](auto i) {
                element_op_(dst_vector_container.template AsType<DstData>()(i),
                            src0_vector_container.template AsType<Src0Data>()[i],
                            src1_vector_container.template AsType<Src1Data>()[i],
                            src2_vector_container.template AsType<Src2Data>()[i]);
            });

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            dst_buf.template Update<DstInMemOp, dst_vector_t>(
                dst_coord_.GetOffset(),
                is_dst_valid,
                dst_vector_container.template AsType<dst_vector_t>()[I0]);

            // move coordinate
            if constexpr(idx_1d.value != num_access - 1)
            {
                constexpr auto forward_step = SpaceFillingCurve::GetForwardStep(idx_1d);
                move_tensor_coordinate(
                    src0_desc, src0_coord_, make_tensor_coordinate_step(src0_desc, forward_step));
                move_tensor_coordinate(
                    src1_desc, src1_coord_, make_tensor_coordinate_step(src1_desc, forward_step));
                move_tensor_coordinate(
                    src2_desc, src2_coord_, make_tensor_coordinate_step(src2_desc, forward_step));
                move_tensor_coordinate(
                    dst_desc, dst_coord_, make_tensor_coordinate_step(dst_desc, forward_step));
            }
        });

        // move coordinate back to slice origin (or not)
        if constexpr(Src0ResetCoordinateAfterRun)
        {
            const auto src0_reset_step =
                make_tensor_coordinate_step(src0_desc, GetCoordinateResetStep());

            move_tensor_coordinate(src0_desc, src0_coord_, src0_reset_step);
        }

        if constexpr(Src1ResetCoordinateAfterRun)
        {
            const auto src1_reset_step =
                make_tensor_coordinate_step(src1_desc, GetCoordinateResetStep());

            move_tensor_coordinate(src1_desc, src1_coord_, src1_reset_step);
        }

        if constexpr(Src2ResetCoordinateAfterRun)
        {
            const auto src2_reset_step =
                make_tensor_coordinate_step(src2_desc, GetCoordinateResetStep());

            move_tensor_coordinate(src2_desc, src2_coord_, src2_reset_step);
        }

        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetCoordinateResetStep());

            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    __device__ static constexpr auto GetCoordinateResetStep()
    {
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector>{}, Number<nDim>{});

        using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                    DimAccessOrder,
                                                    remove_cv_t<decltype(scalar_per_access)>>;

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

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrc0SliceWindow(const Src0Desc& src0_desc,
                                        const Index& src0_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx = Src0ResetCoordinateAfterRun
                                           ? src0_slice_origin_step_idx
                                           : src0_slice_origin_step_idx + GetCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src0_desc, adjusted_step_idx);

        move_tensor_coordinate(src0_desc, src0_coord_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrc1SliceWindow(const Src1Desc& src1_desc,
                                        const Index& src1_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx = Src1ResetCoordinateAfterRun
                                           ? src1_slice_origin_step_idx
                                           : src1_slice_origin_step_idx + GetCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src1_desc, adjusted_step_idx);

        move_tensor_coordinate(src1_desc, src1_coord_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrc2SliceWindow(const Src2Desc& src2_desc,
                                        const Index& src2_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx = Src2ResetCoordinateAfterRun
                                           ? src2_slice_origin_step_idx
                                           : src2_slice_origin_step_idx + GetCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(src2_desc, adjusted_step_idx);

        move_tensor_coordinate(src2_desc, src2_coord_, adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx = DstResetCoordinateAfterRun
                                           ? dst_slice_origin_step_idx
                                           : dst_slice_origin_step_idx + GetCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);

        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    private:
    Src0Coord src0_coord_;
    Src1Coord src1_coord_;
    Src2Coord src2_coord_;
    DstCoord dst_coord_;
    const ElementwiseOperation element_op_;
};

} // namespace ck
