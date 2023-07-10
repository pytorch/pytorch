// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <array>
#include <algorithm>
#include <thread>

#include "ck/utility/math_v2.hpp"
#include "ck/utility/ignore.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/tensor_operation/gpu/device/device_batchnorm_backward.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          index_t Rank,
          index_t NumBatchNormReduceDim>
struct ReferenceBatchNormBwd : public device::DeviceBatchNormBwd<XDataType,
                                                                 DxDataType,
                                                                 DyDataType,
                                                                 AccDataType,
                                                                 ScaleDataType,
                                                                 DscaleDbiasDataType,
                                                                 MeanVarDataType,
                                                                 DyElementwiseOp,
                                                                 Rank,
                                                                 NumBatchNormReduceDim>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");

    static constexpr index_t NumInvariantDim = Rank - NumBatchNormReduceDim;

    struct Argument : public device::BaseArgument
    {
        Argument(const std::array<index_t, Rank> xyLengths,
                 const std::array<index_t, Rank> xStrides,
                 const std::array<index_t, Rank> dxStrides,
                 const std::array<index_t, Rank> dyStrides,
                 const std::array<int, NumBatchNormReduceDim> reduceDims,
                 const std::array<index_t, NumInvariantDim> bnScaleBiasMeanVarLengths,
                 const std::array<index_t, NumInvariantDim> bnScaleStrides,
                 const std::array<index_t, NumInvariantDim> bnDscaleDbiasStrides,
                 const std::array<index_t, NumInvariantDim> bnMeanVarStrides,
                 const XDataType* p_x,
                 const DyDataType* p_dy,
                 const ScaleDataType* p_scale,
                 const MeanVarDataType* p_savedMean,
                 const MeanVarDataType* p_savedInvVar,
                 double epsilon,
                 const DyElementwiseOp dy_elementwise_op,
                 DxDataType* p_dx,
                 DscaleDbiasDataType* p_dscale,
                 DscaleDbiasDataType* p_dbias)
            : reduceDims_(reduceDims),
              bnScaleBiasMeanVarLengths_(bnScaleBiasMeanVarLengths),
              bnScaleStrides_(bnScaleStrides),
              bnDscaleDbiasStrides_(bnDscaleDbiasStrides),
              bnMeanVarStrides_(bnMeanVarStrides),
              p_x_(p_x),
              p_dy_(p_dy),
              p_scale_(p_scale),
              p_savedMean_(p_savedMean),
              p_savedInvVar_(p_savedInvVar),
              dy_elementwise_op_(dy_elementwise_op),
              p_dx_(p_dx),
              p_dscale_(p_dscale),
              p_dbias_(p_dbias)
        {
            using ck::host_common::get_index_set;

            if(std::any_of(
                   reduceDims.begin(), reduceDims.end(), [](int d) { return d < 0 || d >= Rank; }))
                throw std::runtime_error("Invalid reduce dimensions!");

            // get invariant_dims[] and invariant_lengths[]
            for(int dim = 0, i = 0; dim < Rank; dim++)
                if(std::none_of(
                       reduceDims.begin(), reduceDims.end(), [&](int d) { return d == dim; }))
                {
                    invariantDims_[i]     = dim;
                    invariant_lengths_[i] = xyLengths[dim];
                    i++;
                };

            // get reduce_lengths_[]
            for(int j = 0, i = 0; j < NumBatchNormReduceDim; j++)
            {
                int dim              = reduceDims[j];
                reduce_lengths_[i++] = xyLengths[dim];
            };

            for(int i = 0; i < NumInvariantDim; i++)
                if(invariant_lengths_[i] != bnScaleBiasMeanVarLengths_[i])
                    throw std::runtime_error("Invalid lengths parameters!");

            for(int j = 0, i = 0; j < NumInvariantDim; j++)
            {
                int dim                  = invariantDims_[j];
                x_invariant_strides_[i]  = xStrides[dim];
                dy_invariant_strides_[i] = dyStrides[dim];
                dx_invariant_strides_[i] = dxStrides[dim];
                i++;
            };

            for(int j = 0, i = 0; j < NumBatchNormReduceDim; j++)
            {
                int dim               = reduceDims_[j];
                x_reduce_strides_[i]  = xStrides[dim];
                dy_reduce_strides_[i] = dyStrides[dim];
                dx_reduce_strides_[i] = dxStrides[dim];
                i++;
            };

            reduceSize_ = std::accumulate(
                reduce_lengths_.begin(), reduce_lengths_.end(), 1, std::multiplies<size_t>{});

            invariant_index_set_ = get_index_set<NumInvariantDim>(invariant_lengths_);
            reduce_index_set_    = get_index_set<NumBatchNormReduceDim>(reduce_lengths_);

            epsilon_ = type_convert<AccDataType>(epsilon);

            haveSavedMeanInvVar_ = (p_savedMean != nullptr && p_savedInvVar != nullptr);
        }

        std::array<int, NumBatchNormReduceDim> reduceDims_;
        std::array<int, NumInvariantDim> invariantDims_;
        std::array<index_t, NumInvariantDim> invariant_lengths_;
        std::array<index_t, NumBatchNormReduceDim> reduce_lengths_;

        const std::array<index_t, NumInvariantDim> bnScaleBiasMeanVarLengths_;
        const std::array<index_t, NumInvariantDim> bnScaleStrides_;
        const std::array<index_t, NumInvariantDim> bnDscaleDbiasStrides_;
        const std::array<index_t, NumInvariantDim> bnMeanVarStrides_;

        std::array<index_t, NumInvariantDim> x_invariant_strides_;
        std::array<index_t, NumInvariantDim> dy_invariant_strides_;
        std::array<index_t, NumInvariantDim> dx_invariant_strides_;
        std::array<index_t, NumBatchNormReduceDim> x_reduce_strides_;
        std::array<index_t, NumBatchNormReduceDim> dy_reduce_strides_;
        std::array<index_t, NumBatchNormReduceDim> dx_reduce_strides_;

        const XDataType* p_x_;
        const DyDataType* p_dy_;
        const ScaleDataType* p_scale_;
        const MeanVarDataType* p_savedMean_;
        const MeanVarDataType* p_savedInvVar_;
        const DyElementwiseOp dy_elementwise_op_;

        DxDataType* p_dx_;
        DscaleDbiasDataType* p_dscale_;
        DscaleDbiasDataType* p_dbias_;

        bool haveSavedMeanInvVar_;

        std::vector<std::array<index_t, NumInvariantDim>> invariant_index_set_;
        std::vector<std::array<index_t, NumBatchNormReduceDim>> reduce_index_set_;

        AccDataType epsilon_;
        size_t reduceSize_;
    };

    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            using ck::host_common::get_offset_from_index;

            auto thread_reduce_func = [&](auto invariant_index) {
                size_t x_invariant_offset = get_offset_from_index<NumInvariantDim>(
                    arg.x_invariant_strides_, invariant_index);
                size_t dy_invariant_offset = get_offset_from_index<NumInvariantDim>(
                    arg.dy_invariant_strides_, invariant_index);
                size_t dx_invariant_offset = get_offset_from_index<NumInvariantDim>(
                    arg.dx_invariant_strides_, invariant_index);

                AccDataType mean     = type_convert<AccDataType>(0.0f);
                AccDataType variance = type_convert<AccDataType>(0.0f);
                AccDataType invVar;
                int32_t curr_count = 0;

                if(arg.haveSavedMeanInvVar_)
                {
                    size_t mean_invVar_invariant_offset = get_offset_from_index<NumInvariantDim>(
                        arg.bnMeanVarStrides_, invariant_index);

                    mean =
                        type_convert<AccDataType>(arg.p_savedMean_[mean_invVar_invariant_offset]);
                    invVar =
                        type_convert<AccDataType>(arg.p_savedInvVar_[mean_invVar_invariant_offset]);
                }
                else
                {
                    // compute mean, variance using welford method
                    for(const auto& reduce_index : arg.reduce_index_set_)
                    {
                        size_t x_reduce_offset = get_offset_from_index<NumBatchNormReduceDim>(
                            arg.x_reduce_strides_, reduce_index);

                        auto x_offset = x_invariant_offset + x_reduce_offset;

                        curr_count++;

                        AccDataType x = type_convert<AccDataType>(arg.p_x_[x_offset]);

                        AccDataType delta = x - mean;

                        mean += delta / curr_count;

                        AccDataType delta2 = x - mean;

                        variance += delta * delta2;
                    };

                    // actual variance
                    variance = variance / curr_count;

                    // inv-variance defined as 1/sqrt(epsilon+variance)
                    invVar =
                        type_convert<AccDataType>(1.0f) / ck::math::sqrt(arg.epsilon_ + variance);
                };

                AccDataType dbias =
                    type_convert<AccDataType>(0.0f); // Sum on reduced dimensions of dy
                AccDataType dscale =
                    type_convert<AccDataType>(0.0f); // Sum on reduced dimensions of dy * norm_x

                // 1) calculate dy * (x - mean) * inv-variance
                // 2) calculate sum(dy) on reduced dimensions
                // 3) calculate sum(dy * norm_x) on reduced dimensions
                for(const auto& reduce_index : arg.reduce_index_set_)
                {
                    size_t x_reduce_offset = get_offset_from_index<NumBatchNormReduceDim>(
                        arg.x_reduce_strides_, reduce_index);
                    size_t dy_reduce_offset = get_offset_from_index<NumBatchNormReduceDim>(
                        arg.dy_reduce_strides_, reduce_index);

                    auto x_offset  = x_invariant_offset + x_reduce_offset;
                    auto dy_offset = dy_invariant_offset + dy_reduce_offset;

                    AccDataType x = type_convert<AccDataType>(arg.p_x_[x_offset]);

                    AccDataType norm_x = (x - mean) * invVar;
                    AccDataType dy     = type_convert<AccDataType>(arg.p_dy_[dy_offset]);

                    arg.dy_elementwise_op_(dy, dy);

                    dbias += dy;
                    dscale += norm_x * dy;
                };

                size_t dscale_offset = get_offset_from_index<NumInvariantDim>(
                    arg.bnDscaleDbiasStrides_, invariant_index);
                size_t dbias_offset = get_offset_from_index<NumInvariantDim>(
                    arg.bnDscaleDbiasStrides_, invariant_index);

                arg.p_dscale_[dscale_offset] = type_convert<DscaleDbiasDataType>(dscale);
                arg.p_dbias_[dbias_offset]   = type_convert<DscaleDbiasDataType>(dbias);

                size_t scale_offset =
                    get_offset_from_index<NumInvariantDim>(arg.bnScaleStrides_, invariant_index);

                AccDataType scale = type_convert<AccDataType>(arg.p_scale_[scale_offset]);

                AccDataType multiplier = type_convert<AccDataType>(1.0f) /
                                         type_convert<AccDataType>(arg.reduceSize_) * invVar *
                                         scale;

                // 1) calculate tmp = dscale * (x - mean) * inv-variance
                // 2) calculate dx = 1/reduceSize * inv-variance * scale * (reduceSize * dy - dbias
                // - tmp)
                for(const auto& reduce_index : arg.reduce_index_set_)
                {
                    size_t x_reduce_offset = get_offset_from_index<NumBatchNormReduceDim>(
                        arg.x_reduce_strides_, reduce_index);
                    size_t dy_reduce_offset = get_offset_from_index<NumBatchNormReduceDim>(
                        arg.dy_reduce_strides_, reduce_index);
                    size_t dx_reduce_offset = get_offset_from_index<NumBatchNormReduceDim>(
                        arg.dx_reduce_strides_, reduce_index);

                    auto x_offset  = x_invariant_offset + x_reduce_offset;
                    auto dy_offset = dy_invariant_offset + dy_reduce_offset;
                    auto dx_offset = dx_invariant_offset + dx_reduce_offset;

                    AccDataType x = type_convert<AccDataType>(arg.p_x_[x_offset]);

                    AccDataType norm_x = (x - mean) * invVar;
                    AccDataType dy     = type_convert<AccDataType>(arg.p_dy_[dy_offset]);

                    arg.dy_elementwise_op_(dy, dy);

                    AccDataType tmpVal = norm_x * dscale;

                    AccDataType dx = multiplier * (type_convert<AccDataType>(arg.reduceSize_) * dy -
                                                   dbias - tmpVal);

                    arg.p_dx_[dx_offset] = type_convert<DxDataType>(dx);
                };
            };

            std::size_t num_thread = std::thread::hardware_concurrency();
            std::size_t work_per_thread =
                (arg.invariant_index_set_.size() + num_thread - 1) / num_thread;

            std::vector<joinable_thread> threads(num_thread);

            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t i_begin = it * work_per_thread;
                std::size_t i_end   = std::min(static_cast<size_t>((it + 1) * work_per_thread),
                                             arg.invariant_index_set_.size());

                auto f = [=] {
                    for(std::size_t i = i_begin; i < i_end; ++i)
                    {
                        thread_reduce_func(arg.invariant_index_set_[i]);
                    }
                };

                threads[it] = joinable_thread(f);
            }

            return (0.0f);
        };

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /*stream_config*/ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        };
    };

    bool IsSupportedArgument(const device::BaseArgument* p_arg) override
    {
        (void)p_arg;

        return (true);
    };

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const std::array<index_t, Rank> xyLengths,
                        const std::array<index_t, Rank> xStrides,
                        const std::array<index_t, Rank> dxStrides,
                        const std::array<index_t, Rank> dyStrides,
                        const std::array<int, NumBatchNormReduceDim> reduceDims,
                        const std::array<index_t, NumInvariantDim> bnScaleBiasMeanVarLengths,
                        const std::array<index_t, NumInvariantDim> bnScaleStrides,
                        const std::array<index_t, NumInvariantDim> bnDscaleDbiasStrides,
                        const std::array<index_t, NumInvariantDim> bnMeanVarStrides,
                        const void* p_x,
                        const void* p_dy,
                        const void* p_scale,
                        const void* p_savedMean,
                        const void* p_savedInvVar,
                        double epsilon,
                        const DyElementwiseOp dy_elementwise_op,
                        void* p_dx,
                        void* p_dscale,
                        void* p_dbias) override
    {
        return std::make_unique<Argument>(xyLengths,
                                          xStrides,
                                          dxStrides,
                                          dyStrides,
                                          reduceDims,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleStrides,
                                          bnDscaleDbiasStrides,
                                          bnMeanVarStrides,
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const DyDataType*>(p_dy),
                                          static_cast<const ScaleDataType*>(p_scale),
                                          static_cast<const MeanVarDataType*>(p_savedMean),
                                          static_cast<const MeanVarDataType*>(p_savedInvVar),
                                          epsilon,
                                          dy_elementwise_op,
                                          static_cast<DxDataType*>(p_dx),
                                          static_cast<DscaleDbiasDataType*>(p_dscale),
                                          static_cast<DscaleDbiasDataType*>(p_dbias));
    };

    std::unique_ptr<device::BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "Reference_BatchNorm_Backward" << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
