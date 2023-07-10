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
#include "ck/tensor_operation/gpu/device/device_batchnorm_forward.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename YElementwiseOp,
          index_t Rank,
          index_t NumBatchNormReduceDim>
struct ReferenceBatchNormFwd : public device::DeviceBatchNormFwd<XDataType,
                                                                 YDataType,
                                                                 AccDataType,
                                                                 ScaleDataType,
                                                                 BiasDataType,
                                                                 MeanVarDataType,
                                                                 YElementwiseOp,
                                                                 Rank,
                                                                 NumBatchNormReduceDim>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");

    static constexpr index_t NumInvariantDim = Rank - NumBatchNormReduceDim;

    struct Argument : public device::BaseArgument
    {
        Argument(const std::array<index_t, Rank> xyLengths,
                 const std::array<index_t, Rank> xStrides,
                 const std::array<index_t, Rank> yStrides,
                 const std::array<int, NumBatchNormReduceDim> reduceDims,
                 const std::array<index_t, NumInvariantDim> bnScaleBiasMeanVarLengths,
                 const std::array<index_t, NumInvariantDim> bnScaleStrides,
                 const std::array<index_t, NumInvariantDim> bnBiasStrides,
                 const std::array<index_t, NumInvariantDim> bnMeanVarStrides,
                 const XDataType* p_x,
                 const ScaleDataType* bnScale,
                 const BiasDataType* bnBias,
                 double epsilon,
                 const YElementwiseOp y_elementwise_op,
                 YDataType* p_y,
                 MeanVarDataType* resultSaveMean,
                 MeanVarDataType* resultSaveInvVariance,
                 double averageFactor,
                 MeanVarDataType* resultRunningMean,
                 MeanVarDataType* resultRunningVariance)
            : reduceDims_(reduceDims),
              bnScaleBiasMeanVarLengths_(bnScaleBiasMeanVarLengths),
              bnScaleStrides_(bnScaleStrides),
              bnBiasStrides_(bnBiasStrides),
              bnMeanVarStrides_(bnMeanVarStrides),
              p_x_(p_x),
              bnScale_(bnScale),
              bnBias_(bnBias),
              y_elementwise_op_(y_elementwise_op),
              p_y_(p_y),
              resultSaveMean_(resultSaveMean),
              resultSaveInvVariance_(resultSaveInvVariance),
              resultRunningMean_(resultRunningMean),
              resultRunningVariance_(resultRunningVariance)
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
                int dim                 = invariantDims_[j];
                x_invariant_strides_[i] = xStrides[dim];
                y_invariant_strides_[i] = yStrides[dim];
                i++;
            };

            for(int j = 0, i = 0; j < NumBatchNormReduceDim; j++)
            {
                int dim              = reduceDims_[j];
                x_reduce_strides_[i] = xStrides[dim];
                y_reduce_strides_[i] = yStrides[dim];
                i++;
            };

            invariant_index_set_ = get_index_set<NumInvariantDim>(invariant_lengths_);
            reduce_index_set_    = get_index_set<NumBatchNormReduceDim>(reduce_lengths_);

            epsilon_       = type_convert<AccDataType>(epsilon);
            averageFactor_ = type_convert<AccDataType>(averageFactor);

            resultSave    = (resultSaveMean != nullptr && resultSaveInvVariance != nullptr);
            resultRunning = (resultRunningMean != nullptr && resultRunningVariance != nullptr);
        }

        std::array<int, NumBatchNormReduceDim> reduceDims_;
        std::array<int, NumInvariantDim> invariantDims_;
        std::array<index_t, NumInvariantDim> invariant_lengths_;
        std::array<index_t, NumBatchNormReduceDim> reduce_lengths_;

        const std::array<index_t, NumInvariantDim> bnScaleBiasMeanVarLengths_;
        const std::array<index_t, NumInvariantDim> bnScaleStrides_;
        const std::array<index_t, NumInvariantDim> bnBiasStrides_;
        const std::array<index_t, NumInvariantDim> bnMeanVarStrides_;

        std::array<index_t, NumInvariantDim> x_invariant_strides_;
        std::array<index_t, NumInvariantDim> y_invariant_strides_;
        std::array<index_t, NumBatchNormReduceDim> x_reduce_strides_;
        std::array<index_t, NumBatchNormReduceDim> y_reduce_strides_;

        const XDataType* p_x_;
        const ScaleDataType* bnScale_;
        const BiasDataType* bnBias_;
        const YElementwiseOp y_elementwise_op_;
        YDataType* p_y_;

        MeanVarDataType* resultSaveMean_;
        MeanVarDataType* resultSaveInvVariance_;
        MeanVarDataType* resultRunningMean_;
        MeanVarDataType* resultRunningVariance_;

        bool resultSave, resultRunning;

        std::vector<std::array<index_t, NumInvariantDim>> invariant_index_set_;
        std::vector<std::array<index_t, NumBatchNormReduceDim>> reduce_index_set_;

        AccDataType averageFactor_;
        AccDataType epsilon_;
    };

    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            using ck::host_common::get_offset_from_index;

            auto thread_reduce_func = [&](auto invariant_index) {
                size_t x_invariant_offset = get_offset_from_index<NumInvariantDim>(
                    arg.x_invariant_strides_, invariant_index);
                size_t y_invariant_offset = get_offset_from_index<NumInvariantDim>(
                    arg.y_invariant_strides_, invariant_index);
                AccDataType mean     = type_convert<AccDataType>(0.0f);
                AccDataType variance = type_convert<AccDataType>(0.0f);
                int32_t curr_count   = 0;

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
                AccDataType invVariance =
                    type_convert<AccDataType>(1.0f) / ck::math::sqrt(arg.epsilon_ + variance);

                // save the mean/inv-variance if required
                if(arg.resultSave)
                {
                    size_t offset = get_offset_from_index<NumInvariantDim>(arg.bnMeanVarStrides_,
                                                                           invariant_index);

                    arg.resultSaveMean_[offset]        = type_convert<MeanVarDataType>(mean);
                    arg.resultSaveInvVariance_[offset] = type_convert<MeanVarDataType>(invVariance);
                };

                // update the moving average if required
                if(arg.resultRunning)
                {
                    size_t offset = get_offset_from_index<NumInvariantDim>(arg.bnMeanVarStrides_,
                                                                           invariant_index);

                    AccDataType oneMinusAverageFactor =
                        type_convert<AccDataType>(1.0) - arg.averageFactor_;
                    arg.resultRunningMean_[offset] = type_convert<MeanVarDataType>(
                        type_convert<AccDataType>(arg.resultRunningMean_[offset]) *
                            oneMinusAverageFactor +
                        mean * arg.averageFactor_);
                    arg.resultRunningVariance_[offset] = type_convert<MeanVarDataType>(
                        arg.resultRunningVariance_[offset] * oneMinusAverageFactor +
                        variance * arg.averageFactor_);
                };

                size_t scale_offset =
                    get_offset_from_index<NumInvariantDim>(arg.bnScaleStrides_, invariant_index);
                size_t bias_offset =
                    get_offset_from_index<NumInvariantDim>(arg.bnBiasStrides_, invariant_index);

                AccDataType scale = type_convert<AccDataType>(arg.bnScale_[scale_offset]);
                AccDataType bias  = type_convert<AccDataType>(arg.bnBias_[bias_offset]);

                // Normalization
                for(const auto& reduce_index : arg.reduce_index_set_)
                {
                    size_t x_reduce_offset = get_offset_from_index<NumBatchNormReduceDim>(
                        arg.x_reduce_strides_, reduce_index);
                    size_t y_reduce_offset = get_offset_from_index<NumBatchNormReduceDim>(
                        arg.y_reduce_strides_, reduce_index);

                    auto x_offset = x_invariant_offset + x_reduce_offset;
                    auto y_offset = y_invariant_offset + y_reduce_offset;

                    AccDataType x = type_convert<AccDataType>(arg.p_x_[x_offset]);

                    AccDataType norm_x = (x - mean) * invVariance;

                    AccDataType y = scale * norm_x + bias;

                    arg.y_elementwise_op_(y, y);

                    arg.p_y_[y_offset] = type_convert<YDataType>(y);
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
    MakeArgumentPointer(const std::array<index_t, 4> xyLengths,
                        const std::array<index_t, 4> xStrides,
                        const std::array<index_t, 4> yStrides,
                        const std::array<int, 3> reduceDims,
                        const std::array<index_t, 1> bnScaleBiasMeanVarLengths,
                        const std::array<index_t, 1> bnScaleStrides,
                        const std::array<index_t, 1> bnBiasStrides,
                        const std::array<index_t, 1> bnMeanVarStrides,
                        const void* p_x,
                        const void* bnScale,
                        const void* bnBias,
                        double epsilon,
                        const YElementwiseOp y_elementwise_op,
                        void* p_y,
                        void* resultSaveMean,
                        void* resultSaveInvVariance,
                        double averageFactor,
                        void* resultRunningMean,
                        void* resultRunningVariance) override
    {
        return std::make_unique<Argument>(xyLengths,
                                          xStrides,
                                          yStrides,
                                          reduceDims,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleStrides,
                                          bnBiasStrides,
                                          bnMeanVarStrides,
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const ScaleDataType*>(bnScale),
                                          static_cast<const BiasDataType*>(bnBias),
                                          epsilon,
                                          y_elementwise_op,
                                          static_cast<YDataType*>(p_y),
                                          static_cast<MeanVarDataType*>(resultSaveMean),
                                          static_cast<MeanVarDataType*>(resultSaveInvVariance),
                                          averageFactor,
                                          static_cast<MeanVarDataType*>(resultRunningMean),
                                          static_cast<MeanVarDataType*>(resultRunningVariance));
    };

    std::unique_ptr<device::BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "Reference_BatchNorm_Forward" << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
