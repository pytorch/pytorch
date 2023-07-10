// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_batchnorm_forward.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/device/welford_helper.hpp"
#include "ck/tensor_operation/gpu/grid/batchnorm_multiblock/gridwise_multiblock_welford_first_half.hpp"
#include "ck/tensor_operation/gpu/grid/batchnorm_multiblock/gridwise_multiblock_welford_second_half_batchnorm_forward_final.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batchnorm_forward_blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename YElementwiseOp,
          index_t Rank,
          index_t NumBatchNormReduceDim,
          bool UseMultiblockInK,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcYDstVectorDim,
          index_t XSrcVectorSize,
          index_t YDstVectorSize,
          index_t ScaleSrcVectorSize,
          index_t BiasSrcVectorSize,
          index_t MeanVarSrcDstVectorSize>
struct DeviceBatchNormFwdImpl : public DeviceBatchNormFwd<XDataType,
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
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert((XSrcYDstVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcYDstVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr index_t NumInvariantDim = Rank - NumBatchNormReduceDim;

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto MakeXY2dDescriptor(const std::array<index_t, Rank>& xyLengths,
                                   const std::array<index_t, Rank>& xyStrides,
                                   int blkGroupSize,
                                   int numBlockTileIteration)
    {
        const auto tupleXYLengths =
            generate_tuple([&](auto I) { return xyLengths[I]; }, Number<Rank>{});
        const auto tupleXYStrides =
            generate_tuple([&](auto I) { return xyStrides[I]; }, Number<Rank>{});

        const auto raw_grid_desc = make_naive_tensor_descriptor(tupleXYLengths, tupleXYStrides);

        const auto grid_desc_m_k = [&]() {
            using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
            using ReduceDims    = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

            const auto reduceDimLengths =
                generate_tuple([&](auto I) { return xyLengths[NumInvariantDim + I]; },
                               Number<NumBatchNormReduceDim>{});
            const auto invariantDimLengths =
                generate_tuple([&](auto I) { return xyLengths[I]; }, Number<NumInvariantDim>{});

            return transform_tensor_descriptor(raw_grid_desc,
                                               make_tuple(make_merge_transform(invariantDimLengths),
                                                          make_merge_transform(reduceDimLengths)),
                                               make_tuple(InvariantDims{}, ReduceDims{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }();

        const auto invariantLength = grid_desc_m_k.GetLength(Number<0>{});
        const auto reduceLength    = grid_desc_m_k.GetLength(Number<1>{});

        const int workSizePerBlock = K_BlockTileSize * numBlockTileIteration;
        const auto mPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto kPad = workSizePerBlock * blkGroupSize - reduceLength;

        auto grid_desc_m_k_padded =
            transform_tensor_descriptor(grid_desc_m_k,
                                        make_tuple(make_right_pad_transform(invariantLength, mPad),
                                                   make_right_pad_transform(reduceLength, kPad)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (grid_desc_m_k_padded);
    };

    static auto MakeMeanVarCountOutputMG2dDescriptor(int invariantLength, int blkGroupSize)
    {
        const auto grid_desc_m_g =
            make_naive_tensor_descriptor_packed(make_tuple(invariantLength, blkGroupSize));

        const auto mPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto grid_desc_m_g_padded =
            transform_tensor_descriptor(grid_desc_m_g,
                                        make_tuple(make_right_pad_transform(invariantLength, mPad),
                                                   make_pass_through_transform(blkGroupSize)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (grid_desc_m_g_padded);
    };

    static auto MakeMeanVarCountInputMK2dDescriptor(int invariantLength, int blkGroupSize)
    {
        const auto reduceLength = blkGroupSize;
        const auto grid_desc_m_k =
            make_naive_tensor_descriptor_packed(make_tuple(invariantLength, reduceLength));

        const auto mPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto kPad =
            math::integer_least_multiple(reduceLength, KThreadClusterSize) - reduceLength;

        auto grid_desc_m_k_padded =
            transform_tensor_descriptor(grid_desc_m_k,
                                        make_tuple(make_right_pad_transform(invariantLength, mPad),
                                                   make_right_pad_transform(reduceLength, kPad)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (grid_desc_m_k_padded);
    };

    static auto
    MakeScaleBiasMeanVar1dDescriptor(const std::array<index_t, NumInvariantDim>& lengths,
                                     const std::array<index_t, NumInvariantDim>& strides)
    {
        const auto tupleLengths =
            generate_tuple([&](auto I) { return lengths[I]; }, Number<NumInvariantDim>{});
        const auto tupleStrides =
            generate_tuple([&](auto I) { return strides[I]; }, Number<NumInvariantDim>{});

        auto raw_grid_desc = make_naive_tensor_descriptor(tupleLengths, tupleStrides);

        auto grid_desc_m = transform_tensor_descriptor(
            raw_grid_desc,
            make_tuple(make_merge_transform(tupleLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto invariantLength = grid_desc_m.GetLength(Number<0>{});

        const auto mPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto grid_desc_m_padded =
            transform_tensor_descriptor(grid_desc_m,
                                        make_tuple(make_right_pad_transform(invariantLength, mPad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (grid_desc_m_padded);
    };

    using XYGridDesc_M_K             = decltype(MakeXY2dDescriptor({1}, {1}, 1, 1));
    using ScaleBiasMeanVarGridDesc_M = decltype(MakeScaleBiasMeanVar1dDescriptor({1}, {1}));

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, Rank> xyLengths,
                 const std::array<index_t, Rank> xStrides,
                 const std::array<index_t, Rank> yStrides,
                 const std::array<int, NumBatchNormReduceDim> reduceDims,
                 const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths,
                 const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleStrides,
                 const std::array<index_t, Rank - NumBatchNormReduceDim> bnBiasStrides,
                 const std::array<index_t, Rank - NumBatchNormReduceDim> bnMeanVarStrides,
                 const XDataType* p_x,
                 const ScaleDataType* p_scale,
                 const BiasDataType* p_bias,
                 const YElementwiseOp y_elementwise_op,
                 double epsilon,
                 YDataType* p_y,
                 MeanVarDataType* resultSaveMean,
                 MeanVarDataType* resultSaveInvVariance,
                 double averageFactor,
                 MeanVarDataType* resultRunningMean,
                 MeanVarDataType* resultRunningVariance)
            : bnScaleBiasMeanVarLengths_(bnScaleBiasMeanVarLengths),
              bnScaleStrides_(bnScaleStrides),
              bnBiasStrides_(bnBiasStrides),
              bnMeanVarStrides_(bnMeanVarStrides),
              p_x_(p_x),
              p_scale_(p_scale),
              p_bias_(p_bias),
              y_elementwise_op_(y_elementwise_op),
              p_y_(p_y),
              resultSaveMean_(resultSaveMean),
              resultSaveInvVariance_(resultSaveInvVariance),
              resultRunningMean_(resultRunningMean),
              resultRunningVariance_(resultRunningVariance)
        {
            xyLengths_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(xyLengths, reduceDims);
            xStrides_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(xStrides, reduceDims);
            yStrides_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(yStrides, reduceDims);

            std::tie(invariant_length_, reduce_length_) =
                get_2d_lengths<Rank, NumBatchNormReduceDim>(xyLengths_);

            epsilon_       = type_convert<AccDataType>(epsilon);
            averageFactor_ = type_convert<AccDataType>(averageFactor);

            updateMovingAverage_ =
                (resultRunningMean != nullptr && resultRunningVariance != nullptr);
            saveMeanInvVariance_ = (resultSaveMean != nullptr && resultSaveInvVariance_ != nullptr);

            if(UseMultiblockInK)
            {
                int iterations = 1;
                while(true)
                {
                    int testBlkGroupSize = (reduce_length_ + (K_BlockTileSize * iterations) - 1) /
                                           (K_BlockTileSize * iterations);

                    // we want the blkGroupSize be not more than 128
                    if(testBlkGroupSize <= 128)
                        break;

                    iterations++;
                };

                blkGroupSize_ = (reduce_length_ + (K_BlockTileSize * iterations) - 1) /
                                (K_BlockTileSize * iterations);

                numBlockTileIteration_ = iterations;
            }
            else
            {
                blkGroupSize_          = 1;
                numBlockTileIteration_ = (reduce_length_ + K_BlockTileSize - 1) / K_BlockTileSize;
            };

            gridSize_ = (invariant_length_ + M_BlockTileSize - 1) / M_BlockTileSize * blkGroupSize_;

            x_grid_desc_m_k_ =
                MakeXY2dDescriptor(xyLengths_, xStrides_, blkGroupSize_, numBlockTileIteration_);
            y_grid_desc_m_k_ =
                MakeXY2dDescriptor(xyLengths_, yStrides_, blkGroupSize_, numBlockTileIteration_);
            scale_grid_desc_m_ =
                MakeScaleBiasMeanVar1dDescriptor(bnScaleBiasMeanVarLengths, bnScaleStrides_);
            bias_grid_desc_m_ =
                MakeScaleBiasMeanVar1dDescriptor(bnScaleBiasMeanVarLengths, bnBiasStrides_);
            mean_var_grid_desc_m_ =
                MakeScaleBiasMeanVar1dDescriptor(bnScaleBiasMeanVarLengths, bnMeanVarStrides_);
        }

        AccDataType epsilon_;
        AccDataType averageFactor_;

        bool updateMovingAverage_;
        bool saveMeanInvVariance_;

        std::array<index_t, Rank> xyLengths_;
        std::array<index_t, Rank> xStrides_;
        std::array<index_t, Rank> yStrides_;

        std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths_;
        std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleStrides_;
        std::array<index_t, Rank - NumBatchNormReduceDim> bnBiasStrides_;
        std::array<index_t, Rank - NumBatchNormReduceDim> bnMeanVarStrides_;

        const XDataType* p_x_;
        const ScaleDataType* p_scale_;
        const BiasDataType* p_bias_;
        const YElementwiseOp y_elementwise_op_;
        YDataType* p_y_;

        MeanVarDataType* resultSaveMean_;
        MeanVarDataType* resultSaveInvVariance_;

        MeanVarDataType* resultRunningMean_;
        MeanVarDataType* resultRunningVariance_;

        long_index_t invariant_length_;
        long_index_t reduce_length_;

        int blkGroupSize_;
        int numBlockTileIteration_;
        size_t gridSize_;

        XYGridDesc_M_K x_grid_desc_m_k_;
        XYGridDesc_M_K y_grid_desc_m_k_;
        ScaleBiasMeanVarGridDesc_M scale_grid_desc_m_;
        ScaleBiasMeanVarGridDesc_M bias_grid_desc_m_;
        ScaleBiasMeanVarGridDesc_M mean_var_grid_desc_m_;

        void* workspace_mean_;
        void* workspace_variance_;
        void* workspace_count_;
    };

    size_t GetWorkSpaceSize(const BaseArgument* pArg) const override
    {
        const Argument* pArg_ = dynamic_cast<const Argument*>(pArg);

        size_t workspace_size = 0;

        if(UseMultiblockInK && pArg_->blkGroupSize_ > 1)
        {
            // workspace for welford intermediate mean
            workspace_size +=
                pArg_->invariant_length_ * pArg_->blkGroupSize_ * sizeof(MeanVarDataType) + 64;

            // workspace for welford intermediate variance
            workspace_size +=
                pArg_->invariant_length_ * pArg_->blkGroupSize_ * sizeof(MeanVarDataType) + 64;

            // workspace for welford intermediate count
            workspace_size +=
                pArg_->invariant_length_ * pArg_->blkGroupSize_ * sizeof(int32_t) + 64;
        }

        return (workspace_size);
    };

    void SetWorkSpacePointer(BaseArgument* pArg, void* p_workspace) const override
    {
        Argument* pArg_ = dynamic_cast<Argument*>(pArg);

        pArg_->p_workspace_ = p_workspace;

        if(UseMultiblockInK && pArg_->blkGroupSize_ > 1)
        {

            // setup buffer used for intermediate welford mean
            pArg_->workspace_mean_ = static_cast<char*>(pArg_->p_workspace_);

            index_t mean_space_sz =
                pArg_->invariant_length_ * pArg_->blkGroupSize_ * sizeof(MeanVarDataType);

            mean_space_sz = math::integer_least_multiple(mean_space_sz, 64);

            // setup buffer used for intermediate welford varirance
            pArg_->workspace_variance_ =
                reinterpret_cast<char*>(pArg_->workspace_mean_) + mean_space_sz;

            index_t variance_space_sz =
                pArg_->invariant_length_ * pArg_->blkGroupSize_ * sizeof(MeanVarDataType);

            variance_space_sz = math::integer_least_multiple(variance_space_sz, 64);

            // setup buffer used for intermediate welfor count
            pArg_->workspace_count_ =
                reinterpret_cast<char*>(pArg_->workspace_variance_) + variance_space_sz;
        };
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0;

            if(UseMultiblockInK && arg.blkGroupSize_ > 1)
            {
                using GetReduceCountPerThreadFunctor =
                    GetReduceCountPerThreadForMultiblockWelford<K_BlockTileSize, KThreadSliceSize>;

                GetReduceCountPerThreadFunctor get_reduce_count_per_thread(
                    arg.blkGroupSize_, arg.numBlockTileIteration_, arg.reduce_length_);

                const auto mean_var_count_grid_desc_m_g =
                    DeviceBatchNormFwdImpl::MakeMeanVarCountOutputMG2dDescriptor(
                        arg.invariant_length_, arg.blkGroupSize_);

                const auto mean_var_count_grid_desc_m_k =
                    DeviceBatchNormFwdImpl::MakeMeanVarCountInputMK2dDescriptor(
                        arg.invariant_length_, arg.blkGroupSize_);

                using MeanVarCountGridDesc_M_G = decltype(mean_var_count_grid_desc_m_g);
                using MeanVarCountGridDesc_M_K = decltype(mean_var_count_grid_desc_m_k);

                using GridwiseMultiblockWelfordFirstHalf_ =
                    GridwiseMultiblockWelfordFirstHalf<XDataType,
                                                       AccDataType,
                                                       MeanVarDataType,
                                                       XYGridDesc_M_K,
                                                       MeanVarCountGridDesc_M_G,
                                                       GetReduceCountPerThreadFunctor,
                                                       BlockSize,
                                                       MThreadClusterSize,
                                                       KThreadClusterSize,
                                                       MThreadSliceSize,
                                                       KThreadSliceSize,
                                                       XSrcYDstVectorDim,
                                                       XSrcVectorSize>;

                using GridwiseWelfordSecondHalfBatchNormForwardFinal_ =
                    GridwiseWelfordSecondHalfBatchNormForwardFinal<XDataType,
                                                                   YDataType,
                                                                   AccDataType,
                                                                   ScaleDataType,
                                                                   BiasDataType,
                                                                   MeanVarDataType,
                                                                   YElementwiseOp,
                                                                   XYGridDesc_M_K,
                                                                   MeanVarCountGridDesc_M_K,
                                                                   ScaleBiasMeanVarGridDesc_M,
                                                                   ScaleBiasMeanVarGridDesc_M,
                                                                   BlockSize,
                                                                   MThreadClusterSize,
                                                                   KThreadClusterSize,
                                                                   MThreadSliceSize,
                                                                   KThreadSliceSize,
                                                                   XSrcYDstVectorDim,
                                                                   XSrcVectorSize,
                                                                   YDstVectorSize,
                                                                   ScaleSrcVectorSize,
                                                                   BiasSrcVectorSize,
                                                                   MeanVarSrcDstVectorSize>;

                index_t numMeanVarCountBlockTileIteration =
                    (arg.blkGroupSize_ + KThreadClusterSize - 1) / KThreadClusterSize;

                const auto kern_multiblock_welford_first_half =
                    kernel_multiblock_welford_first_half<GridwiseMultiblockWelfordFirstHalf_,
                                                         XDataType,
                                                         MeanVarDataType,
                                                         XYGridDesc_M_K,
                                                         MeanVarCountGridDesc_M_G,
                                                         GetReduceCountPerThreadFunctor>;

                const auto kern_welford_second_half_batchnorm_forward_final =
                    kernel_welford_second_half_batchnorm_forward_final<
                        GridwiseWelfordSecondHalfBatchNormForwardFinal_,
                        XDataType,
                        YDataType,
                        AccDataType,
                        ScaleDataType,
                        BiasDataType,
                        MeanVarDataType,
                        YElementwiseOp,
                        XYGridDesc_M_K,
                        MeanVarCountGridDesc_M_K,
                        ScaleBiasMeanVarGridDesc_M,
                        ScaleBiasMeanVarGridDesc_M>;

                avg_time +=
                    launch_and_time_kernel(stream_config,
                                           kern_multiblock_welford_first_half,
                                           dim3(arg.gridSize_),
                                           dim3(BlockSize),
                                           0,
                                           arg.x_grid_desc_m_k_,
                                           mean_var_count_grid_desc_m_g,
                                           get_reduce_count_per_thread,
                                           arg.numBlockTileIteration_,
                                           arg.p_x_,
                                           static_cast<MeanVarDataType*>(arg.workspace_mean_),
                                           static_cast<MeanVarDataType*>(arg.workspace_variance_),
                                           static_cast<int32_t*>(arg.workspace_count_));

                avg_time +=
                    launch_and_time_kernel(stream_config,
                                           kern_welford_second_half_batchnorm_forward_final,
                                           dim3(arg.gridSize_),
                                           dim3(BlockSize),
                                           0,
                                           arg.x_grid_desc_m_k_,
                                           arg.y_grid_desc_m_k_,
                                           mean_var_count_grid_desc_m_k,
                                           arg.scale_grid_desc_m_,
                                           arg.bias_grid_desc_m_,
                                           arg.mean_var_grid_desc_m_,
                                           arg.blkGroupSize_,
                                           arg.numBlockTileIteration_,
                                           numMeanVarCountBlockTileIteration,
                                           arg.epsilon_,
                                           static_cast<MeanVarDataType*>(arg.workspace_mean_),
                                           static_cast<MeanVarDataType*>(arg.workspace_variance_),
                                           static_cast<int32_t*>(arg.workspace_count_),
                                           arg.p_x_,
                                           arg.p_scale_,
                                           arg.p_bias_,
                                           arg.y_elementwise_op_,
                                           arg.p_y_,
                                           arg.updateMovingAverage_,
                                           arg.averageFactor_,
                                           arg.resultRunningMean_,
                                           arg.resultRunningVariance_,
                                           arg.saveMeanInvVariance_,
                                           arg.resultSaveMean_,
                                           arg.resultSaveInvVariance_);
            }
            else
            {
                using GetReduceCountPerThreadFunctor =
                    GetReduceCountPerThreadForBlockwiseWelford<K_BlockTileSize, KThreadSliceSize>;

                GetReduceCountPerThreadFunctor get_reduce_count_per_thread(
                    arg.numBlockTileIteration_, arg.reduce_length_);

                using GridwiseBatchNormForwardWithBlockwiseWelford_ =
                    GridwiseBatchNormForwardWithBlockwiseWelford<XDataType,
                                                                 YDataType,
                                                                 AccDataType,
                                                                 ScaleDataType,
                                                                 BiasDataType,
                                                                 MeanVarDataType,
                                                                 YElementwiseOp,
                                                                 XYGridDesc_M_K,
                                                                 ScaleBiasMeanVarGridDesc_M,
                                                                 ScaleBiasMeanVarGridDesc_M,
                                                                 GetReduceCountPerThreadFunctor,
                                                                 BlockSize,
                                                                 MThreadClusterSize,
                                                                 KThreadClusterSize,
                                                                 MThreadSliceSize,
                                                                 KThreadSliceSize,
                                                                 XSrcYDstVectorDim,
                                                                 XSrcVectorSize,
                                                                 YDstVectorSize,
                                                                 ScaleSrcVectorSize,
                                                                 BiasSrcVectorSize,
                                                                 MeanVarSrcDstVectorSize>;

                const auto kern_batchnorm_fwd = kernel_batchnorm_forward_with_blockwise_welford<
                    GridwiseBatchNormForwardWithBlockwiseWelford_,
                    XDataType,
                    YDataType,
                    AccDataType,
                    ScaleDataType,
                    BiasDataType,
                    MeanVarDataType,
                    YElementwiseOp,
                    XYGridDesc_M_K,
                    ScaleBiasMeanVarGridDesc_M,
                    ScaleBiasMeanVarGridDesc_M,
                    GetReduceCountPerThreadFunctor>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kern_batchnorm_fwd,
                                                   dim3(arg.gridSize_),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.x_grid_desc_m_k_,
                                                   arg.y_grid_desc_m_k_,
                                                   arg.scale_grid_desc_m_,
                                                   arg.bias_grid_desc_m_,
                                                   arg.mean_var_grid_desc_m_,
                                                   get_reduce_count_per_thread,
                                                   arg.numBlockTileIteration_,
                                                   arg.epsilon_,
                                                   arg.p_x_,
                                                   arg.p_scale_,
                                                   arg.p_bias_,
                                                   arg.y_elementwise_op_,
                                                   arg.p_y_,
                                                   arg.updateMovingAverage_, // true or false
                                                   arg.averageFactor_,
                                                   arg.resultRunningMean_,
                                                   arg.resultRunningVariance_,
                                                   arg.saveMeanInvVariance_, // true or false
                                                   arg.resultSaveMean_,
                                                   arg.resultSaveInvVariance_);
            };

            return (avg_time);
        };

        float Run(const BaseArgument* pArg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(pArg), stream_config);
        };
    };

    bool IsSupportedArgument(const BaseArgument* pArg) override
    {
        const Argument* pArg_ = dynamic_cast<const Argument*>(pArg);

        if constexpr(XSrcYDstVectorDim == 0)
        {
            if(pArg_->xStrides_[NumInvariantDim - 1] != 1 ||
               pArg_->yStrides_[NumInvariantDim - 1] != 1)
                return false;

            if(pArg_->xyLengths_[NumInvariantDim - 1] % XSrcVectorSize != 0 ||
               pArg_->xyLengths_[NumInvariantDim - 1] % YDstVectorSize != 0)
                return false;
        }
        else
        {
            if(pArg_->xStrides_[Rank - 1] != 1 || pArg_->yStrides_[Rank - 1] != 1)
                return false;

            if(pArg_->xyLengths_[Rank - 1] % XSrcVectorSize != 0 ||
               pArg_->xyLengths_[Rank - 1] % YDstVectorSize != 0)
                return false;
        };

        if(pArg_->bnScaleStrides_[NumInvariantDim - 1] != 1 && ScaleSrcVectorSize != 1)
            return false;
        if(pArg_->bnBiasStrides_[NumInvariantDim - 1] != 1 && BiasSrcVectorSize != 1)
            return false;

        if(pArg_->bnScaleBiasMeanVarLengths_[NumInvariantDim - 1] % ScaleSrcVectorSize != 0)
            return false;
        if(pArg_->bnScaleBiasMeanVarLengths_[NumInvariantDim - 1] % BiasSrcVectorSize != 0)
            return false;

        if(pArg_->bnMeanVarStrides_[NumInvariantDim - 1] != 1 && MeanVarSrcDstVectorSize != 1)
            return false;

        if(pArg_->bnScaleBiasMeanVarLengths_[NumInvariantDim - 1] % MeanVarSrcDstVectorSize != 0)
            return false;

        bool is_valid = true;

        static_for<0, NumInvariantDim, 1>{}([&](auto I) {
            if(pArg_->xyLengths_[I] != pArg_->bnScaleBiasMeanVarLengths_[I])
                is_valid = false;
        });

        if(!is_valid)
            return false;

        return true;
    };

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const std::array<index_t, Rank> xyLengths,
        const std::array<index_t, Rank> xStrides,
        const std::array<index_t, Rank> yStrides,
        const std::array<int, NumBatchNormReduceDim> reduceDims,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleStrides,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnBiasStrides,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnMeanVarStrides,
        const void* p_x,
        const void* p_scale,
        const void* p_bias,
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
                                          static_cast<const ScaleDataType*>(p_scale),
                                          static_cast<const BiasDataType*>(p_bias),
                                          y_elementwise_op,
                                          epsilon,
                                          static_cast<YDataType*>(p_y),
                                          static_cast<MeanVarDataType*>(resultSaveMean),
                                          static_cast<MeanVarDataType*>(resultSaveInvVariance),
                                          averageFactor,
                                          static_cast<MeanVarDataType*>(resultRunningMean),
                                          static_cast<MeanVarDataType*>(resultRunningVariance));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchNormFwdImpl<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str << "XSrcYDstVectorDim_" << XSrcYDstVectorDim  << ",";
        str << "VectorSize_X" << XSrcVectorSize << "_scale_" << ScaleSrcVectorSize << "_bias_" << BiasSrcVectorSize << "_mean_var_" << MeanVarSrcDstVectorSize << "_Y" << YDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
