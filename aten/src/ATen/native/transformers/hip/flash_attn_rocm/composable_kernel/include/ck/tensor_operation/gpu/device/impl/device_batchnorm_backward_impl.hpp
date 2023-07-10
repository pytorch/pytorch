// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_batchnorm_backward.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batchnorm_backward_blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/grid/batchnorm_multiblock/gridwise_multiblock_welford_first_half.hpp"
#include "ck/tensor_operation/gpu/grid/batchnorm_multiblock/gridwise_multiblock_welford_second_half_multiblock_reduce_first_half.hpp"
#include "ck/tensor_operation/gpu/grid/batchnorm_multiblock/gridwise_multiblock_reduce_second_half_batchnorm_backward_final.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/welford_helper.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          index_t Rank,
          index_t NumBatchNormReduceDim,
          bool UseMultiblockInK,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XDyDxVectorDim,
          index_t XSrcVectorSize,
          index_t DySrcVectorSize,
          index_t DxDstVectorSize,
          index_t ScaleSrcVectorSize,
          index_t DscaleDbiasDstVectorSize,
          index_t MeanVarSrcVectorSize>
struct DeviceBatchNormBwdImpl : public DeviceBatchNormBwd<XDataType,
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
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert((XDyDxVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0 &&
                   MThreadSliceSize % DySrcVectorSize == 0 &&
                   MThreadSliceSize % DxDstVectorSize == 0) ||
                      (XDyDxVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0 &&
                       KThreadSliceSize % DySrcVectorSize == 0 &&
                       KThreadSliceSize % DxDstVectorSize == 0),
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

    static auto MakeMultiblockFirstReduceOutputMG2dDescriptor(int invariantLength, int blkGroupSize)
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

    static auto MakeMultiblockFinalReduceInputMK2dDescriptor(int invariantLength, int blkGroupSize)
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

    using XYGridDesc_M_K      = decltype(MakeXY2dDescriptor({1}, {1}, 1, 1));
    using ScaleBiasGridDesc_M = decltype(MakeScaleBiasMeanVar1dDescriptor({1}, {1}));
    using MeanVarGridDesc_M   = ScaleBiasGridDesc_M;

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, Rank> xyLengths,
                 const std::array<index_t, Rank> xStrides,
                 const std::array<index_t, Rank> dyStrides,
                 const std::array<index_t, Rank> dxStrides,
                 const std::array<int, NumBatchNormReduceDim> reduceDims,
                 const std::array<ck::index_t, NumInvariantDim> bnScaleBiasMeanVarLengths,
                 const std::array<ck::index_t, NumInvariantDim> bnScaleStrides,
                 const std::array<ck::index_t, NumInvariantDim> bnDscaleDbiasStrides,
                 const std::array<ck::index_t, NumInvariantDim> bnMeanVarStrides,
                 const XDataType* p_x,
                 const DyDataType* p_dy,
                 const ScaleDataType* p_scale,
                 const MeanVarDataType* p_savedMean,
                 const MeanVarDataType* p_savedInvVar,
                 const DyElementwiseOp dy_elementwise_op,
                 double epsilon,
                 DxDataType* p_dx,
                 DscaleDbiasDataType* p_dscale,
                 DscaleDbiasDataType* p_dbias)
            : bnScaleBiasMeanVarLengths_(bnScaleBiasMeanVarLengths),
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
            xyLengths_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(xyLengths, reduceDims);
            xStrides_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(xStrides, reduceDims);
            dyStrides_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(dyStrides, reduceDims);
            dxStrides_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(dxStrides, reduceDims);

            std::tie(invariant_length, reduce_length) =
                get_2d_lengths<Rank, NumBatchNormReduceDim>(xyLengths_);

            epsilon_ = type_convert<AccDataType>(epsilon);

            haveSavedMeanInvVar_ = (p_savedMean_ != nullptr && p_savedInvVar_ != nullptr);

            if(UseMultiblockInK)
            {
                int iterations = 1;
                while(true)
                {
                    int testBlkGroupSize = (reduce_length + (K_BlockTileSize * iterations) - 1) /
                                           (K_BlockTileSize * iterations);

                    // we want the blkGroupSize be not more than 128
                    if(testBlkGroupSize <= 128)
                        break;

                    iterations++;
                };

                blkGroupSize = (reduce_length + (K_BlockTileSize * iterations) - 1) /
                               (K_BlockTileSize * iterations);

                numBlockTileIteration = iterations;
            }
            else
            {
                blkGroupSize          = 1;
                numBlockTileIteration = (reduce_length + K_BlockTileSize - 1) / K_BlockTileSize;
            };

            gridSize = (invariant_length + M_BlockTileSize - 1) / M_BlockTileSize * blkGroupSize;

            x_grid_desc_m_k =
                MakeXY2dDescriptor(xyLengths_, xStrides_, blkGroupSize, numBlockTileIteration);
            dy_grid_desc_m_k =
                MakeXY2dDescriptor(xyLengths_, dyStrides_, blkGroupSize, numBlockTileIteration);
            dx_grid_desc_m_k =
                MakeXY2dDescriptor(xyLengths_, dxStrides_, blkGroupSize, numBlockTileIteration);
            scale_grid_desc_m =
                MakeScaleBiasMeanVar1dDescriptor(bnScaleBiasMeanVarLengths, bnScaleStrides);
            dscale_dbias_grid_desc_m =
                MakeScaleBiasMeanVar1dDescriptor(bnScaleBiasMeanVarLengths, bnDscaleDbiasStrides);
            mean_var_grid_desc_m =
                MakeScaleBiasMeanVar1dDescriptor(bnScaleBiasMeanVarLengths, bnMeanVarStrides);
        }

        AccDataType epsilon_;

        bool haveSavedMeanInvVar_;

        std::array<index_t, Rank> xyLengths_;
        std::array<index_t, Rank> xStrides_;
        std::array<index_t, Rank> dyStrides_;
        std::array<index_t, Rank> dxStrides_;

        std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths_;
        std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleStrides_;
        std::array<index_t, Rank - NumBatchNormReduceDim> bnDscaleDbiasStrides_;
        std::array<index_t, Rank - NumBatchNormReduceDim> bnMeanVarStrides_;

        const XDataType* p_x_;
        const DyDataType* p_dy_;
        const ScaleDataType* p_scale_;
        const MeanVarDataType* p_savedMean_;
        const MeanVarDataType* p_savedInvVar_;
        const DyElementwiseOp dy_elementwise_op_;
        DxDataType* p_dx_;
        DscaleDbiasDataType* p_dscale_;
        DscaleDbiasDataType* p_dbias_;

        long_index_t invariant_length;
        long_index_t reduce_length;

        int blkGroupSize;
        int numBlockTileIteration;
        size_t gridSize;

        XYGridDesc_M_K x_grid_desc_m_k;
        XYGridDesc_M_K dy_grid_desc_m_k;
        XYGridDesc_M_K dx_grid_desc_m_k;
        ScaleBiasGridDesc_M scale_grid_desc_m;
        ScaleBiasGridDesc_M dscale_dbias_grid_desc_m;
        MeanVarGridDesc_M mean_var_grid_desc_m;

        void* workspace_mean;
        void* workspace_variance;
        void* workspace_count;

        void* workspace_savedMean;
        void* workspace_savedInvVar;

        void* workspace_reduce_dscale;
        void* workspace_reduce_dbias;
    };

    size_t GetWorkSpaceSize(const BaseArgument* pArg) const override
    {
        const Argument* pArg_ = dynamic_cast<const Argument*>(pArg);

        size_t workspace_size = 0;

        if(UseMultiblockInK && pArg_->blkGroupSize > 1)
        {
            // workspace for the partial reduced result for dscale
            workspace_size +=
                pArg_->invariant_length * pArg_->blkGroupSize * sizeof(DscaleDbiasDataType) + 64;

            // workspace for the partial reduced result for dbias
            workspace_size +=
                pArg_->invariant_length * pArg_->blkGroupSize * sizeof(DscaleDbiasDataType) + 64;

            if(!pArg_->haveSavedMeanInvVar_)
            {
                // workspace for welford intermediate mean
                workspace_size +=
                    pArg_->invariant_length * pArg_->blkGroupSize * sizeof(MeanVarDataType) + 64;

                // workspace for welford intermediate variance
                workspace_size +=
                    pArg_->invariant_length * pArg_->blkGroupSize * sizeof(MeanVarDataType) + 64;

                // workspace for welford intermediate count
                workspace_size +=
                    pArg_->invariant_length * pArg_->blkGroupSize * sizeof(int32_t) + 64;

                // workspace for welford result mean
                workspace_size += pArg_->invariant_length * sizeof(MeanVarDataType) + 64;

                // workspace for welford result inv_variance
                workspace_size += pArg_->invariant_length * sizeof(MeanVarDataType) + 64;
            };
        }

        return (workspace_size);
    };

    void SetWorkSpacePointer(BaseArgument* pArg, void* p_workspace) const override
    {
        Argument* pArg_ = dynamic_cast<Argument*>(pArg);

        pArg_->p_workspace_ = p_workspace;

        index_t space_sz;

        // setup buffer for the partial reduced result for dscale
        pArg_->workspace_reduce_dscale = pArg_->p_workspace_;

        space_sz = pArg_->invariant_length * pArg_->blkGroupSize * sizeof(DscaleDbiasDataType);
        space_sz = math::integer_least_multiple(space_sz, 64);

        // setup buffer for the partial reduced result for dbias
        pArg_->workspace_reduce_dbias =
            reinterpret_cast<char*>(pArg_->workspace_reduce_dscale) + space_sz;

        if(UseMultiblockInK && pArg_->blkGroupSize > 1)
        {
            space_sz = pArg_->invariant_length * pArg_->blkGroupSize * sizeof(DscaleDbiasDataType);
            space_sz = math::integer_least_multiple(space_sz, 64);

            // setup buffer for welford intermediate mean
            pArg_->workspace_mean =
                reinterpret_cast<char*>(pArg_->workspace_reduce_dbias) + space_sz;

            space_sz = pArg_->invariant_length * pArg_->blkGroupSize * sizeof(MeanVarDataType);
            space_sz = math::integer_least_multiple(space_sz, 64);

            // setup buffer for welford intermediate varirance
            pArg_->workspace_variance = reinterpret_cast<char*>(pArg_->workspace_mean) + space_sz;

            space_sz = pArg_->invariant_length * pArg_->blkGroupSize * sizeof(MeanVarDataType);
            space_sz = math::integer_least_multiple(space_sz, 64);

            // setup buffer for welford intermediate count
            pArg_->workspace_count = reinterpret_cast<char*>(pArg_->workspace_variance) + space_sz;

            space_sz = pArg_->invariant_length * pArg_->blkGroupSize * sizeof(int32_t);
            space_sz = math::integer_least_multiple(space_sz, 64);

            // setup buffer for welford result mean
            pArg_->workspace_savedMean = reinterpret_cast<char*>(pArg_->workspace_count) + space_sz;

            space_sz = pArg_->invariant_length * sizeof(MeanVarDataType);
            space_sz = math::integer_least_multiple(space_sz, 64);

            // setup buffer for welford result inv_variance
            pArg_->workspace_savedInvVar =
                reinterpret_cast<char*>(pArg_->workspace_savedMean) + space_sz;
        };
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0;

            const auto mean_var_count_grid_desc_m_g =
                DeviceBatchNormBwdImpl::MakeMultiblockFirstReduceOutputMG2dDescriptor(
                    arg.invariant_length, arg.blkGroupSize);

            const auto dscale_dbias_grid_desc_m_g =
                DeviceBatchNormBwdImpl::MakeMultiblockFirstReduceOutputMG2dDescriptor(
                    arg.invariant_length, arg.blkGroupSize);

            const auto mean_var_count_grid_desc_m_k =
                DeviceBatchNormBwdImpl::MakeMultiblockFinalReduceInputMK2dDescriptor(
                    arg.invariant_length, arg.blkGroupSize);

            const auto dscale_dbias_grid_desc_m_k =
                DeviceBatchNormBwdImpl::MakeMultiblockFinalReduceInputMK2dDescriptor(
                    arg.invariant_length, arg.blkGroupSize);

            using MeanVarCountGridDesc_M_G = decltype(mean_var_count_grid_desc_m_g);
            using MeanVarCountGridDesc_M_K = decltype(mean_var_count_grid_desc_m_k);
            using DscaleDbiasGridDesc_M_G  = decltype(dscale_dbias_grid_desc_m_g);
            using DscaleDbiasGridDesc_M_K  = decltype(dscale_dbias_grid_desc_m_k);

            using GridwiseWelfordSecondHalfReduceFirstHalf_ =
                GridwiseWelfordSecondHalfReduceFirstHalf<XDataType,
                                                         DyDataType,
                                                         AccDataType,
                                                         ScaleDataType,
                                                         DscaleDbiasDataType,
                                                         MeanVarDataType,
                                                         DyElementwiseOp,
                                                         XYGridDesc_M_K,
                                                         MeanVarGridDesc_M,
                                                         MeanVarCountGridDesc_M_K,
                                                         DscaleDbiasGridDesc_M_G,
                                                         BlockSize,
                                                         MThreadClusterSize,
                                                         KThreadClusterSize,
                                                         MThreadSliceSize,
                                                         KThreadSliceSize,
                                                         XDyDxVectorDim,
                                                         XSrcVectorSize,
                                                         DySrcVectorSize,
                                                         MeanVarSrcVectorSize>;

            using GridwiseReduceSecondHalfBatchNormBwdFinal_ =
                GridwiseReduceSecondHalfBatchNormBackwardFinal<XDataType,
                                                               DyDataType,
                                                               DxDataType,
                                                               AccDataType,
                                                               ScaleDataType,
                                                               DscaleDbiasDataType,
                                                               MeanVarDataType,
                                                               DyElementwiseOp,
                                                               XYGridDesc_M_K,
                                                               DscaleDbiasGridDesc_M_K,
                                                               MeanVarGridDesc_M,
                                                               ScaleBiasGridDesc_M,
                                                               BlockSize,
                                                               MThreadClusterSize,
                                                               KThreadClusterSize,
                                                               MThreadSliceSize,
                                                               KThreadSliceSize,
                                                               XDyDxVectorDim,
                                                               XSrcVectorSize,
                                                               DySrcVectorSize,
                                                               DxDstVectorSize,
                                                               ScaleSrcVectorSize,
                                                               DscaleDbiasDstVectorSize,
                                                               MeanVarSrcVectorSize>;

            if(UseMultiblockInK && arg.blkGroupSize > 1)
            {
                using GetReduceCountPerThreadFunctor =
                    GetReduceCountPerThreadForMultiblockWelford<K_BlockTileSize, KThreadSliceSize>;

                GetReduceCountPerThreadFunctor get_reduce_count_per_thread(
                    arg.blkGroupSize, arg.numBlockTileIteration, arg.reduce_length);

                if(!arg.haveSavedMeanInvVar_)
                {
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
                                                           XDyDxVectorDim,
                                                           XSrcVectorSize>;

                    const auto kern_multiblock_welford_first_half =
                        kernel_multiblock_welford_first_half<GridwiseMultiblockWelfordFirstHalf_,
                                                             XDataType,
                                                             MeanVarDataType,
                                                             XYGridDesc_M_K,
                                                             MeanVarCountGridDesc_M_G,
                                                             GetReduceCountPerThreadFunctor>;

                    avg_time += launch_and_time_kernel(
                        stream_config,
                        kern_multiblock_welford_first_half,
                        dim3(arg.gridSize),
                        dim3(BlockSize),
                        0,
                        arg.x_grid_desc_m_k,
                        mean_var_count_grid_desc_m_g,
                        get_reduce_count_per_thread,
                        arg.numBlockTileIteration,
                        arg.p_x_,
                        static_cast<MeanVarDataType*>(arg.workspace_mean),
                        static_cast<MeanVarDataType*>(arg.workspace_variance),
                        static_cast<int32_t*>(arg.workspace_count));
                };

                const auto kern_welford_second_half_reduce_first_half =
                    kernel_welford_second_half_reduce_first_half<
                        GridwiseWelfordSecondHalfReduceFirstHalf_,
                        XDataType,
                        DyDataType,
                        AccDataType,
                        ScaleDataType,
                        DscaleDbiasDataType,
                        MeanVarDataType,
                        DyElementwiseOp,
                        XYGridDesc_M_K,
                        MeanVarGridDesc_M,
                        MeanVarCountGridDesc_M_K,
                        DscaleDbiasGridDesc_M_G>;

                const auto kern_reduce_second_half_batchnorm_backward_final =
                    kernel_reduce_second_half_batchnorm_backward_final<
                        GridwiseReduceSecondHalfBatchNormBwdFinal_,
                        XDataType,
                        DyDataType,
                        DxDataType,
                        ScaleDataType,
                        DscaleDbiasDataType,
                        MeanVarDataType,
                        DyElementwiseOp,
                        XYGridDesc_M_K,
                        DscaleDbiasGridDesc_M_K,
                        MeanVarGridDesc_M,
                        ScaleBiasGridDesc_M>;

                index_t numDscaleDbiasBlockTileIteration =
                    (arg.blkGroupSize + KThreadClusterSize - 1) / KThreadClusterSize;

                avg_time += launch_and_time_kernel(
                    stream_config,
                    kern_welford_second_half_reduce_first_half,
                    dim3(arg.gridSize),
                    dim3(BlockSize),
                    0,
                    arg.x_grid_desc_m_k,
                    arg.dy_grid_desc_m_k,
                    arg.mean_var_grid_desc_m,
                    mean_var_count_grid_desc_m_k,
                    dscale_dbias_grid_desc_m_g,
                    arg.blkGroupSize,
                    arg.numBlockTileIteration,
                    numDscaleDbiasBlockTileIteration,
                    arg.epsilon_,
                    arg.haveSavedMeanInvVar_,
                    arg.haveSavedMeanInvVar_ ? arg.p_savedMean_ : nullptr,
                    arg.haveSavedMeanInvVar_ ? arg.p_savedInvVar_ : nullptr,
                    arg.haveSavedMeanInvVar_
                        ? nullptr
                        : static_cast<const MeanVarDataType*>(arg.workspace_mean),
                    arg.haveSavedMeanInvVar_
                        ? nullptr
                        : static_cast<const MeanVarDataType*>(arg.workspace_variance),
                    arg.haveSavedMeanInvVar_ ? nullptr
                                             : static_cast<const int32_t*>(arg.workspace_count),
                    arg.dy_elementwise_op_,
                    arg.haveSavedMeanInvVar_
                        ? nullptr
                        : static_cast<MeanVarDataType*>(arg.workspace_savedMean),
                    arg.haveSavedMeanInvVar_
                        ? nullptr
                        : static_cast<MeanVarDataType*>(arg.workspace_savedInvVar),
                    arg.p_x_,
                    arg.p_dy_,
                    static_cast<DscaleDbiasDataType*>(arg.workspace_reduce_dscale),
                    static_cast<DscaleDbiasDataType*>(arg.workspace_reduce_dbias));

                avg_time += launch_and_time_kernel(
                    stream_config,
                    kern_reduce_second_half_batchnorm_backward_final,
                    dim3(arg.gridSize),
                    dim3(BlockSize),
                    0,
                    arg.x_grid_desc_m_k,
                    arg.dy_grid_desc_m_k,
                    arg.dx_grid_desc_m_k,
                    dscale_dbias_grid_desc_m_k,
                    arg.mean_var_grid_desc_m,
                    arg.scale_grid_desc_m,
                    arg.dscale_dbias_grid_desc_m,
                    arg.blkGroupSize,
                    arg.reduce_length,
                    arg.numBlockTileIteration,
                    numDscaleDbiasBlockTileIteration,
                    static_cast<const DscaleDbiasDataType*>(arg.workspace_reduce_dscale),
                    static_cast<const DscaleDbiasDataType*>(arg.workspace_reduce_dbias),
                    arg.haveSavedMeanInvVar_
                        ? arg.p_savedMean_
                        : static_cast<const MeanVarDataType*>(arg.workspace_savedMean),
                    arg.haveSavedMeanInvVar_
                        ? arg.p_savedInvVar_
                        : static_cast<const MeanVarDataType*>(arg.workspace_savedInvVar),
                    arg.p_x_,
                    arg.p_dy_,
                    arg.p_scale_,
                    arg.dy_elementwise_op_,
                    arg.p_dx_,
                    arg.p_dscale_,
                    arg.p_dbias_);
            }
            else
            {
                using GetReduceCountPerThreadFunctor =
                    GetReduceCountPerThreadForBlockwiseWelford<K_BlockTileSize, KThreadSliceSize>;

                GetReduceCountPerThreadFunctor get_reduce_count_per_thread(
                    arg.numBlockTileIteration, arg.reduce_length);

                using GridwiseBatchNormBackwardWithBlockwiseWelford_ =
                    GridwiseBatchNormBackwardWithBlockwiseWelford<XDataType,
                                                                  DyDataType,
                                                                  DxDataType,
                                                                  AccDataType,
                                                                  ScaleDataType,
                                                                  DscaleDbiasDataType,
                                                                  MeanVarDataType,
                                                                  DyElementwiseOp,
                                                                  XYGridDesc_M_K,
                                                                  ScaleBiasGridDesc_M,
                                                                  MeanVarGridDesc_M,
                                                                  GetReduceCountPerThreadFunctor,
                                                                  BlockSize,
                                                                  MThreadClusterSize,
                                                                  KThreadClusterSize,
                                                                  MThreadSliceSize,
                                                                  KThreadSliceSize,
                                                                  XDyDxVectorDim,
                                                                  XSrcVectorSize,
                                                                  DySrcVectorSize,
                                                                  DxDstVectorSize,
                                                                  ScaleSrcVectorSize,
                                                                  DscaleDbiasDstVectorSize,
                                                                  MeanVarSrcVectorSize>;

                const auto kern_batchnorm_bwd = kernel_batchnorm_backward_with_blockwise_welford<
                    GridwiseBatchNormBackwardWithBlockwiseWelford_,
                    XDataType,
                    DyDataType,
                    DxDataType,
                    AccDataType,
                    ScaleDataType,
                    DscaleDbiasDataType,
                    MeanVarDataType,
                    DyElementwiseOp,
                    XYGridDesc_M_K,
                    ScaleBiasGridDesc_M,
                    MeanVarGridDesc_M,
                    GetReduceCountPerThreadFunctor>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kern_batchnorm_bwd,
                                                   dim3(arg.gridSize),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.x_grid_desc_m_k,
                                                   arg.dy_grid_desc_m_k,
                                                   arg.dx_grid_desc_m_k,
                                                   arg.scale_grid_desc_m,
                                                   arg.dscale_dbias_grid_desc_m,
                                                   arg.mean_var_grid_desc_m,
                                                   get_reduce_count_per_thread,
                                                   arg.reduce_length,
                                                   arg.numBlockTileIteration,
                                                   arg.epsilon_,
                                                   arg.p_x_,
                                                   arg.p_dy_,
                                                   arg.p_scale_,
                                                   arg.haveSavedMeanInvVar_,
                                                   arg.p_savedMean_,
                                                   arg.p_savedInvVar_,
                                                   arg.dy_elementwise_op_,
                                                   arg.p_dx_,
                                                   arg.p_dscale_,
                                                   arg.p_dbias_);
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

        if constexpr(XDyDxVectorDim == 0)
        {
            if(pArg_->xStrides_[NumInvariantDim - 1] != 1 ||
               pArg_->dyStrides_[NumInvariantDim - 1] != 1 ||
               pArg_->dxStrides_[NumInvariantDim - 1] != 1)
                return false;

            if(pArg_->xyLengths_[NumInvariantDim - 1] % XSrcVectorSize != 0 ||
               pArg_->xyLengths_[NumInvariantDim - 1] % DySrcVectorSize != 0 ||
               pArg_->xyLengths_[NumInvariantDim - 1] % DxDstVectorSize != 0)
                return false;
        }
        else
        {
            if(pArg_->xStrides_[Rank - 1] != 1 || pArg_->dyStrides_[Rank - 1] != 1 ||
               pArg_->dxStrides_[Rank - 1] != 1)
                return false;

            if(pArg_->xyLengths_[Rank - 1] % XSrcVectorSize != 0 ||
               pArg_->xyLengths_[Rank - 1] % DySrcVectorSize != 0 ||
               pArg_->xyLengths_[Rank - 1] % DxDstVectorSize != 0)
                return false;
        };

        if(pArg_->bnScaleStrides_[NumInvariantDim - 1] != 1 && ScaleSrcVectorSize != 1)
            return false;

        if(pArg_->bnDscaleDbiasStrides_[NumInvariantDim - 1] != 1 && DscaleDbiasDstVectorSize != 1)
            return false;

        if(pArg_->bnScaleBiasMeanVarLengths_[NumInvariantDim - 1] % ScaleSrcVectorSize != 0)
            return false;

        if(pArg_->bnScaleBiasMeanVarLengths_[NumInvariantDim - 1] % DscaleDbiasDstVectorSize != 0)
            return false;

        if(pArg_->haveSavedMeanInvVar_)
        {
            if(pArg_->bnMeanVarStrides_[NumInvariantDim - 1] != 1 && MeanVarSrcVectorSize != 1)
                return false;

            if(pArg_->bnScaleBiasMeanVarLengths_[NumInvariantDim - 1] % MeanVarSrcVectorSize != 0)
                return false;
        };

        bool is_valid = true;

        static_for<0, NumInvariantDim, 1>{}([&](auto I) {
            if(pArg_->xyLengths_[I] != pArg_->bnScaleBiasMeanVarLengths_[I])
                is_valid = false;
        });

        if(!is_valid)
            return false;

        return true;
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, Rank> xyLengths,
                        const std::array<index_t, Rank> xStrides,
                        const std::array<index_t, Rank> dyStrides,
                        const std::array<index_t, Rank> dxStrides,
                        const std::array<int, NumBatchNormReduceDim> reduceDims,
                        const std::array<ck::index_t, NumInvariantDim> bnScaleBiasMeanVarLengths,
                        const std::array<ck::index_t, NumInvariantDim> bnScaleStrides,
                        const std::array<ck::index_t, NumInvariantDim> bnDscaleDbiasStrides,
                        const std::array<ck::index_t, NumInvariantDim> bnMeanVarStrides,
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
                                          dyStrides,
                                          dxStrides,
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
                                          dy_elementwise_op,
                                          epsilon,
                                          static_cast<DxDataType*>(p_dx),
                                          static_cast<DscaleDbiasDataType*>(p_dscale),
                                          static_cast<DscaleDbiasDataType*>(p_dbias));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchNormBwdImpl<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str << "XDyDxVectorDim_" << XDyDxVectorDim  << ",";
        str << "VectorSize_X" << XSrcVectorSize << "_scale_" << ScaleSrcVectorSize << "_bias_" << DscaleDbiasDstVectorSize << "_mean_var_" << MeanVarSrcVectorSize << "_Dx_" << DxDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
