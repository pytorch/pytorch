// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/sequence.hpp"
#include "ck/utility/reduction_operator.hpp"

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_multiple_reduce.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_2d_multiple_reduction_multiblock.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_set_multiple_buffer_value.hpp"

#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumReduction,
          typename InDataType,
          typename AccDataType,
          typename OutDataTypeTuple,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOperationTuple,
          typename AccElementwiseOperationTuple,
          InMemoryDataOperationEnum OutMemoryDataOperation,
          bool PropagateNan,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          typename OutDstVectorSizeSeq>
struct DeviceMultipleReduceMultiBlock : public DeviceMultipleReduce<Rank,
                                                                    NumReduceDim,
                                                                    NumReduction,
                                                                    InElementwiseOperationTuple,
                                                                    AccElementwiseOperationTuple>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                      (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert(NumReduction == OutDataTypeTuple::Size() &&
                      NumReduction == InElementwiseOperationTuple::Size() &&
                      NumReduction == AccElementwiseOperationTuple::Size() &&
                      NumReduction == OutDstVectorSizeSeq::Size(),
                  "All tuple should have the same size as the number of Reductions!");

    static_assert(sequence_all_of(OutDstVectorSizeSeq{},
                                  [](auto vectorSize) {
                                      return (MThreadSliceSize % vectorSize == 0);
                                  }),
                  "The OutDstVectorSize should completely divide the MThreadSliceSize!");

    static constexpr bool CheckDataTypeTuple()
    {
        bool flag = true;

        static_for<0, NumReduction, 1>{}([&](auto I) {
            using OutDataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;
            flag =
                flag && ck::reduce::InMemoryDataOperatonSupportedOnDataType<OutMemoryDataOperation,
                                                                            OutDataType>::value;
        });

        return flag;
    };

    static_assert(CheckDataTypeTuple(),
                  "The OutDataType must support the specified OutMemoryDataOperation!");

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;

    static constexpr index_t NumInputDim  = Rank;
    static constexpr index_t NumOutputDim = (NumInvariantDim == 0) ? 1 : NumInvariantDim;
    static constexpr bool reduceAllDim    = (NumInvariantDim == 0);

    // So far, only AtomicAdd is considered, other Atomic Operation like AtomicMax can be added
    // later
    static constexpr bool use_multiblock =
        (OutMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd);

    static_assert(
        ReduceOperation::IsCompatibleInMemoryDataOperation(OutMemoryDataOperation),
        "The reduction accumulation operation must be compatible with the OutMemoryDataOperation!");

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto GenerateOutDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;

                return static_cast<DataType*>(nullptr);
            },
            Number<NumReduction>{});
    };

    using OutDataTypePointerTuple = decltype(GenerateOutDataTypePointerTuple());

    static auto MakeSrc2dDescriptor(const std::array<index_t, NumInputDim>& inLengths,
                                    const std::array<index_t, NumInputDim>& inStrides,
                                    int blkGroupSize,
                                    int numBlockTileIteration)
    {
        const auto tupleSrcLengths =
            generate_tuple([&](auto I) { return inLengths[I]; }, Number<NumInputDim>{});
        const auto tupleSrcStrides =
            generate_tuple([&](auto I) { return inStrides[I]; }, Number<NumInputDim>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
            if constexpr(reduceAllDim)
            {
                const auto one_dim_inDesc = transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, NumInputDim, 1>::type{}),
                    make_tuple(Sequence<0>{}));

                return transform_tensor_descriptor(one_dim_inDesc,
                                                   make_tuple(make_unmerge_transform(make_tuple(
                                                       1, one_dim_inDesc.GetLength(Number<0>{})))),
                                                   make_tuple(Sequence<0>{}),
                                                   make_tuple(Sequence<0, 1>{}));
            }
            else
            {
                using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
                using ReduceDims = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

                const auto reduceDimLengths = generate_tuple(
                    [&](auto I) { return inLengths[NumInvariantDim + I]; }, Number<NumReduceDim>{});
                const auto invariantDimLengths =
                    generate_tuple([&](auto I) { return inLengths[I]; }, Number<NumInvariantDim>{});

                return transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(invariantDimLengths),
                               make_merge_transform(reduceDimLengths)),
                    make_tuple(InvariantDims{}, ReduceDims{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto invariantLength = in_grid_desc_m_k.GetLength(Number<0>{});
        const auto reduceLength    = in_grid_desc_m_k.GetLength(Number<1>{});

        const int reduceSizePerBlock = K_BlockTileSize * numBlockTileIteration;
        const auto inPad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto inPad_K = reduceSizePerBlock * blkGroupSize - reduceLength;

        auto in_grid_desc_m_k_padded = transform_tensor_descriptor(
            in_grid_desc_m_k,
            make_tuple(make_right_pad_transform(invariantLength, inPad_M),
                       make_right_pad_transform(reduceLength, inPad_K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_grid_desc_m_k_padded);
    };

    static auto MakeDst1dDescriptor(const std::array<index_t, NumOutputDim>& outLengths,
                                    const std::array<index_t, NumOutputDim>& outStrides)
    {
        const auto tupleDstLengths =
            generate_tuple([&](auto I) { return outLengths[I]; }, Number<NumOutputDim>{});
        const auto tupleDstStrides =
            generate_tuple([&](auto I) { return outStrides[I]; }, Number<NumOutputDim>{});

        auto outDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto out_grid_desc_m = transform_tensor_descriptor(
            outDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, NumOutputDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto invariantLength = out_grid_desc_m.GetLength(Number<0>{});

        const auto outPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto out_grid_desc_m_padded = transform_tensor_descriptor(
            out_grid_desc_m,
            make_tuple(make_right_pad_transform(invariantLength, outPad)),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0>{}));
        return (out_grid_desc_m_padded);
    };

    static auto GenerateOutGrid1dDescTuple()
    {
        return generate_tuple(
            [&](auto I) {
                (void)I;
                return MakeDst1dDescriptor(std::array<index_t, NumOutputDim>{},
                                           std::array<index_t, NumOutputDim>{});
            },
            Number<NumReduction>{});
    };

    using InGridDesc_M_K      = decltype(MakeSrc2dDescriptor(
        std::array<index_t, NumInputDim>{}, std::array<index_t, NumInputDim>{}, 1, 1));
    using OutGridDesc_M_Tuple = decltype(GenerateOutGrid1dDescTuple());

    static auto MakeDst1dDescriptorForBufferSet(const std::array<index_t, NumOutputDim>& outLengths,
                                                const std::array<index_t, NumOutputDim>& outStrides)
    {
        const auto tupleDstLengths =
            generate_tuple([&](auto I) { return outLengths[I]; }, Number<NumOutputDim>{});
        const auto tupleDstStrides =
            generate_tuple([&](auto I) { return outStrides[I]; }, Number<NumOutputDim>{});

        auto outDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto out_grid_desc_m = transform_tensor_descriptor(
            outDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, NumOutputDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto length = out_grid_desc_m.GetLength(Number<0>{});

        const auto pad = math::integer_least_multiple(length, BlockSize) - length;

        auto out_grid_desc_m_padded =
            transform_tensor_descriptor(out_grid_desc_m,
                                        make_tuple(make_right_pad_transform(length, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (out_grid_desc_m_padded);
    };

    static auto GenerateOutGrid1dDescTuple_2()
    {
        return generate_tuple(
            [&](auto I) {
                (void)I;
                return MakeDst1dDescriptorForBufferSet(std::array<index_t, NumOutputDim>{},
                                                       std::array<index_t, NumOutputDim>{});
            },
            Number<NumReduction>{});
    };

    using OutGridDesc_M_Tuple_2 = decltype(GenerateOutGrid1dDescTuple_2());

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, NumInputDim>& inLengths,
                 const std::array<index_t, NumInputDim>& inStrides,
                 const std::array<index_t, NumOutputDim>& outLengths,
                 const std::array<std::array<index_t, NumOutputDim>, NumReduction>& outStridesArray,
                 const std::array<int, NumReduceDim>& reduceDims,
                 const std::array<const void*, NumReduction>& alphas,
                 const std::array<const void*, NumReduction>& betas,
                 const void* in_dev,
                 const std::array<void*, NumReduction>& out_dev_buffers,
                 const InElementwiseOperationTuple in_elementwise_op_tuple,
                 const AccElementwiseOperationTuple acc_elementwise_op_tuple)
            : outLengths_{outLengths},
              outStridesArray_{outStridesArray},
              in_elementwise_op_tuple_{in_elementwise_op_tuple},
              acc_elementwise_op_tuple_{acc_elementwise_op_tuple}
        {
            inLengths_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inLengths, reduceDims);
            inStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inStrides, reduceDims);

            for(size_t i = 0; i < NumReduction; i++)
            {
                alpha_values_(i) = *static_cast<const AccDataType*>(alphas[i]);
                beta_values_(i)  = *static_cast<const AccDataType*>(betas[i]);
            };

            in_dev_ = static_cast<const InDataType*>(in_dev);

            out_dev_buffers_ = generate_tuple(
                [&](auto iR) {
                    using OutDataTypePointer =
                        remove_cvref_t<decltype(OutDataTypePointerTuple{}[iR])>;
                    using OutDataType = remove_cvref_t<remove_pointer_t<OutDataTypePointer>>;
                    return static_cast<OutDataType*>(out_dev_buffers[iR]);
                },
                Number<NumReduction>{});

            std::tie(invariant_total_length, reduce_total_length) =
                get_2d_lengths<Rank, NumReduceDim>(inLengths_);

            if constexpr(use_multiblock)
            {

                int iterations = 1;
                while(true)
                {
                    int testBlkGroupSize =
                        (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                        (K_BlockTileSize * iterations);

                    // we want the blkGroupSize be not more than 128
                    if(testBlkGroupSize <= 128)
                        break;

                    iterations++;
                };

                blkGroupSize = (reduce_total_length + (K_BlockTileSize * iterations) - 1) /
                               (K_BlockTileSize * iterations);

                numBlockTileIteration = iterations;
            }
            else
            {
                blkGroupSize = 1;
                numBlockTileIteration =
                    (reduce_total_length + K_BlockTileSize - 1) / K_BlockTileSize;
            };

            in_grid_desc_m_k =
                MakeSrc2dDescriptor(inLengths_, inStrides_, blkGroupSize, numBlockTileIteration);

            out_grid_desc_m_tuple = generate_tuple(
                [&](auto I) { return MakeDst1dDescriptor(outLengths, outStridesArray[I]); },
                Number<NumReduction>{});

            out_grid_desc_m_tuple_2 = generate_tuple(
                [&](auto I) {
                    return MakeDst1dDescriptorForBufferSet(outLengths, outStridesArray[I]);
                },
                Number<NumReduction>{});

            gridSize = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                       M_BlockTileSize * blkGroupSize;

            gridSize_pre =
                math::integer_least_multiple(invariant_total_length, BlockSize) / BlockSize;
        }

        std::array<index_t, NumInputDim> inLengths_;
        std::array<index_t, NumInputDim> inStrides_;

        std::array<index_t, NumOutputDim> outLengths_;
        std::array<std::array<index_t, NumOutputDim>, NumReduction> outStridesArray_;

        Array<AccDataType, NumReduction> alpha_values_;
        Array<AccDataType, NumReduction> beta_values_;

        const InDataType* in_dev_;
        OutDataTypePointerTuple out_dev_buffers_;

        InGridDesc_M_K in_grid_desc_m_k;
        OutGridDesc_M_Tuple out_grid_desc_m_tuple;
        OutGridDesc_M_Tuple_2 out_grid_desc_m_tuple_2;

        InElementwiseOperationTuple in_elementwise_op_tuple_;
        AccElementwiseOperationTuple acc_elementwise_op_tuple_;

        long_index_t invariant_total_length;
        long_index_t reduce_total_length;

        int blkGroupSize;
        int numBlockTileIteration;
        size_t gridSize;

        size_t gridSize_pre;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            using GridwiseMultipleReduce =
                GridwiseMultipleReduction_mk_to_m_multiblock<NumReduction,
                                                             InDataType,
                                                             OutDataTypePointerTuple,
                                                             AccDataType,
                                                             InGridDesc_M_K,
                                                             OutGridDesc_M_Tuple,
                                                             ReduceOperation,
                                                             InElementwiseOperationTuple,
                                                             AccElementwiseOperationTuple,
                                                             OutMemoryDataOperation,
                                                             PropagateNan,
                                                             BlockSize,
                                                             MThreadClusterSize,
                                                             KThreadClusterSize,
                                                             MThreadSliceSize,
                                                             KThreadSliceSize,
                                                             InSrcVectorDim,
                                                             InSrcVectorSize,
                                                             OutDstVectorSizeSeq>;

            const auto kernel_main =
                kernel_multiple_reduce_multiblock<GridwiseMultipleReduce,
                                                  NumReduction,
                                                  InDataType,
                                                  OutDataTypePointerTuple,
                                                  AccDataType,
                                                  InGridDesc_M_K,
                                                  OutGridDesc_M_Tuple,
                                                  InElementwiseOperationTuple,
                                                  AccElementwiseOperationTuple>;

            float avg_time = 0;

            if constexpr(use_multiblock)
            {
                auto identity_values = generate_tuple(
                    [&](auto iR) {
                        using OutDataType = remove_cvref_t<decltype(OutDataTypeTuple{}[iR])>;
                        return ck::reduce::GetIdentityValueForInMemoryDataOperation<OutDataType>(
                            OutMemoryDataOperation);
                    },
                    Number<NumReduction>{});

                const auto kernel_pre = kernel_multiple_buffer_set_value<OutGridDesc_M_Tuple_2,
                                                                         NumReduction,
                                                                         BlockSize,
                                                                         OutDataTypePointerTuple,
                                                                         OutDataTypeTuple>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kernel_pre,
                                                   dim3(arg.gridSize_pre),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.out_grid_desc_m_tuple_2,
                                                   arg.out_dev_buffers_,
                                                   identity_values);
            };

            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize),
                                               dim3(BlockSize),
                                               0,
                                               arg.in_grid_desc_m_k,
                                               arg.out_grid_desc_m_tuple,
                                               arg.in_elementwise_op_tuple_,
                                               arg.acc_elementwise_op_tuple_,
                                               arg.blkGroupSize,
                                               arg.numBlockTileIteration,
                                               arg.alpha_values_,
                                               arg.in_dev_,
                                               arg.beta_values_,
                                               arg.out_dev_buffers_);

            return (avg_time);
        };

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if constexpr(use_multiblock)
        {
            for(size_t i = 0; i < pArg->beta_values_.Size(); i++)
                if(pArg->beta_values_[i] != 0.0f)
                    return (false);
        };

        if constexpr(InSrcVectorDim == 0)
        {
            if constexpr(NumInvariantDim == 0)
            {
                return (false);
            }
            else
            {
                if(pArg->inStrides_[NumInvariantDim - 1] != 1 && InSrcVectorSize != 1)
                    return (false);

                if(pArg->inLengths_[NumInvariantDim - 1] % InSrcVectorSize != 0)
                    return (false);
            };
        }
        else
        {
            if(pArg->inStrides_[Rank - 1] != 1 && InSrcVectorSize != 1)
                return (false);

            if(pArg->inLengths_[Rank - 1] % InSrcVectorSize != 0)
                return (false);
        };
        // To improve
        bool valid = true;
        static_for<0, NumReduction, 1>{}([&](auto I) {
            if(pArg->outStridesArray_[I.value][NumOutputDim - 1] != 1 &&
               OutDstVectorSizeSeq::At(I) != 1)
                valid = false;

            if(pArg->outLengths_[NumOutputDim - 1] % OutDstVectorSizeSeq::At(I) != 0)
                valid = false;
        });

        if(!valid)
            return (false);

        if constexpr(use_multiblock)
        {
            // blkGroupSize of 1 should be handled by Blockwise path using
            // InMemoryDataOperationEnum::Set
            if(pArg->blkGroupSize == 1)
                return (false);

            // This is very strong restriction, but needed to avoid some failure
            if(pArg->outLengths_[NumOutputDim - 1] % M_BlockTileSize != 0)
                return (false);
        }
        else
        {
            // cases with very small reduce_total_length should be handled by ThreadWise kernel
            if(pArg->reduce_total_length / KThreadSliceSize < 2)
                return (false);
        };

        return (true);
    };

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const std::array<index_t, NumInputDim> inLengths,
        const std::array<index_t, NumInputDim> inStrides,
        const std::array<index_t, NumOutputDim> outLengths,
        const std::array<std::array<index_t, NumOutputDim>, NumReduction> outStridesArray,
        const std::array<int, NumReduceDim> reduceDims,
        const std::array<const void*, NumReduction> alphas,
        const std::array<const void*, NumReduction> betas,
        const void* in_dev,
        const std::array<void*, NumReduction> out_dev_buffers,
        const InElementwiseOperationTuple in_elementwise_op_tuple,
        const AccElementwiseOperationTuple acc_elementwise_op_tuple) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStridesArray,
                                          reduceDims,
                                          alphas,
                                          betas,
                                          in_dev,
                                          out_dev_buffers,
                                          in_elementwise_op_tuple,
                                          acc_elementwise_op_tuple);
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << (OutMemoryDataOperation == InMemoryDataOperationEnum::Set? "DeviceMultipleReduceBlockWise<" : "DeviceMultipleReduceMultiBlock<") << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str << "InSrcVectorDim_" << InSrcVectorDim << "_InSrcVectorSize_" << InSrcVectorSize << ",";
        str << "OutDstVectorSize"; 
        static_for<0, OutDstVectorSizeSeq::Size(), 1>{}([&](auto I) {str << "_" << OutDstVectorSizeSeq::At(I); }); 
        str << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
