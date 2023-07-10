// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_softmax.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          typename InElementwiseOp,
          typename AccElementwiseOp,
          index_t Rank,
          index_t NumReduceDim,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceSoftmaxImpl : public DeviceSoftmax<InDataType,
                                                AccDataType,
                                                OutDataType,
                                                InElementwiseOp,
                                                AccElementwiseOp,
                                                Rank>
{
    static constexpr index_t kRank            = Rank;
    static constexpr index_t kNumReduceDim    = NumReduceDim;
    static constexpr index_t kNumInvariantDim = Rank - NumReduceDim;

    virtual index_t GetRank() const override { return kRank; }

    virtual index_t GetNumReduceDim() const override { return kNumReduceDim; }

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;

    static constexpr index_t NumSrcDim = Rank;
    static constexpr index_t NumDstDim = (NumInvariantDim == 0) ? 1 : NumInvariantDim;
    static constexpr bool reduceAllDim = (NumInvariantDim == 0);

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto MakeSrc2dDescriptor(const std::vector<index_t>& inLengths,
                                    const std::vector<index_t>& inStrides,
                                    int blkGroupSize,
                                    int numBlockTileIteration)
    {
        const auto tupleSrcLengths =
            generate_tuple([&](auto I) { return inLengths[I]; }, Number<Rank>{});
        const auto tupleSrcStrides =
            generate_tuple([&](auto I) { return inStrides[I]; }, Number<Rank>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
            if constexpr(reduceAllDim)
            {
                const auto one_dim_inDesc = transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, NumSrcDim, 1>::type{}),
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

    using GridDesc_M_K = decltype(MakeSrc2dDescriptor({1}, {1}, 1, 1));

    using GridwiseSoftmaxGeneric = GridwiseSoftmax_mk_to_mk<InDataType,
                                                            OutDataType,
                                                            AccDataType,
                                                            GridDesc_M_K,
                                                            BlockSize,
                                                            MThreadClusterSize,
                                                            KThreadClusterSize,
                                                            MThreadSliceSize,
                                                            KThreadSliceSize,
                                                            InSrcVectorDim,
                                                            InSrcVectorSize,
                                                            OutDstVectorSize,
                                                            false>;

    using GridwiseSoftmaxSweepOnce = GridwiseSoftmax_mk_to_mk<InDataType,
                                                              OutDataType,
                                                              AccDataType,
                                                              GridDesc_M_K,
                                                              BlockSize,
                                                              MThreadClusterSize,
                                                              KThreadClusterSize,
                                                              MThreadSliceSize,
                                                              KThreadSliceSize,
                                                              InSrcVectorDim,
                                                              InSrcVectorSize,
                                                              OutDstVectorSize,
                                                              true>;

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t> inLengths,
                 const std::vector<index_t> inStrides,
                 const std::vector<index_t> reduceDims,
                 AccDataType alpha,
                 AccDataType beta,
                 const InDataType* in_dev,
                 OutDataType* out_dev,
                 InElementwiseOp in_elementwise_op,
                 AccElementwiseOp acc_elementwise_op)
            : alpha_{alpha},
              beta_{beta},
              in_dev_{in_dev},
              out_dev_{out_dev},
              in_elementwise_op_{in_elementwise_op},
              acc_elementwise_op_{acc_elementwise_op}
        {
            if(Rank != inLengths.size() || Rank != inStrides.size() ||
               NumReduceDim != reduceDims.size())
            {
                throw std::runtime_error(
                    "One of inLengths/inStrides/reduceDims has invalid size!"
                    "\nExpected size inLengths: " +
                    std::to_string(Rank) + ", inStrides: " + std::to_string(Rank) +
                    ", reduceDims: " + std::to_string(NumReduceDim) +
                    "\nBut have inLengths: " + std::to_string(inLengths.size()) +
                    ", inStrides: " + std::to_string(inStrides.size()) +
                    ", reduceDims: " + std::to_string(reduceDims.size()));
            }

            for(std::size_t i = 0; i < reduceDims.size(); ++i)
            {
                if(reduceDims[i] < 0 || reduceDims[i] >= Rank)
                {
                    throw std::runtime_error("Provided reduce dimension exceed input tensor Rank!"
                                             "\nHave reduceDims[" +
                                             std::to_string(i) +
                                             "]: " + std::to_string(reduceDims[i]));
                }
            }

            inLengths_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inLengths, reduceDims);
            inStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inStrides, reduceDims);

            long_index_t invariant_total_length;
            long_index_t reduce_total_length;

            std::tie(invariant_total_length, reduce_total_length) =
                get_2d_lengths<Rank, NumReduceDim>(inLengths_);

            if constexpr(NumInvariantDim == 0)
                invariant_lowest_length_ = 1;
            else
                invariant_lowest_length_ = inLengths_[NumInvariantDim - 1];

            blkGroupSize          = 1;
            numBlockTileIteration = (reduce_total_length + K_BlockTileSize - 1) / K_BlockTileSize;

            gridSize = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                       M_BlockTileSize * blkGroupSize;
        }

        std::vector<index_t> inLengths_;
        std::vector<index_t> inStrides_;

        AccDataType alpha_;
        AccDataType beta_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;

        InElementwiseOp in_elementwise_op_;
        AccElementwiseOp acc_elementwise_op_;

        index_t invariant_lowest_length_;

        int blkGroupSize;
        int numBlockTileIteration;
        size_t gridSize;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto in_grid_desc_m_k = DeviceSoftmaxImpl::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);
            const auto out_grid_desc_m_k = DeviceSoftmaxImpl::MakeSrc2dDescriptor(
                arg.inLengths_, arg.inStrides_, arg.blkGroupSize, arg.numBlockTileIteration);

            bool sweep_once =
                in_grid_desc_m_k.GetLength(Number<1>{}) <= KThreadClusterSize * KThreadSliceSize;

            const auto kernel_main = sweep_once ? kernel_softmax<GridwiseSoftmaxSweepOnce,
                                                                 InDataType,
                                                                 OutDataType,
                                                                 AccDataType,
                                                                 GridDesc_M_K>
                                                : kernel_softmax<GridwiseSoftmaxGeneric,
                                                                 InDataType,
                                                                 OutDataType,
                                                                 AccDataType,
                                                                 GridDesc_M_K>;

            float avg_time = 0;

            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize),
                                               dim3(BlockSize),
                                               0,
                                               in_grid_desc_m_k,
                                               out_grid_desc_m_k,
                                               arg.blkGroupSize,
                                               arg.numBlockTileIteration,
                                               arg.alpha_,
                                               arg.in_dev_,
                                               arg.beta_,
                                               arg.out_dev_);

            return (avg_time);
        };

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if constexpr(InSrcVectorDim == 0)
        {
            if constexpr(kNumInvariantDim == 0)
            {
                return false;
            }
            else
            {
                if(arg.inStrides_[kNumInvariantDim - 1] != 1 && InSrcVectorSize != 1)
                {
                    return false;
                }
                if(arg.invariant_lowest_length_ % InSrcVectorSize != 0)
                {
                    return false;
                }
            }
        }
        else
        {
            if(arg.inStrides_[Rank - 1] != 1 && InSrcVectorSize != 1)
            {
                return false;
            }
            if(arg.inLengths_[Rank - 1] % InSrcVectorSize != 0)
            {
                return false;
            }
        }

        // To improve
        if(kNumInvariantDim > 0 && arg.invariant_lowest_length_ % OutDstVectorSize != 0)
        {
            return false;
        }

        if(arg.inLengths_[Rank - 1] % OutDstVectorSize != 0)
        {
            return false;
        }

        return true;
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const std::vector<index_t> inLengths,
                             const std::vector<index_t> inStrides,
                             const std::vector<int> reduceDims,
                             const AccDataType alpha,
                             const AccDataType beta,
                             const InDataType* in_dev,
                             OutDataType* out_dev,
                             InElementwiseOp in_elementwise_op,
                             AccElementwiseOp acc_elementwise_op)
    {
        return Argument{inLengths,
                        inStrides,
                        reduceDims,
                        alpha,
                        beta,
                        in_dev,
                        out_dev,
                        in_elementwise_op,
                        acc_elementwise_op};
    };

    //
    // @brief      Makes a pointer to Argument class.
    //
    // @param[in]  inLengths           Input tensor extent(s) from high to low dimension
    // @param[in]  inStrides           Input tensor stride(s) from high to low dimension
    // @param[in]  reduceDims          The dimension(s) the normalization operation is applied
    // @param[in]  alpha               Typeless pointer in host memory storing the alpha scaling
    //                                 value as type AccDataType
    // @param[in]  beta                Typeless pointer in host memory storing the beta scaling
    //                                 value as type AccDataType
    // @param[in]  in_dev              Typeless const pointer in device memory storing the input
    //                                 tensor
    // @param      out_dev             Typeless pointer in device memory storing the output tensor
    // @param[in]  in_elementwise_op   The input elementwise operation.
    // @param[in]  acc_elementwise_op  The accumulation elementwise operation.
    //
    // @return     Unique pointer to the Argument class.
    //
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<index_t> inLengths,
                                                      const std::vector<index_t> inStrides,
                                                      const std::vector<int> reduceDims,
                                                      const void* alpha,
                                                      const void* beta,
                                                      const void* in_dev,
                                                      void* out_dev,
                                                      InElementwiseOp in_elementwise_op,
                                                      AccElementwiseOp acc_elementwise_op) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          reduceDims,
                                          *static_cast<const AccDataType*>(alpha),
                                          *static_cast<const AccDataType*>(beta),
                                          static_cast<const InDataType*>(in_dev),
                                          static_cast<OutDataType*>(out_dev),
                                          in_elementwise_op,
                                          acc_elementwise_op);
    };

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceReduceSoftmax<" 
            << Rank << "," << NumReduceDim << "," << BlockSize << ","
            << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ","
            << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ","
            << "InSrcVectorDim_" << InSrcVectorDim 
            << "_InSrcVectorSize_" << InSrcVectorSize 
            << "_OutDstVectorSize_" << OutDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
