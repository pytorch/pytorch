// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/reduction_operator.hpp"

#include "ck/tensor_operation/gpu/device/device_elementwise_normalization.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_layernorm_welford_variance.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_set_buffer_value.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

// X = Elementwise(input1, input2, input3, ...)
// Y = Normalization(X, beta, gamma)
namespace ck {
template <typename GridwiseElementwiseReduction,
          typename InDataTypePointerTuple, // Datatype tuple of inputs
          typename XDataType,              // Datatype of X
          typename GammaDataType,          // Datatype of Gamma
          typename BetaDataType,           // Datatype of Beta
          typename YDataType,              // Datatype of Y
          typename AccDataType,            // AccDatatype
          typename XElementwiseOperation,  // Operation of input
          typename YElementwiseOperation,  // Operation of output of normalization
          typename InGrid2dDescTuple,      // Descriptor tuple of inputs
          typename GridDesc_M_K>           // Descriptor of inputs, Gamma, Beta
__global__ void kernel_elementwise_layernorm(
    const InGrid2dDescTuple in_grid_2d_desc_tuple,          // Descriptor tuple of inputs
    const GridDesc_M_K x_grid_desc_m_k,                     // Descriptor of X
    const GridDesc_M_K gamma_grid_desc_m_k,                 // Descriptor of gamma
    const GridDesc_M_K beta_grid_desc_m_k,                  // Descriptor of beta
    const GridDesc_M_K y_grid_desc_m_k,                     // Descriptor of Y
    index_t num_k_block_tile_iteration,                     //
    AccDataType epsilon,                                    // Datatype of epsilon
    const InDataTypePointerTuple p_in_global_tuple,         // Ptr tuple of input matrixs
    const GammaDataType* const __restrict__ p_gamma_global, // Ptr of gamma
    const BetaDataType* const __restrict__ p_beta_global,   // Ptr of beta
    YDataType* const __restrict__ p_y_global,               // Ptr of y
    const XElementwiseOperation x_elementwise_op,           // Operation of input
    const YElementwiseOperation y_elementwise_op)           // Operation of output of normalization
{
    extern __shared__ XDataType p_x_lds[];
    GridwiseElementwiseReduction::Run(in_grid_2d_desc_tuple,      // Descriptor tuple of inputs
                                      x_grid_desc_m_k,            // Descriptor of X
                                      gamma_grid_desc_m_k,        // Descriptor of Gamma
                                      beta_grid_desc_m_k,         // Descriptor of Beta
                                      y_grid_desc_m_k,            // Descriptor of Y
                                      num_k_block_tile_iteration, //
                                      epsilon,                    // epsilon
                                      p_in_global_tuple,          // Ptr tuple of inputs
                                      p_x_lds,                    // Ptr of X
                                      p_gamma_global,             // Ptr of gamma
                                      p_beta_global,              // Ptr of beta
                                      p_y_global,                 // Ptr of Y
                                      x_elementwise_op,           // Operation of input
                                      y_elementwise_op); // Operation of output of normalization
};
} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// Y = LayerNorm(A + B, Beta, Gamma)
template <typename InDataTypeTuple,       // Datatype of inputs
          typename GammaDataType,         // Datatype of gamma
          typename BetaDataType,          // Datatype of beta
          typename AccDataType,           //
          typename YDataType,             //
          typename XElementwiseOperation, //
          typename YElementwiseOperation, //
          index_t Rank,                   //
          index_t NumReduceDim,           //
          index_t BlockSize,              //
          index_t MThreadClusterSize,     // Num of threads in a block on M direction
          index_t KThreadClusterSize,     // Num of threads in a block on N direction
          index_t MThreadSliceSize,       // Each thread calculate rows
          index_t KThreadSliceSize,       // Each thread calculate columns
          index_t XYSrcVectorDim,         // Dimension to do reduce
          index_t XSrcVectorSize,         // Size to fetch source x
          index_t GammaSrcVectorDim,      // Dimension for gamma to do reduce
          index_t GammaSrcVectorSize,     // Size to fetch source gamma
          index_t BetaSrcVectorDim,       // Dimension for beta to do reduce
          index_t BetaSrcVectorSize,      // Size to fetch source beta
          index_t YDstVectorSize>         // Size to write destination Y
struct DeviceElementwiseNormalizationImpl
    : public DeviceElementwiseNormalization<InDataTypeTuple,
                                            GammaDataType,
                                            BetaDataType,
                                            AccDataType,
                                            YDataType,
                                            XElementwiseOperation,
                                            YElementwiseOperation,
                                            Rank,
                                            NumReduceDim>
{
    static constexpr int NumInput = InDataTypeTuple::Size();

    using XDataType = YDataType;

    static_assert(
        (KThreadSliceSize % GammaSrcVectorSize == 0),
        "Invalid thread slice sizes and/or gamma vector sizes configuration, please check!");

    static_assert(
        (KThreadSliceSize % BetaSrcVectorSize == 0),
        "Invalid thread slice sizes and/or beta vector sizes configuration, please check!");

    static constexpr index_t M_BlockTileSize =
        MThreadClusterSize * MThreadSliceSize; // num of rows calculated in a block
    static constexpr index_t K_BlockTileSize =
        KThreadClusterSize * KThreadSliceSize; // num of columns calculated in a block

    static auto GenerateInDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;
                return static_cast<const DataType*>(nullptr);
            },
            Number<NumInput>{});
    };

    using InDataTypePointerTuple = decltype(GenerateInDataTypePointerTuple());

    static auto MakeSrc2dDescriptor(const std::vector<index_t>& inLengths,
                                    const std::vector<index_t>& inStrides,
                                    int blkGroupSize,
                                    int numBlockTileIteration)
    {
        constexpr index_t NumInvariantDim  = Rank - NumReduceDim;
        static constexpr index_t numSrcDim = Rank;
        static constexpr bool reduceAllDim = (NumInvariantDim == 0);

        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<numSrcDim>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<numSrcDim>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
            if constexpr(reduceAllDim)
            {
                const auto one_dim_inDesc = transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, numSrcDim, 1>::type{}),
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

                const auto reduceDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, ReduceDims{});
                const auto invariantDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, InvariantDims{});

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

    template <index_t TupleSize>
    static auto GenerateSrcGrid2dDescTuple(Number<TupleSize>)
    {
        return generate_tuple([&](auto) { return MakeSrc2dDescriptor({1}, {1}, 1, 1); },
                              Number<TupleSize>{});
    };

    using InGrid2dDescTuple = decltype(GenerateSrcGrid2dDescTuple(Number<NumInput>{}));

    using GridDesc_M_K = decltype(MakeSrc2dDescriptor({1}, {1}, 1, 1));

    using GridwiseReduceLayernormGeneric =
        GridwiseElementwiseLayernormWelfordVariance_mk_to_mk<InDataTypePointerTuple,
                                                             XDataType,
                                                             GammaDataType,
                                                             BetaDataType,
                                                             YDataType,
                                                             AccDataType,
                                                             XElementwiseOperation,
                                                             YElementwiseOperation,
                                                             InGrid2dDescTuple,
                                                             GridDesc_M_K,
                                                             BlockSize,
                                                             MThreadClusterSize,
                                                             KThreadClusterSize,
                                                             MThreadSliceSize,
                                                             KThreadSliceSize,
                                                             XYSrcVectorDim,
                                                             XSrcVectorSize,
                                                             GammaSrcVectorDim,
                                                             GammaSrcVectorSize,
                                                             BetaSrcVectorDim,
                                                             BetaSrcVectorSize,
                                                             XYSrcVectorDim,
                                                             YDstVectorSize,
                                                             false>;

    using GridwiseReduceLayernormSweepOnce =
        GridwiseElementwiseLayernormWelfordVariance_mk_to_mk<InDataTypePointerTuple,
                                                             XDataType,
                                                             GammaDataType,
                                                             BetaDataType,
                                                             YDataType,
                                                             AccDataType,
                                                             XElementwiseOperation,
                                                             YElementwiseOperation,
                                                             InGrid2dDescTuple,
                                                             GridDesc_M_K,
                                                             BlockSize,
                                                             MThreadClusterSize,
                                                             KThreadClusterSize,
                                                             MThreadSliceSize,
                                                             KThreadSliceSize,
                                                             XYSrcVectorDim,
                                                             XSrcVectorSize,
                                                             GammaSrcVectorDim,
                                                             GammaSrcVectorSize,
                                                             BetaSrcVectorDim,
                                                             BetaSrcVectorSize,
                                                             XYSrcVectorDim,
                                                             YDstVectorSize,
                                                             true>;

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t> lengths,
                 const std::array<std::vector<index_t>, NumInput> inStridesArray,
                 const std::vector<index_t> gammaStrides,
                 const std::vector<index_t> betaStrides,
                 const std::vector<index_t> yStrides,
                 const std::vector<index_t> reduceDims,
                 XElementwiseOperation x_elementwise_op,
                 YElementwiseOperation y_elementwise_op,
                 AccDataType epsilon,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 const GammaDataType* p_gamma,
                 const BetaDataType* p_beta,
                 YDataType* p_y)
            : epsilon_(epsilon),
              p_gamma_(p_gamma),
              p_beta_(p_beta),
              p_y_(p_y),
              x_elementwise_op_(x_elementwise_op),
              y_elementwise_op_(y_elementwise_op)
        {

            Lengths_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(lengths, reduceDims);
            for(int i = 0; i < NumInput; i++)
            {
                inStridesArray_[i] =
                    shuffle_tensor_dimensions<Rank, NumReduceDim>(inStridesArray[i], reduceDims);
            }

            yStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(yStrides, reduceDims);
            xStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(yStrides, reduceDims);

            gammaStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(gammaStrides, reduceDims);
            betaStrides_  = shuffle_tensor_dimensions<Rank, NumReduceDim>(betaStrides, reduceDims);

            in_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;
                    return static_cast<const DataType*>(in_dev_buffers[I.value]);
                },
                Number<NumInput>{});

            long_index_t invariant_total_length;
            long_index_t reduce_total_length;

            std::tie(invariant_total_length, reduce_total_length) =
                get_2d_lengths<Rank, NumReduceDim>(Lengths_);

            blkGroupSize_          = 1;
            numBlockTileIteration_ = (reduce_total_length + K_BlockTileSize - 1) / K_BlockTileSize;

            gridSize_ = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                        M_BlockTileSize * blkGroupSize_;

            in_grid_2d_desc_tuple_ = generate_tuple(
                [&](auto I) {
                    return MakeSrc2dDescriptor(
                        Lengths_, inStridesArray_[I.value], blkGroupSize_, numBlockTileIteration_);
                },
                Number<NumInput>{});

            x_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, xStrides_, blkGroupSize_, numBlockTileIteration_);

            gamma_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, gammaStrides_, blkGroupSize_, numBlockTileIteration_);

            beta_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, betaStrides_, blkGroupSize_, numBlockTileIteration_);

            y_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, yStrides_, blkGroupSize_, numBlockTileIteration_);

            sweep_once_ =
                x_grid_desc_m_k_.GetLength(Number<1>{}) <= KThreadClusterSize * KThreadSliceSize;

            if(!sweep_once_) // if not sweep once, compute memory size for matrix X in lds for
                             // store Intermediate results
            {
                int block_TileSize = M_BlockTileSize * reduce_total_length;
                x_lds_size_        = block_TileSize * sizeof(XDataType);
            }
            else
                x_lds_size_ = 0;
        }

        AccDataType epsilon_;

        InDataTypePointerTuple in_dev_buffers_;
        const GammaDataType* p_gamma_;
        const BetaDataType* p_beta_;
        YDataType* p_y_;

        std::vector<index_t> Lengths_;
        std::array<std::vector<index_t>, NumInput> inStridesArray_;
        std::vector<index_t> xStrides_;
        std::vector<index_t> gammaStrides_;
        std::vector<index_t> betaStrides_;
        std::vector<index_t> yStrides_;

        XElementwiseOperation x_elementwise_op_;
        YElementwiseOperation y_elementwise_op_;

        int blkGroupSize_;
        int numBlockTileIteration_;
        size_t gridSize_;

        InGrid2dDescTuple in_grid_2d_desc_tuple_;
        GridDesc_M_K x_grid_desc_m_k_;
        GridDesc_M_K gamma_grid_desc_m_k_;
        GridDesc_M_K beta_grid_desc_m_k_;
        GridDesc_M_K y_grid_desc_m_k_;
        bool sweep_once_;
        int x_lds_size_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel_main =
                arg.sweep_once_ ? kernel_elementwise_layernorm<GridwiseReduceLayernormSweepOnce,
                                                               InDataTypePointerTuple,
                                                               XDataType,
                                                               GammaDataType,
                                                               BetaDataType,
                                                               YDataType,
                                                               AccDataType,
                                                               XElementwiseOperation,
                                                               YElementwiseOperation,
                                                               InGrid2dDescTuple,
                                                               GridDesc_M_K>
                                : kernel_elementwise_layernorm<GridwiseReduceLayernormGeneric,
                                                               InDataTypePointerTuple,
                                                               XDataType,
                                                               GammaDataType,
                                                               BetaDataType,
                                                               YDataType,
                                                               AccDataType,
                                                               XElementwiseOperation,
                                                               YElementwiseOperation,
                                                               InGrid2dDescTuple,
                                                               GridDesc_M_K>;

            float avg_time = 0;
            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize_),
                                               dim3(BlockSize),
                                               arg.x_lds_size_,
                                               arg.in_grid_2d_desc_tuple_,
                                               arg.x_grid_desc_m_k_,
                                               arg.gamma_grid_desc_m_k_,
                                               arg.beta_grid_desc_m_k_,
                                               arg.y_grid_desc_m_k_,
                                               arg.numBlockTileIteration_,
                                               arg.epsilon_,
                                               arg.in_dev_buffers_,
                                               arg.p_gamma_,
                                               arg.p_beta_,
                                               arg.p_y_,
                                               arg.x_elementwise_op_,
                                               arg.y_elementwise_op_);

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
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);

        constexpr index_t NumInvariantDim = Rank - NumReduceDim;

        if constexpr(XYSrcVectorDim == 0)
        {
            if constexpr(NumInvariantDim == 0)
            {
                return false;
            }
            else
            {
                for(int i = 0; i < NumInput; i++)
                {
                    if(p_arg_->inStridesArray_[i][NumInvariantDim - 1] != 1)
                        return false;
                }

                if(p_arg_->inStridesArray_[0][NumInvariantDim - 1] != 1 &&
                   p_arg_->inStridesArray_[1][NumInvariantDim - 1] != 1)
                    return false;

                if(p_arg_->invariant_lowest_length % XSrcVectorSize != 0)
                    return false;
            };
        }
        else
        {
            for(int i = 0; i < NumInput; i++)
            {
                if(p_arg_->inStridesArray_[i][Rank - 1] != 1)
                    return false;
            }

            if(p_arg_->Lengths_[Rank - 1] % XSrcVectorSize != 0)
                return false;
        };

        if(p_arg_->Lengths_[Rank - 1] % YDstVectorSize != 0)
        {
            return false;
        }

        auto IsScalarPerVectorValid = [](bool isLastDimensionCoalesced, int scalarPerVector) {
            bool ret = true;

            if(!isLastDimensionCoalesced)
                ret = scalarPerVector == 1;
            else
                ret = KThreadSliceSize % scalarPerVector == 0;

            return ret;
        };

        if(!IsScalarPerVectorValid(p_arg_->gammaStrides_.back() == 1, GammaSrcVectorSize))
            return false;

        if(!IsScalarPerVectorValid(p_arg_->betaStrides_.back() == 1, BetaSrcVectorSize))
            return false;

        // if fastest dim is not reduced
        if constexpr(XYSrcVectorDim == 0) //
        {
            if(p_arg_->gammaStrides_[NumInvariantDim - 1] != 1)
                return (false);

            if(p_arg_->Lengths_[Rank - 1] % GammaSrcVectorSize != 0)
                return (false);
        }
        else // if fastest dim is reduced
        {
            if(p_arg_->gammaStrides_[Rank - 1] != 1)
                return (false);

            if(p_arg_->Lengths_[Rank - 1] % GammaSrcVectorSize != 0)
                return (false);
        }

        // if fastest dim is not reduced
        if constexpr(XYSrcVectorDim == 0)
        {
            if(p_arg_->betaStrides_[NumInvariantDim - 1] != 1)
                return (false);

            if(p_arg_->invariant_lowest_length % BetaSrcVectorSize != 0)
                return (false);
        }
        else // if fastest dim is reduced
        {
            if(p_arg_->betaStrides_[Rank - 1] != 1)
                return (false);

            if(p_arg_->Lengths_[Rank - 1] % BetaSrcVectorSize != 0)
                return (false);
        }

        return true;
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> lengths,
                        const std::array<std::vector<index_t>, NumInput> inStridesArray,
                        const std::vector<index_t> gammaStrides,
                        const std::vector<index_t> betaStrides,
                        const std::vector<index_t> yStrides,
                        const std::vector<index_t> reduceDims,
                        AccDataType epsilon,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        const void* p_gamma,
                        const void* p_beta,
                        void* p_y,
                        XElementwiseOperation x_elementwise_op,
                        YElementwiseOperation y_elementwise_op) override
    {
        return std::make_unique<Argument>(lengths,
                                          inStridesArray,
                                          gammaStrides,
                                          betaStrides,
                                          yStrides,
                                          reduceDims,
                                          x_elementwise_op,
                                          y_elementwise_op,
                                          epsilon,
                                          in_dev_buffers,
                                          static_cast<const GammaDataType*>(p_gamma),
                                          static_cast<const BetaDataType*>(p_beta),
                                          static_cast<YDataType*>(p_y));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceElementwiseNormalizationImpl<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str << "XYSrcVectorDim_" << XYSrcVectorDim  << ",";
        str << "VectorSize_X" << XSrcVectorSize << "_Gamma" << GammaSrcVectorSize << "_Beta" << BetaSrcVectorSize << "_Y" << YDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
