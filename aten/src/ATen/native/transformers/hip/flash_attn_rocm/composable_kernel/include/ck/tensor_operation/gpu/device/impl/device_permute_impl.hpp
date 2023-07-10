// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_permute.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_permute.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Swap last 2 dimensions
// input shape: [d[0], d[1], d[2], ..., d[NumDim-3], d[NumDim-2], d[NumDim-1]]
//                                                                ^^^^^^^^^^^
// output shape: [d[0], d[1], d[2], ..., d[NumDim-3], d[NumDim-1], d[NumDim-2]]
//                                                    ^^^^^^^^^^^
template <index_t NumDim,
          typename InDataType,
          typename OutDataType,
          typename ElementwiseOperation,
          index_t BlockSize,
          index_t NPerBlock,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t InBlockLdsExtraW,
          typename InBlockTransferThreadClusterLengths,
          typename InBlockTransferThreadClusterArrangeOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector>
struct DevicePermuteImpl : DevicePermute<NumDim, InDataType, OutDataType, ElementwiseOperation>
{
    using BaseType = DevicePermute<NumDim, InDataType, OutDataType, ElementwiseOperation>;
    using typename BaseType::Lengths;
    using typename BaseType::Strides;

    static_assert(3 <= NumDim, "Only accept at least 3D dimension tensor");
    static_assert((NumDim - 2) <= SrcVectorDim && SrcVectorDim < NumDim);
    static_assert((NumDim - 2) <= DstVectorDim && DstVectorDim < NumDim);
    static_assert(SrcVectorDim != DstVectorDim);

    template <index_t N = NumDim>
    static auto ConvertArrayToTuple(const std::array<index_t, NumDim>& array)
    {
        static_assert(1 <= N && N <= NumDim);

        return generate_tuple([&](auto I) { return array[I]; }, Number<N>{});
    }

    static auto MakeDescriptor_N_H_W(const Lengths& lengths, const Strides& stride)
    {
        // create nd descriptor, shape: [d[0], d[1], d[2], ..., d[NumDim-3], d[NumDim-2],
        // d[NumDim-1]]
        const auto desc =
            make_naive_tensor_descriptor(ConvertArrayToTuple(lengths), ConvertArrayToTuple(stride));

        // merge nd to 3d descriptor, shape: [(d[0] * d[1] * d[2] * ... * d[NumDim-3]), d[NumDim-2],
        // d[NumDim-1]]
        //                                   => [N, H, W]
        const index_t H       = *std::next(rbegin(lengths));
        const index_t W       = *rbegin(lengths);
        const auto desc_n_h_w = transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform(ConvertArrayToTuple<NumDim - 2>(lengths)),
                       make_pass_through_transform(H),
                       make_pass_through_transform(W)),
            make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<NumDim - 2>{}),
                       Sequence<NumDim - 2>{},
                       Sequence<NumDim - 1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        return PadTensorDescriptor(
            desc_n_h_w, make_tuple(NPerBlock, HPerBlock, WPerBlock), Sequence<true, true, true>{});
    }

    using InGridDesc  = decltype(MakeDescriptor_N_H_W({1, 1}, {1, 1}));
    using OutGridDesc = InGridDesc;

    using GridwisePermute = GridwisePermute<
        InGridDesc,
        OutGridDesc,
        InDataType,
        OutDataType,
        ElementwiseOperation,
        BlockSize,
        NPerBlock,
        HPerBlock,
        WPerBlock,
        InBlockLdsExtraW,
        InBlockTransferThreadClusterLengths,
        InBlockTransferThreadClusterArrangeOrder,
        SrcVectorDim - (NumDim - 3), // calculate new SrcVectorDim for the merged descriptor
        DstVectorDim - (NumDim - 3), // calculate new DstVectorDim for the merged descriptor
        SrcScalarPerVector,
        DstScalarPerVector>;

    using Block2TileMap = typename GridwisePermute::DefaultBlock2TileMap;

    struct Argument : public BaseArgument
    {
        Argument(const Lengths& in_lengths,
                 const Strides& in_strides,
                 const Lengths& out_lengths,
                 const Strides& out_strides,
                 const void* in_dev_buffer,
                 void* out_dev_buffer,
                 ElementwiseOperation elementwise_op)
            : in_dev_buffer_(static_cast<const InDataType*>(in_dev_buffer)),
              out_dev_buffer_(static_cast<OutDataType*>(out_dev_buffer)),
              in_grid_desc_(MakeDescriptor_N_H_W(in_lengths, in_strides)),
              out_grid_desc_(MakeDescriptor_N_H_W(out_lengths, out_strides)),
              in_lengths_(in_lengths),
              in_strides_(in_strides),
              out_lengths_(out_lengths),
              out_strides_(out_strides),
              elementwise_op_(elementwise_op),
              block_2_tile_map_(GridwisePermute::MakeDefaultBlock2TileMap(in_grid_desc_))
        {
        }

        const InDataType* in_dev_buffer_;
        OutDataType* out_dev_buffer_;
        InGridDesc in_grid_desc_;
        OutGridDesc out_grid_desc_;

        Lengths in_lengths_;
        Strides in_strides_;
        Lengths out_lengths_;
        Strides out_strides_;

        ElementwiseOperation elementwise_op_;

        Block2TileMap block_2_tile_map_;
    };

    struct Invoker : BaseInvoker
    {
        static float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const index_t grid_size = arg.block_2_tile_map_.CalculateGridSize(arg.in_grid_desc_);

            const auto kernel = kernel_nd_permute<GridwisePermute,
                                                  InGridDesc,
                                                  OutGridDesc,
                                                  InDataType,
                                                  OutDataType,
                                                  ElementwiseOperation,
                                                  Block2TileMap>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(grid_size),
                                                        dim3(BlockSize),
                                                        0,
                                                        arg.in_grid_desc_,
                                                        arg.out_grid_desc_,
                                                        arg.in_dev_buffer_,
                                                        arg.out_dev_buffer_,
                                                        arg.elementwise_op_,
                                                        arg.block_2_tile_map_);
            return elapsed_time;
        }

        float Run(const BaseArgument* arg,
                  const StreamConfig& stream_config = StreamConfig{}) override final
        {
            const auto* const argument = dynamic_cast<const Argument*>(arg);
            if(!argument)
            {
                return NAN;
            }

            return Run(*argument, stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        constexpr auto GetPaddedLength = [](index_t length, index_t tile_length) {
            return math::integer_divide_ceil(length, tile_length) * tile_length;
        };

        constexpr auto IsScalarPerVectorValid =
            [](index_t length, index_t stride, index_t scalar_per_vector) {
                if(stride == 1 && length % scalar_per_vector == 0)
                {
                    return true;
                }
                else if(stride != 1 && scalar_per_vector == 1)
                {
                    return true;
                }

                return false;
            };

        return IsScalarPerVectorValid(arg.in_lengths_[SrcVectorDim],
                                      arg.in_strides_[SrcVectorDim],
                                      SrcScalarPerVector) &&
               IsScalarPerVectorValid(
                   GetPaddedLength(arg.in_lengths_[SrcVectorDim],
                                   (SrcVectorDim == NumDim - 2 ? HPerBlock : WPerBlock)),
                   arg.in_strides_[SrcVectorDim],
                   SrcScalarPerVector) &&
               IsScalarPerVectorValid(arg.out_lengths_[DstVectorDim],
                                      arg.out_strides_[DstVectorDim],
                                      DstScalarPerVector) &&
               IsScalarPerVectorValid(
                   GetPaddedLength(arg.out_lengths_[DstVectorDim],
                                   (DstVectorDim == NumDim - 2 ? HPerBlock : WPerBlock)),
                   arg.in_strides_[DstVectorDim],
                   DstScalarPerVector) &&
               GridwisePermute::CheckValidity(arg.in_grid_desc_, arg.out_grid_desc_);
    };

    // override methods inherited from 'BaseOperator'
    bool IsSupportedArgument(const BaseArgument* arg) override final
    {
        const auto* const argument = dynamic_cast<const Argument*>(arg);
        if(!argument)
        {
            return false;
        }

        return IsSupportedArgument(*argument);
    }

    // override methods inherited from 'DevicePermute'
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const Lengths& in_lengths,
                        const Strides& in_strides,
                        const Lengths& out_lengths,
                        const Strides& out_strides,
                        const void* in_dev_buffer,
                        void* out_dev_buffer,
                        ElementwiseOperation elementwise_op) override final
    {
        return std::make_unique<Argument>(in_lengths,
                                          in_strides,
                                          out_lengths,
                                          out_strides,
                                          in_dev_buffer,
                                          out_dev_buffer,
                                          elementwise_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override final
    {
        return std::make_unique<Invoker>();
    };

    // other constructor methods
    template <typename... Args>
    static std::enable_if_t<std::is_constructible_v<Argument, Args...>, Argument>
    MakeArgument(Args&&... args) noexcept(std::is_nothrow_constructible_v<Argument, Args...>)
    {
        return Argument{std::forward<Args>(args)...};
    }

    static std::enable_if_t<std::is_default_constructible_v<Invoker>, Invoker>
    MakeInvoker() noexcept(std::is_nothrow_default_constructible_v<Invoker>)
    {
        return Invoker{};
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
