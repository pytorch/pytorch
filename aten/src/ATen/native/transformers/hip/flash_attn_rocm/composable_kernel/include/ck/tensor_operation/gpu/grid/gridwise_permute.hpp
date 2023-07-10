// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <numeric>
#include <iterator>

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwisePermute,
          typename InGridDesc,
          typename OutGridDesc,
          typename InDataType,
          typename OutDataType,
          typename ElementwiseOperation,
          typename Block2TileMap>
__global__ void kernel_nd_permute(const InGridDesc in_grid_desc,
                                  const OutGridDesc out_grid_desc,
                                  const InDataType* p_in_global,
                                  OutDataType* p_out_global,
                                  const ElementwiseOperation elementwise_op,
                                  const Block2TileMap block_2_tile_map)
{
    __shared__ char p_shared[GridwisePermute::GetSharedMemoryNumberOfByte()];

    GridwisePermute::Run(in_grid_desc,
                         out_grid_desc,
                         p_in_global,
                         p_out_global,
                         p_shared,
                         elementwise_op,
                         block_2_tile_map);
}

template <typename InGridDesc,
          typename OutGridDesc,
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
struct GridwisePermute
{
    static_assert(InGridDesc::GetNumOfDimension() == OutGridDesc::GetNumOfDimension());
    static_assert(3 <= InGridDesc::GetNumOfDimension());
    static_assert((InGridDesc::GetNumOfDimension() - 2) <= SrcVectorDim &&
                  SrcVectorDim < InGridDesc::GetNumOfDimension());
    static_assert((OutGridDesc::GetNumOfDimension() - 2) <= DstVectorDim &&
                  DstVectorDim < OutGridDesc::GetNumOfDimension());
    static_assert(SrcVectorDim != DstVectorDim);

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    struct Block2TileMap
    {
        static constexpr index_t NumDim = InGridDesc::GetNumOfDimension();
        static_assert(3 <= NumDim);

        static constexpr auto I0 = Number<0>{};

        Block2TileMap()                     = delete;
        Block2TileMap(const Block2TileMap&) = default;
        Block2TileMap(Block2TileMap&&)      = delete;

        ~Block2TileMap() = default;

        Block2TileMap& operator=(const Block2TileMap&) = delete;
        Block2TileMap& operator=(Block2TileMap&&) = delete;

        explicit Block2TileMap(const InGridDesc& desc) : desc_(desc) {}

        __host__ constexpr index_t CalculateGridSize(const InGridDesc& desc) const
        {
            const auto N0 =
                math::integer_divide_ceil(desc.GetLength(Number<NumDim - 3>{}), NPerBlock);
            const auto H0 =
                math::integer_divide_ceil(desc.GetLength(Number<NumDim - 2>{}), HPerBlock);
            const auto W0 =
                math::integer_divide_ceil(desc.GetLength(Number<NumDim - 1>{}), WPerBlock);

            const index_t grid_size = N0 * H0 * W0;

            return grid_size;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            static_assert(TopIdx::Size() == 1);

            auto block_1d_id = idx_top[I0];

            const auto N0 =
                math::integer_divide_ceil(desc_.GetLength(Number<NumDim - 3>{}), NPerBlock);
            const auto H0 =
                math::integer_divide_ceil(desc_.GetLength(Number<NumDim - 2>{}), HPerBlock);
            const auto W0 =
                math::integer_divide_ceil(desc_.GetLength(Number<NumDim - 1>{}), WPerBlock);

            block_1d_id = block_1d_id % (N0 * H0 * W0);

            index_t idx_N0 = block_1d_id / (H0 * W0);
            index_t idx_H0 = (block_1d_id % (H0 * W0)) / W0;
            index_t idx_W0 = block_1d_id % W0;

            return make_tuple(idx_N0, idx_H0, idx_W0);
        }

        private:
        const InGridDesc desc_;
    };

    using DefaultBlock2TileMap = Block2TileMap;

    // use an [NPerBlock, HPerBlock, WPerBlock] tensor as element-copy relay
    __host__ __device__ static constexpr auto GetInBlockDesc_NPerBlock_HPerBlock_WPerBlock()
    {
        return make_naive_tensor_descriptor(
            make_tuple(Number<NPerBlock>{}, Number<HPerBlock>{}, Number<WPerBlock>{}),
            make_tuple(Number<HPerBlock*(WPerBlock + InBlockLdsExtraW)>{},
                       Number<WPerBlock + InBlockLdsExtraW>{},
                       I1));
    }

    // for N-dimension descriptor, reserve its last 2 dimensions, then merge its leading dimensions
    // into single one. finally, form a 3D descriptor: [d(0), d(1), ..., d(N - 2), d(N - 1)] ->
    // [(d(0) x d(1) x ...), d(N - 2), d(N - 1)]
    template <typename GridDesc>
    __host__ __device__ static constexpr auto GetMergedDesc(const GridDesc& desc)
    {
        constexpr index_t NumDim = GridDesc::GetNumOfDimension();
        static_assert(3 <= NumDim);

        const auto merged_desc = transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform(generate_tuple(
                           [&](auto I) { return desc.GetLength(I); }, Number<NumDim - 2>{})),
                       make_pass_through_transform(desc.GetLength(Number<NumDim - 2>{})),
                       make_pass_through_transform(desc.GetLength(Number<NumDim - 1>{}))),
            make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<NumDim - 2>{}),
                       Sequence<NumDim - 2>{},
                       Sequence<NumDim - 1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        return merged_desc;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto in_block_desc_nperblock_hperblock_wperblock =
            GetInBlockDesc_NPerBlock_HPerBlock_WPerBlock();

        return in_block_desc_nperblock_hperblock_wperblock.GetElementSpaceSize() *
               sizeof(InDataType);
    }

    __host__ __device__ static constexpr auto MakeDefaultBlock2TileMap(const InGridDesc& desc)
    {
        return DefaultBlock2TileMap{desc};
    }

    __host__ __device__ static constexpr bool CheckValidity(const InGridDesc& in_grid_desc,
                                                            const OutGridDesc& out_grid_desc)
    {
        constexpr index_t NumDim = InGridDesc::GetNumOfDimension();

        // check if we only swap last 2 dimensions
        bool valid = true;
        static_for<0, NumDim - 2, 1>{}([&](auto I) {
            if(valid && in_grid_desc.GetLength(I) != out_grid_desc.GetLength(I))
            {
                valid = false;
            }
        });

        return valid &&
               (in_grid_desc.GetLength(Number<NumDim - 1>{}) ==
                out_grid_desc.GetLength(Number<NumDim - 2>{})) &&
               (in_grid_desc.GetLength(Number<NumDim - 2>{}) ==
                out_grid_desc.GetLength(Number<NumDim - 1>{}));
    }

    template <typename Block2TileMap>
    __device__ static void Run(const InGridDesc in_grid_desc,
                               const OutGridDesc out_grid_desc,
                               const InDataType* p_in_global,
                               OutDataType* p_out_global,
                               void* __restrict__ p_shared,
                               const ElementwiseOperation elementwise_op,
                               const Block2TileMap& block_2_tile_map)
    {
        auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_global, in_grid_desc.GetElementSpaceSize());

        auto out_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_global, out_grid_desc.GetElementSpaceSize());

        // each workgroup handles an [NPerBlock, HPerBlock, WPerBLock] slice-transpose problem
        const auto block_work_idx =
            block_2_tile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * NPerBlock);

        const index_t h_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * HPerBlock);

        const index_t w_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * WPerBlock);

        // create [NPerBlock, HPerBlock, WPerBLock] shaped LDS buffer
        constexpr auto in_block_desc_nperblock_hperblock_wperblock =
            GetInBlockDesc_NPerBlock_HPerBlock_WPerBlock();

        auto in_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<InDataType*>(p_shared),
            in_block_desc_nperblock_hperblock_wperblock.GetElementSpaceSize());

        using BlockSliceLengths          = Sequence<NPerBlock, HPerBlock, WPerBlock>;
        using InBlockTransferAccessOrder = Sequence<0, 1, 2>;

        constexpr index_t SrcVectorDimAfterMerge =
            SrcVectorDim - (InGridDesc::GetNumOfDimension() - 3);
        constexpr index_t DstVectorDimAfterMerge = SrcVectorDimAfterMerge;

        using ck::tensor_operation::element_wise::PassThrough;

        // merge input descriptor into [(in_grid_desc.GetLength(0) x in_grid_desc.GetLength(1) x
        // ...), in_grid_desc.GetLength(NumDim - 2), in_grid_desc.GetLength(NumDim - 1)]
        const auto in_grid_desc_n_h_w = GetMergedDesc(in_grid_desc);

        // a workgroup copies an [NPerBlock, HPerBlock, WPerBlock] slice from global memory to LDS
        auto in_global_load = ThreadGroupTensorSliceTransfer_v4r1<
            ThisThreadBlock,
            ElementwiseOperation,
            PassThrough,
            InMemoryDataOperationEnum::Set,
            BlockSliceLengths,
            InBlockTransferThreadClusterLengths,
            InBlockTransferThreadClusterArrangeOrder,
            InDataType,
            InDataType,
            decltype(in_grid_desc_n_h_w),
            decltype(in_block_desc_nperblock_hperblock_wperblock),
            InBlockTransferAccessOrder,
            InBlockTransferAccessOrder,
            SrcVectorDimAfterMerge,
            2,
            SrcScalarPerVector,
            1,
            1,
            1,
            true,
            true>(in_grid_desc_n_h_w,
                  make_multi_index(
                      n_block_data_idx_on_grid, h_block_data_idx_on_grid, w_block_data_idx_on_grid),
                  PassThrough{},
                  in_block_desc_nperblock_hperblock_wperblock,
                  make_multi_index(0, 0, 0),
                  PassThrough{});

        // merge output descriptor into [(out_grid_desc.GetLength(0) x out_grid_desc.GetLength(1) x
        // ...), out_grid_desc.GetLength(NumDim - 2), out_grid_desc.GetLength(NumDim - 1)]
        const auto out_grid_desc_n_w_h = GetMergedDesc(out_grid_desc);

        // create transposed view of output tensor
        const auto out_grid_desc_n_h_w = transform_tensor_descriptor(
            out_grid_desc_n_w_h,
            make_tuple(make_pass_through_transform(out_grid_desc_n_w_h.GetLength(I0)),
                       make_pass_through_transform(out_grid_desc_n_w_h.GetLength(I1)),
                       make_pass_through_transform(out_grid_desc_n_w_h.GetLength(I2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1>{}));

        // a workgroup copies an [NPerBlock, HPerBlock, WPerBlock] slice from LDS to global memory
        auto out_global_store = ThreadGroupTensorSliceTransfer_v4r1<
            ThisThreadBlock,
            ElementwiseOperation,
            PassThrough,
            InMemoryDataOperationEnum::Set,
            BlockSliceLengths,
            InBlockTransferThreadClusterLengths,
            InBlockTransferThreadClusterArrangeOrder,
            InDataType,
            OutDataType,
            decltype(in_block_desc_nperblock_hperblock_wperblock),
            decltype(out_grid_desc_n_h_w),
            InBlockTransferAccessOrder,
            InBlockTransferAccessOrder,
            2,
            DstVectorDimAfterMerge,
            1,
            DstScalarPerVector,
            1,
            1,
            true,
            true>(in_block_desc_nperblock_hperblock_wperblock,
                  make_multi_index(0, 0, 0),
                  PassThrough{},
                  out_grid_desc_n_h_w,
                  make_multi_index(
                      n_block_data_idx_on_grid, h_block_data_idx_on_grid, w_block_data_idx_on_grid),
                  elementwise_op);

        in_global_load.Run(in_grid_desc_n_h_w,
                           in_global_buf,
                           in_block_desc_nperblock_hperblock_wperblock,
                           in_block_buf,
                           I0);

        out_global_store.Run(in_block_desc_nperblock_hperblock_wperblock,
                             in_block_buf,
                             out_grid_desc_n_h_w,
                             out_global_buf,
                             I0);
    }
};

} // namespace ck
