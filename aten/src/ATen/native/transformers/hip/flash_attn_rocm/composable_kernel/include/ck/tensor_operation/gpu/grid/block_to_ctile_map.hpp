// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/math.hpp"
#include "ck/utility/number.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"

namespace ck {

// Rows of column-vectors
template <index_t MPerBlock,
          index_t NPerBlock,
          typename CGridDesc_M_N,
          bool DeviceCTileIndexCheck = false>
struct BlockToCTileMap_M00_N0_M01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ BlockToCTileMap_M00_N0_M01() = default;

    __host__ __device__ BlockToCTileMap_M00_N0_M01(const CGridDesc_M_N& c_grid_desc_m_n,
                                                   index_t M01 = 1)
        : M01_(M01), underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01))
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);

        const index_t grid_size = M00 * M01_ * N0;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return underlying_map_.CalculateBottomIndex(idx_top);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return DefaultValidCTileIndex(c_tile_idx, c_tile_dim);
        else
            return true;
    }

    __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return true; // validity check moved to kernel

        const index_t M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        if(M0 % M01_ == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private:
    __host__ __device__ static constexpr auto
    GetBlockToCTileMap(const CGridDesc_M_N& c_grid_desc_m_n, index_t M01)
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01);

        const auto m00_n0_m01_to_m0_n0_block_cluster_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_insert_transform(1),
                       make_unmerge_transform(make_tuple(M00, M01)),
                       make_pass_through_transform(make_tuple(N0))),
            make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));

        const auto cblockid_to_m00_n0_m01_block_cluster_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(1, M00, N0, M01))),
            make_tuple(Sequence<0, 1, 2, 3>{}),
            make_tuple(Sequence<0>{}));

        const auto cblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(m00_n0_m01_to_m0_n0_block_cluster_adaptor,
                                  cblockid_to_m00_n0_m01_block_cluster_adaptor);

        return cblockid_to_m0_n0_block_cluster_adaptor;
    }

    index_t M01_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1));
    UnderlyingMap underlying_map_;
};

// Rows of column-vectors
// This C-tile map dynamically adjusts M01 when C-tile index is out of range
template <index_t MPerBlock, index_t NPerBlock, typename CGridDesc_M_N>
struct BlockToCTileMap_M00_N0_M01Adapt
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ BlockToCTileMap_M00_N0_M01Adapt() = default;

    __host__ __device__ BlockToCTileMap_M00_N0_M01Adapt(const CGridDesc_M_N& c_grid_desc_m_n,
                                                        index_t M01 = 8)
        : M01_(M01), c_grid_desc_m_n_(c_grid_desc_m_n)
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const index_t grid_size = M0 * N0;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        auto block_1d_id = idx_top[I0];

        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I1), NPerBlock);

        block_1d_id = block_1d_id % (M0 * N0); // swallow batch index

        index_t idx_N0 = block_1d_id % N0;
        index_t idx_M0 = block_1d_id / N0;

        const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

        index_t idx_M00          = idx_M0 / M01_;
        index_t idx_M01          = idx_M0 % M01_;
        index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

        return make_tuple(idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                          idx_N0_M01_local / M01_adapt);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                             const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    __host__ bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const { return true; }

    private:
    index_t M01_;
    CGridDesc_M_N c_grid_desc_m_n_;
};

// 2D slices of column-vectors in 3D space
// This C-tile map dynamically adjusts M01 when C-tile index is out of range
template <index_t MPerBlock, index_t NPerBlock, typename CGridDesc_M_N>
struct BlockToCTileMap_KSplit_M00_N0_M01Adapt
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ BlockToCTileMap_KSplit_M00_N0_M01Adapt() = default;

    __host__ __device__ BlockToCTileMap_KSplit_M00_N0_M01Adapt(const CGridDesc_M_N& c_grid_desc_m_n,
                                                               index_t M01    = 8,
                                                               index_t KSplit = 1)
        : M01_(M01), KSplit_(KSplit), c_grid_desc_m_n_(c_grid_desc_m_n)
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const index_t grid_size = M0 * N0 * KSplit_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        auto block_1d_id = idx_top[I0];

        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I1), NPerBlock);

        block_1d_id = block_1d_id % (M0 * N0 * KSplit_); // hide groups

        const index_t idx_ksplit = block_1d_id / (M0 * N0);
        block_1d_id              = block_1d_id % (M0 * N0);

        index_t idx_N0 = block_1d_id % N0;
        index_t idx_M0 = block_1d_id / N0;

        const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

        index_t idx_M00          = idx_M0 / M01_;
        index_t idx_M01          = idx_M0 % M01_;
        index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

        return make_tuple(idx_ksplit,
                          idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                          idx_N0_M01_local / M01_adapt);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                             const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    __host__ bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const { return true; }

    private:
    index_t M01_;
    index_t KSplit_;
    CGridDesc_M_N c_grid_desc_m_n_;
};

// Blocks of row-vectors
template <index_t MPerBlock,
          index_t NPerBlock,
          typename CGridDesc_M_N,
          bool DeviceCTileIndexCheck = false>
struct BlockToCTileMap_M00_N00_M01_N01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ BlockToCTileMap_M00_N00_M01_N01() = default;

    __host__ __device__ BlockToCTileMap_M00_N00_M01_N01(const CGridDesc_M_N& c_grid_desc_m_n,
                                                        index_t M01 = 1,
                                                        index_t N01 = 1)
        : M01_(M01), N01_(N01), underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01, N01))
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);
        const auto N00 = math::integer_divide_ceil(N0, N01_);

        const index_t grid_size = M00 * M01_ * N00 * N01_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return underlying_map_.CalculateBottomIndex(idx_top);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return DefaultValidCTileIndex(c_tile_idx, c_tile_dim);
        else
            return true;
    }

    __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return true; // validity check moved to kernel

        const index_t M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const index_t N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);
        if(M0 % M01_ == 0 && N0 % N01_ == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private:
    __host__ __device__ static constexpr auto
    GetBlockToCTileMap(const CGridDesc_M_N& c_grid_desc_m_n, index_t M01, index_t N01)
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01);
        const auto N00 = math::integer_divide_ceil(N0, N01);

        const auto m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_insert_transform(1), // swallow the carry from lower dimensions
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto cblockid_to_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(1, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto cblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  cblockid_to_m00_m01_n00_n01_block_cluster_adaptor);

        return cblockid_to_m0_n0_block_cluster_adaptor;
    }

    index_t M01_, N01_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1, 1));
    UnderlyingMap underlying_map_;
};

// 2D slices of row-vectors in 3D space
template <index_t MPerBlock,
          index_t NPerBlock,
          typename CGridDesc_M_N,
          bool DeviceCTileIndexCheck = false>
struct BlockToCTileMap_KSplit_M00_N00_M01_N01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ BlockToCTileMap_KSplit_M00_N00_M01_N01() = default;

    __host__ BlockToCTileMap_KSplit_M00_N00_M01_N01(const CGridDesc_M_N& c_grid_desc_m_n,
                                                    index_t M01    = 1,
                                                    index_t N01    = 1,
                                                    index_t KSplit = 1)
        : c_grid_desc_m_n_(c_grid_desc_m_n),
          M01_(M01),
          N01_(N01),
          KSplit_(KSplit),
          underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01, N01, KSplit))
    {
    }

    __host__ __device__ constexpr index_t
    CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);
        const auto N00 = math::integer_divide_ceil(N0, N01_);

        const index_t grid_size = M00 * M01_ * N00 * N01_ * KSplit_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        static_assert(TopIdx::Size() == 1);

        return underlying_map_.CalculateBottomIndex(
            make_multi_index(idx_top[I0] % CalculateGridSize()));
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return DefaultValidCTileIndex(c_tile_idx, c_tile_dim);
        else
            return true;
    }

    __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return true; // validity check moved to kernel

        const index_t M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const index_t N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);
        if(M0 % M01_ == 0 && N0 % N01_ == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private:
    __device__ constexpr index_t CalculateGridSize() const
    {
        return CalculateGridSize(c_grid_desc_m_n_);
    }

    __host__ static constexpr auto GetBlockToCTileMap(const CGridDesc_M_N& c_grid_desc_m_n,
                                                      index_t M01,
                                                      index_t N01,
                                                      index_t KSplit)
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01);
        const auto N00 = math::integer_divide_ceil(N0, N01);

        const auto ksplit_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(KSplit),
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto c_blockid_to_ksplit_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(KSplit, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto c_blockid_to_ksplit_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(ksplit_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  c_blockid_to_ksplit_m00_m01_n00_n01_block_cluster_adaptor);

        return c_blockid_to_ksplit_m0_n0_block_cluster_adaptor;
    }

    CGridDesc_M_N c_grid_desc_m_n_;
    index_t M01_, N01_, KSplit_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1, 1, 1));
    UnderlyingMap underlying_map_;
};

template <typename CTileIdx, typename CTileDim>
__host__ __device__ bool DefaultValidCTileIndex(const CTileIdx& c_tile_idx,
                                                const CTileDim& c_tile_dim)
{
    bool is_valid = false;

    const index_t m_block = c_tile_dim[Number<0>{}];
    const index_t n_block = c_tile_dim[Number<1>{}];

    if constexpr(CTileIdx::Size() == 2)
    {
        const index_t m_block_idx = c_tile_idx[Number<0>{}];
        const index_t n_block_idx = c_tile_idx[Number<1>{}];
        if(0 <= m_block_idx && m_block_idx < m_block && 0 <= n_block_idx && n_block_idx < n_block)
        {
            is_valid = true;
        }
    }
    else if constexpr(CTileIdx::Size() == 3)
    {
        const index_t ksplit_idx  = c_tile_idx[Number<0>{}];
        const index_t m_block_idx = c_tile_idx[Number<1>{}];
        const index_t n_block_idx = c_tile_idx[Number<2>{}];
        if(0 <= m_block_idx && m_block_idx < m_block && 0 <= n_block_idx && n_block_idx < n_block)
        {
            is_valid = true;
        }
        ignore = ksplit_idx;
    }

    return is_valid;
}

// This wrapper class is for grouped gemm where it subtracts blockIdx by a value so that the
// workgroups assigned to a given gemm problem have top index offsetted to range [0,
// grid_size_per_gemm]
template <typename UnderlyingBlockToCTileMap>
struct OffsettedBlockToCTileMap
{
    using underlying_type = UnderlyingBlockToCTileMap;

    OffsettedBlockToCTileMap(UnderlyingBlockToCTileMap block_to_ctile_map, index_t block_start)
    {
        block_to_ctile_map_ = block_to_ctile_map;
        block_start_        = block_start;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return block_to_ctile_map_.CalculateBottomIndex(
            make_multi_index(idx_top[Number<0>{}] - block_start_));
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        return block_to_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
    }

    template <typename CGridDesc_M_N>
    __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        return block_to_ctile_map_.CheckValidity(c_grid_desc_m_n);
    }

    template <typename CGridDesc_M_N>
    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        return block_to_ctile_map_.CalculateGridSize(c_grid_desc_m_n);
    }

    UnderlyingBlockToCTileMap block_to_ctile_map_;
    index_t block_start_;
};

} // namespace ck
