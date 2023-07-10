// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"

using namespace ck;

static auto I0 = Number<0>{};
static auto I1 = Number<1>{};
static auto I2 = Number<2>{};

TEST(BlockToCTileMap, TestBlockToCTileMap_M00_N00_M01_N01_DeviceCTileIndexCheck1)
{
    const index_t M         = 384;
    const index_t N         = 384;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;
    const index_t MBlock    = M / MPerBlock;
    const index_t NBlock    = N / NPerBlock;
    const index_t M01       = 4;
    const index_t N01       = 4;

    auto c_grid_desc_m_n = make_naive_tensor_descriptor_packed(make_tuple(M, N));

    printf("(M, N, MPerBlock, NPerBlock, M01, N01) = (%d, %d, %d, %d, %d, %d)\n",
           M,
           N,
           MPerBlock,
           NPerBlock,
           M01,
           N01);

    BlockToCTileMap_M00_N00_M01_N01<MPerBlock, NPerBlock, decltype(c_grid_desc_m_n), true> tile_map(
        c_grid_desc_m_n, M01, N01);

    EXPECT_TRUE(tile_map.CheckValidity(c_grid_desc_m_n) == true);
    EXPECT_TRUE(tile_map.CalculateGridSize(c_grid_desc_m_n) == 16);

    // clang-format off
    std::vector<std::vector<int>> expected_m0idx_n0idx_valid = {
        {0, 0, 1},
        {0, 1, 1},
        {0, 2, 1},
        {0, 3, 0},
        {1, 0, 1},
        {1, 1, 1},
        {1, 2, 1},
        {1, 3, 0},
        {2, 0, 1},
        {2, 1, 1},
        {2, 2, 1},
        {2, 3, 0},
        {3, 0, 0},
        {3, 1, 0},
        {3, 2, 0},
        {3, 3, 0}
    };
    // clang-format on

    for(index_t i = 0; i < tile_map.CalculateGridSize(c_grid_desc_m_n); i++)
    {
        auto m0n0_idx = tile_map.CalculateBottomIndex(make_multi_index(i));
        std::cout << "block_1d_id = " << i << ", m0, n0 = " << m0n0_idx[I0] << ", " << m0n0_idx[I1];
        std::cout << ", valid = " << tile_map.ValidCTileIndex(m0n0_idx, make_tuple(MBlock, NBlock))
                  << std::endl;
        bool equal =
            expected_m0idx_n0idx_valid[i] ==
            std::vector<int>{m0n0_idx[I0],
                             m0n0_idx[I1],
                             tile_map.ValidCTileIndex(m0n0_idx, make_tuple(MBlock, NBlock))};
        EXPECT_TRUE(equal);
    }
}

TEST(BlockToCTileMap, TestBlockToCTileMap_M00_N00_M01_N01_DeviceCTileIndexCheck0)
{
    const index_t M         = 384;
    const index_t N         = 384;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;

    const index_t M01 = 4;
    const index_t N01 = 4;

    auto c_grid_desc_m_n = make_naive_tensor_descriptor_packed(make_tuple(M, N));

    printf("(M, N, MPerBlock, NPerBlock, M01, N01) = (%d, %d, %d, %d, %d, %d)\n",
           M,
           N,
           MPerBlock,
           NPerBlock,
           M01,
           N01);

    BlockToCTileMap_M00_N00_M01_N01<MPerBlock, NPerBlock, decltype(c_grid_desc_m_n), false>
        tile_map(c_grid_desc_m_n, M01, N01);

    EXPECT_TRUE(tile_map.CheckValidity(c_grid_desc_m_n) == false);
}

TEST(BlockToCTileMap, TestBlockToCTileMap_M00_N0_M01_DeviceCTileIndexCheck1)
{
    const index_t M         = 384;
    const index_t N         = 512;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;
    const index_t MBlock    = M / MPerBlock;
    const index_t NBlock    = N / NPerBlock;
    const index_t M01       = 4;

    auto c_grid_desc_m_n = make_naive_tensor_descriptor_packed(make_tuple(M, N));

    printf("(M, N, MPerBlock, NPerBlock, M01) = (%d, %d, %d, %d, %d)\n",
           M,
           N,
           MPerBlock,
           NPerBlock,
           M01);

    BlockToCTileMap_M00_N0_M01<MPerBlock, NPerBlock, decltype(c_grid_desc_m_n), true> tile_map(
        c_grid_desc_m_n, M01);

    EXPECT_TRUE(tile_map.CheckValidity(c_grid_desc_m_n) == true);
    EXPECT_TRUE(tile_map.CalculateGridSize(c_grid_desc_m_n) == 16);

    // clang-format off
    std::vector<std::vector<int>> expected_m0idx_n0idx_valid = {
        {0, 0, 1},
        {1, 0, 1},
        {2, 0, 1},
        {3, 0, 0},
        {0, 1, 1},
        {1, 1, 1},
        {2, 1, 1},
        {3, 1, 0},
        {0, 2, 1},
        {1, 2, 1},
        {2, 2, 1},
        {3, 2, 0},
        {0, 3, 1},
        {1, 3, 1},
        {2, 3, 1},
        {3, 3, 0}
    };
    // clang-format on

    for(index_t i = 0; i < tile_map.CalculateGridSize(c_grid_desc_m_n); i++)
    {
        auto m0n0_idx = tile_map.CalculateBottomIndex(make_multi_index(i));
        std::cout << "block_1d_id = " << i << ", m0, n0 = " << m0n0_idx[I0] << ", " << m0n0_idx[I1];
        std::cout << ", valid = " << tile_map.ValidCTileIndex(m0n0_idx, make_tuple(MBlock, NBlock))
                  << std::endl;
        bool equal =
            expected_m0idx_n0idx_valid[i] ==
            std::vector<int>{m0n0_idx[I0],
                             m0n0_idx[I1],
                             tile_map.ValidCTileIndex(m0n0_idx, make_tuple(MBlock, NBlock))};
        EXPECT_TRUE(equal);
    }
}

TEST(BlockToCTileMap, TestBlockToCTileMap_M00_N0_M01_DeviceCTileIndexCheck0)
{
    const index_t M         = 512;
    const index_t N         = 384;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;

    auto c_grid_desc_m_n = make_naive_tensor_descriptor_packed(make_tuple(M, N));

    // clang-format off
    std::vector<std::tuple<int, int, bool>> expected_m0_gridsize_validity = {
        {5, 15, false},
        {4, 12, true},
        {3, 18, false},
        {2, 12, true},
        {1, 12, true}
    };
    // clang-format on

    for(auto e : expected_m0_gridsize_validity)
    {
        const index_t M01 = std::get<0>(e);

        printf("(M, N, MPerBlock, NPerBlock, M01) = (%d, %d, %d, %d, %d)\n",
               M,
               N,
               MPerBlock,
               NPerBlock,
               M01);

        BlockToCTileMap_M00_N0_M01<MPerBlock, NPerBlock, decltype(c_grid_desc_m_n), false> tile_map(
            c_grid_desc_m_n, M01);

        EXPECT_EQ(tile_map.CalculateGridSize(c_grid_desc_m_n), std::get<1>(e));
        EXPECT_EQ(tile_map.CheckValidity(c_grid_desc_m_n), std::get<2>(e));
    }
}

TEST(BlockToCTileMap, TestBlockToCTileMap_M00_N0_M01Adapt)
{
    const index_t M         = 768;
    const index_t N         = 384;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;
    const index_t MBlock    = M / MPerBlock;
    const index_t NBlock    = N / NPerBlock;
    constexpr index_t M01   = 4;

    auto c_grid_desc_m_n = make_naive_tensor_descriptor_packed(make_tuple(M, N));

    printf("(M, N, MPerBlock, NPerBlock, M01) = (%d, %d, %d, %d, %d)\n",
           M,
           N,
           MPerBlock,
           NPerBlock,
           M01);

    BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, decltype(c_grid_desc_m_n)> tile_map(
        c_grid_desc_m_n, M01);

    EXPECT_TRUE(tile_map.CheckValidity(c_grid_desc_m_n) == true);
    EXPECT_TRUE(tile_map.CalculateGridSize(c_grid_desc_m_n) == 18);

    // clang-format off
    std::vector<std::vector<int>> expected_m0idx_n0idx_valid = {
        {0, 0, 1},
        {1, 0, 1},
        {2, 0, 1},
        {3, 0, 1},
        {0, 1, 1},
        {1, 1, 1},
        {2, 1, 1},
        {3, 1, 1},
        {0, 2, 1},
        {1, 2, 1},
        {2, 2, 1},
        {3, 2, 1},
        {4, 0, 1},
        {5, 0, 1},
        {4, 1, 1},
        {5, 1, 1},
        {4, 2, 1},
        {5, 2, 1},
    };
    // clang-format on

    for(index_t i = 0; i < tile_map.CalculateGridSize(c_grid_desc_m_n); i++)
    {
        auto m0n0_idx = tile_map.CalculateBottomIndex(make_multi_index(i));
        std::cout << "block_1d_id = " << i << ", m0, n0 = " << m0n0_idx[I0] << ", " << m0n0_idx[I1];
        std::cout << ", valid = " << tile_map.ValidCTileIndex(m0n0_idx, make_tuple(MBlock, NBlock))
                  << std::endl;
        bool equal =
            expected_m0idx_n0idx_valid[i] ==
            std::vector<int>{m0n0_idx[I0],
                             m0n0_idx[I1],
                             tile_map.ValidCTileIndex(m0n0_idx, make_tuple(MBlock, NBlock))};
        EXPECT_TRUE(equal);
    }
}

TEST(BlockToCTileMap, TestBlockToCTileMap_KSplit_M00_N0_M01Adapt)
{
    const index_t M         = 768;
    const index_t N         = 384;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;
    const index_t MBlock    = M / MPerBlock;
    const index_t NBlock    = N / NPerBlock;
    constexpr index_t M01   = 4;
    const index_t KSplit    = 3;

    auto c_grid_desc_m_n = make_naive_tensor_descriptor_packed(make_tuple(M, N));

    printf("(M, N, MPerBlock, NPerBlock, M01) = (%d, %d, %d, %d, %d)\n",
           M,
           N,
           MPerBlock,
           NPerBlock,
           M01);

    BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, decltype(c_grid_desc_m_n)>
        tile_map(c_grid_desc_m_n, M01, KSplit);

    EXPECT_TRUE(tile_map.CheckValidity(c_grid_desc_m_n) == true);
    EXPECT_TRUE(tile_map.CalculateGridSize(c_grid_desc_m_n) == 18 * KSplit);

    std::vector<std::vector<int>> expected_ksplitidx_m0idx_n0idx_valid = {
        {0, 0, 0, 1}, {0, 1, 0, 1}, {0, 2, 0, 1}, {0, 3, 0, 1}, {0, 0, 1, 1}, {0, 1, 1, 1},
        {0, 2, 1, 1}, {0, 3, 1, 1}, {0, 0, 2, 1}, {0, 1, 2, 1}, {0, 2, 2, 1}, {0, 3, 2, 1},
        {0, 4, 0, 1}, {0, 5, 0, 1}, {0, 4, 1, 1}, {0, 5, 1, 1}, {0, 4, 2, 1}, {0, 5, 2, 1},
        {1, 0, 0, 1}, {1, 1, 0, 1}, {1, 2, 0, 1}, {1, 3, 0, 1}, {1, 0, 1, 1}, {1, 1, 1, 1},
        {1, 2, 1, 1}, {1, 3, 1, 1}, {1, 0, 2, 1}, {1, 1, 2, 1}, {1, 2, 2, 1}, {1, 3, 2, 1},
        {1, 4, 0, 1}, {1, 5, 0, 1}, {1, 4, 1, 1}, {1, 5, 1, 1}, {1, 4, 2, 1}, {1, 5, 2, 1},
        {2, 0, 0, 1}, {2, 1, 0, 1}, {2, 2, 0, 1}, {2, 3, 0, 1}, {2, 0, 1, 1}, {2, 1, 1, 1},
        {2, 2, 1, 1}, {2, 3, 1, 1}, {2, 0, 2, 1}, {2, 1, 2, 1}, {2, 2, 2, 1}, {2, 3, 2, 1},
        {2, 4, 0, 1}, {2, 5, 0, 1}, {2, 4, 1, 1}, {2, 5, 1, 1}, {2, 4, 2, 1}, {2, 5, 2, 1},
    };

    for(index_t i = 0; i < tile_map.CalculateGridSize(c_grid_desc_m_n); i++)
    {
        auto ksplitm0n0_idx = tile_map.CalculateBottomIndex(make_multi_index(i));
        std::cout << "block_1d_id = " << i << ", ksplit, m0, n0 = " << ksplitm0n0_idx[I0] << ", "
                  << ksplitm0n0_idx[I1] << ", " << ksplitm0n0_idx[I2];
        std::cout << ", valid = "
                  << tile_map.ValidCTileIndex(ksplitm0n0_idx, make_tuple(MBlock, NBlock))
                  << std::endl;
        bool equal =
            expected_ksplitidx_m0idx_n0idx_valid[i] ==
            std::vector<int>{ksplitm0n0_idx[I0],
                             ksplitm0n0_idx[I1],
                             ksplitm0n0_idx[I2],
                             tile_map.ValidCTileIndex(ksplitm0n0_idx, make_tuple(MBlock, NBlock))};
        EXPECT_TRUE(equal);
    }
}
