// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename ContractionMultiDKernelArg,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_contraction_multiple_d_xdl_cshuffle(
            const void CK_CONSTANT_ADDRESS_SPACE* contraction_args,
            const index_t group_count,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const index_t block_id = get_block_1d_id();

    const auto contraction_arg_ptr = reinterpret_cast<const ContractionMultiDKernelArg*>(
        cast_pointer_to_generic_address_space(contraction_args));

    index_t left     = 0;
    index_t right    = group_count;
    index_t group_id = index_t((left + right) / 2);

    while((!(block_id >= contraction_arg_ptr[group_id].block_start_ &&
             block_id < contraction_arg_ptr[group_id].block_end_)) &&
          left <= right)
    {
        if(block_id < contraction_arg_ptr[group_id].block_start_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    GridwiseGemm::template Run<HasMainKBlockLoop>(
        contraction_arg_ptr[group_id].p_a_grid_,
        contraction_arg_ptr[group_id].p_b_grid_,
        contraction_arg_ptr[group_id].p_ds_grid_,
        contraction_arg_ptr[group_id].p_e_grid_,
        p_shared,
        a_element_op,
        b_element_op,
        cde_element_op,
        contraction_arg_ptr[group_id].a_grid_desc_ak0_m_ak1_,
        contraction_arg_ptr[group_id].b_grid_desc_bk0_n_bk1_,
        contraction_arg_ptr[group_id].ds_grid_desc_mblock_mperblock_nblock_nperblock_,
        contraction_arg_ptr[group_id].e_grid_desc_mblock_mperblock_nblock_nperblock_,
        contraction_arg_ptr[group_id].block_2_etile_map_);
#else
    ignore = contraction_args;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
#endif
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// Tensor Contraction:
//   input : A
//   input : B
//   input : D0, D1, ...
//   output : E
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   A[M0, M1, M2, ..., K0, K1, K2, ...]
//   B[N0, N1, N2, ..., K0, K1, K2, ...]
//   D[M0, M1, M2, ..., N0, N1, N2, ...]
//   E[M0, M1, M2, ..., N0, N1, N2, ...]
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          TensorSpecialization ASpec,
          TensorSpecialization BSpec,
          TensorSpecialization DESpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceGroupedContractionMultipleD_Xdl_CShuffle
    : public DeviceGroupedContractionMultipleD<NumDimM,
                                               NumDimN,
                                               NumDimK,
                                               ADataType,
                                               BDataType,
                                               DsDataType,
                                               EDataType,
                                               AElementwiseOperation,
                                               BElementwiseOperation,
                                               CDEElementwiseOperation>
{
    using DeviceOp = DeviceGroupedContractionMultipleD_Xdl_CShuffle;

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    // Assume: A[M0, M1, M2, ..., K0, K1, K2, ...]
    static auto MakeAGridDescriptor_M_K(const std::vector<index_t>& a_ms_ks_lengths_vec,
                                        const std::vector<index_t>& a_ms_ks_strides_vec)
    {
        assert(a_ms_ks_lengths_vec.size() == NumDimM + NumDimK &&
               a_ms_ks_strides_vec.size() == NumDimM + NumDimK);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto a_ms_ks_lengths = to_tuple(a_ms_ks_lengths_vec, Number<NumDimM + NumDimK>{});
        const auto a_ms_ks_strides = to_tuple(a_ms_ks_strides_vec, Number<NumDimM + NumDimK>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimK, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(a_ms_ks_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(a_ms_ks_lengths, kDimIds);

        if constexpr(ASpec == TensorSpecialization::Packed)
        {
            auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
            auto K = container_reduce(kLengths, math::multiplies{}, Number<1>{});
            const auto a_grid_desc_mraw_kraw = make_naive_tensor_descriptor(
                make_tuple(M, K),
                make_tuple(a_ms_ks_strides[Number<NumDimM - 1>{}],
                           a_ms_ks_strides[Number<NumDimM + NumDimK - 1>{}]));
            return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
        }
        else
        {
            // naive tensor A[M0, M1, M2, ..., K0, K1, K2...]
            const auto a_grid_desc_ms_ks =
                make_naive_tensor_descriptor(a_ms_ks_lengths, a_ms_ks_strides);

            // transformed tensor A[MRaw = M0 * M1 * M2 * ... , KRaw = K0 * K1 * K2 * ...]
            const auto a_grid_desc_mraw_kraw = transform_tensor_descriptor(
                a_grid_desc_ms_ks,
                make_tuple(make_merge_transform(mLengths), make_merge_transform(kLengths)),
                make_tuple(mDimIds, kDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
        }
    }

    // Assume: B[N0, N1, N2, ..., K0, K1, K2, ...]
    static auto MakeBGridDescriptor_N_K(const std::vector<index_t>& b_ns_ks_lengths_vec,
                                        const std::vector<index_t>& b_ns_ks_strides_vec)
    {
        assert(b_ns_ks_lengths_vec.size() == NumDimN + NumDimK &&
               b_ns_ks_strides_vec.size() == NumDimN + NumDimK);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto b_ns_ks_lengths = to_tuple(b_ns_ks_lengths_vec, Number<NumDimN + NumDimK>{});
        const auto b_ns_ks_strides = to_tuple(b_ns_ks_strides_vec, Number<NumDimN + NumDimK>{});

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds = typename arithmetic_sequence_gen<0, NumDimN, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimN, NumDimN + NumDimK, 1>::type{};

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(b_ns_ks_lengths, kDimIds);

        // lengths for N0, N1, ...
        const auto nLengths = get_container_subset(b_ns_ks_lengths, nDimIds);

        if constexpr(BSpec == TensorSpecialization::Packed)
        {
            auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
            auto K = container_reduce(kLengths, math::multiplies{}, Number<1>{});
            const auto b_grid_desc_nraw_kraw = make_naive_tensor_descriptor(
                make_tuple(N, K),
                make_tuple(b_ns_ks_strides[Number<NumDimN - 1>{}],
                           b_ns_ks_strides[Number<NumDimN + NumDimK - 1>{}]));
            return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
        }
        else
        {
            // naive tensor B[N0, N1, N2, ..., K0, K1, K2, ...]
            const auto b_grid_desc_ns_ks =
                make_naive_tensor_descriptor(b_ns_ks_lengths, b_ns_ks_strides);

            // transformed tensor B[NRaw = N0 * N1 * N2 * ..., KRaw = K0 * K1 * K2 * ...]
            const auto b_grid_desc_nraw_kraw = transform_tensor_descriptor(
                b_grid_desc_ns_ks,
                make_tuple(make_merge_transform(nLengths), make_merge_transform(kLengths)),
                make_tuple(nDimIds, kDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
        }
    }

    // assume E[M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeEGridDescriptor_M_N(const std::vector<index_t>& e_ms_ns_lengths_vec,
                                        const std::vector<index_t>& e_ms_ns_strides_vec)
    {
        assert(e_ms_ns_lengths_vec.size() == NumDimM + NumDimN &&
               e_ms_ns_strides_vec.size() == NumDimM + NumDimN);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto e_ms_ns_lengths = to_tuple(e_ms_ns_lengths_vec, Number<NumDimM + NumDimN>{});
        const auto e_ms_ns_strides = to_tuple(e_ms_ns_strides_vec, Number<NumDimM + NumDimN>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimN, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(e_ms_ns_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto nLengths = get_container_subset(e_ms_ns_lengths, nDimIds);

        if constexpr(DESpec == TensorSpecialization::Packed)
        {
            auto M = container_reduce(mLengths, math::multiplies{}, Number<1>{});
            auto N = container_reduce(nLengths, math::multiplies{}, Number<1>{});
            const auto e_grid_desc_mraw_nraw = make_naive_tensor_descriptor(
                make_tuple(M, N),
                make_tuple(e_ms_ns_strides[Number<NumDimM - 1>{}],
                           e_ms_ns_strides[Number<NumDimM + NumDimN - 1>{}]));
            return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
        }
        else
        {
            // naive tensor E[M0, M1, M2, ..., N0, N1, N2...]
            const auto e_grid_desc_ms_ns =
                make_naive_tensor_descriptor(e_ms_ns_lengths, e_ms_ns_strides);

            // transformed tensor E[MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 * N2 * ...]
            const auto e_grid_desc_mraw_nraw = transform_tensor_descriptor(
                e_grid_desc_ms_ns,
                make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
                make_tuple(mDimIds, nDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
        }
    }

    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths_vec,
        const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides_vec)
    {
        return generate_tuple(
            [&](auto i) {
                return DeviceOp::MakeEGridDescriptor_M_N(ds_ms_ns_lengths_vec[i],
                                                         ds_ms_ns_strides_vec[i]);
            },
            Number<NumDTensor>{});
    }

    using AGridDesc_M_K  = decltype(MakeAGridDescriptor_M_K({}, {}));
    using BGridDesc_N_K  = decltype(MakeBGridDescriptor_N_K({}, {}));
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({{}}, {{}}))>;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N({}, {}));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleD_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1                          = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1                          = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(BGridDesc_N_K{}))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DsGridDesc_M_N{}))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock  = remove_cvref_t<decltype(
        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(EGridDesc_M_N{}))>;

    struct GroupedContractionBlock2ETileMap
    {
        // block-to-e-tile map
        using Block2ETileMap =
            remove_cvref_t<decltype(GridwiseGemm::MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

        GroupedContractionBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n,
                                         ck::index_t BlockStart)
        {
            default_block_2_etile_map_ = GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n);
            block_start_               = BlockStart;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            return default_block_2_etile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[I0] - block_start_));
        }

        // it's actually E-Tile
        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return default_block_2_etile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        __host__ bool CheckValidity(const EGridDesc_M_N& e_grid_desc_m_n) const
        {
            return default_block_2_etile_map_.CheckValidity(e_grid_desc_m_n);
        }

        Block2ETileMap default_block_2_etile_map_;
        ck::index_t block_start_;
    };

    struct ContractionMultiDKernelArg
    {
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // lock-to-e-tile map
        GroupedContractionBlock2ETileMap block_2_etile_map_;

        ck::index_t block_start_, block_end_;
    };

    struct ContractionMultiDDeviceArg
    {
        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // Strides for the last M/N/K dimensions of A/B/Ds/E
        //   for sanity check of vector load/store
        index_t a_mz_stride_;
        index_t a_kz_stride_;
        index_t b_nz_stride_;
        index_t b_kz_stride_;
        std::array<index_t, NumDTensor> ds_nz_stride_;
        // index_t e_mz_stride_;
        index_t e_nz_stride_;
    };

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*> p_a_vec,
                 std::vector<const void*> p_b_vec,
                 std::vector<std::array<const void*, NumDTensor>> p_ds_vec,
                 std::vector<void*> p_e_vec,
                 std::vector<ContractionDesc<NumDTensor>> contraction_descs,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            group_count_ = contraction_descs.size();

            if(!(group_count_ == p_a_vec.size() && group_count_ == p_b_vec.size() &&
                 group_count_ == p_e_vec.size()))
            {
                throw std::runtime_error("wrong! group_count_ != a/b/e_vec.size");
            }

            contraction_multi_d_kernel_args_.reserve(group_count_);

            grid_size_ = 0;

            for(std::size_t i = 0; i < group_count_; i++)
            {
                const auto p_a_grid = static_cast<const ADataType*>(p_a_vec[i]);
                const auto p_b_grid = static_cast<const BDataType*>(p_b_vec[i]);
                const auto p_e_grid = static_cast<EDataType*>(p_e_vec[i]);

                const auto a_grid_desc_m_k = DeviceOp::MakeAGridDescriptor_M_K(
                    contraction_descs[i].a_ms_ks_lengths, contraction_descs[i].a_ms_ks_strides);
                const auto b_grid_desc_n_k = DeviceOp::MakeBGridDescriptor_N_K(
                    contraction_descs[i].b_ns_ks_lengths, contraction_descs[i].b_ns_ks_strides);

                DsGridDesc_M_N ds_grid_desc_m_n;
                typename GridwiseGemm::DsGridPointer p_ds_grid;

                // populate pointer, batch stride, desc for Ds
                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    using DDataType = remove_cvref_t<tuple_element_t<j.value, DsDataType>>;

                    // D pointer
                    p_ds_grid(j) = static_cast<const DDataType*>(p_ds_vec[i][j]);

                    // D desc
                    ds_grid_desc_m_n(j) =
                        DeviceOp::MakeEGridDescriptor_M_N(contraction_descs[i].ds_ms_ns_lengths[j],
                                                          contraction_descs[i].ds_ms_ns_strides[j]);
                });

                const auto e_grid_desc_m_n = DeviceOp::MakeEGridDescriptor_M_N(
                    contraction_descs[i].e_ms_ns_lengths, contraction_descs[i].e_ms_ns_strides);

                const auto a_grid_desc_ak0_m_ak1 =
                    GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k);
                const auto b_grid_desc_bk0_n_bk1 =
                    GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k);

                const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        ds_grid_desc_m_n);
                const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n);

                const index_t grid_size_grp =
                    GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n)
                        .CalculateGridSize(e_grid_desc_m_n);

                const index_t BlockStart = grid_size_;
                const index_t BlockEnd   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                const auto block_2_etile_map =
                    GroupedContractionBlock2ETileMap(e_grid_desc_m_n, BlockStart);

                // for sanity check of vector memory access
                const index_t a_mz_stride = contraction_descs[i].a_ms_ks_strides[NumDimM - 1];
                const index_t a_kz_stride =
                    contraction_descs[i].a_ms_ks_strides[NumDimM + NumDimK - 1];

                const index_t b_nz_stride = contraction_descs[i].b_ns_ks_strides[NumDimN - 1];
                const index_t b_kz_stride =
                    contraction_descs[i].b_ns_ks_strides[NumDimN + NumDimK - 1];

                std::array<index_t, NumDTensor> ds_nz_stride;
                for(index_t j = 0; j < NumDTensor; ++j)
                {
                    ds_nz_stride[j] =
                        contraction_descs[i].ds_ms_ns_strides[j][NumDimM + NumDimN - 1];
                }

                const index_t e_nz_stride =
                    contraction_descs[i].e_ms_ns_strides[NumDimM + NumDimN - 1];

                if(GridwiseGemm::CheckValidity(a_grid_desc_m_k,
                                               b_grid_desc_n_k,
                                               ds_grid_desc_m_n,
                                               e_grid_desc_m_n,
                                               block_2_etile_map))
                {
                    contraction_multi_d_kernel_args_.push_back(
                        {p_a_grid,
                         p_b_grid,
                         p_ds_grid,
                         p_e_grid,
                         a_grid_desc_ak0_m_ak1,
                         b_grid_desc_bk0_n_bk1,
                         ds_grid_desc_mblock_mperblock_nblock_nperblock,
                         e_grid_desc_mblock_mperblock_nblock_nperblock,
                         block_2_etile_map,
                         BlockStart,
                         BlockEnd});

                    contraction_multi_d_device_args_.push_back({a_grid_desc_m_k,
                                                                b_grid_desc_n_k,
                                                                ds_grid_desc_m_n,
                                                                e_grid_desc_m_n,
                                                                a_mz_stride,
                                                                a_kz_stride,
                                                                b_nz_stride,
                                                                b_kz_stride,
                                                                ds_nz_stride,
                                                                e_nz_stride});
                }
            }
        }

        std::vector<ContractionMultiDKernelArg> contraction_multi_d_kernel_args_;
        std::vector<ContractionMultiDDeviceArg> contraction_multi_d_device_args_;

        std::size_t group_count_;
        index_t grid_size_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            bool has_main_k_block_loop = true;

            for(std::size_t i = 0; i < arg.group_count_; i++)
            {
                const auto K =
                    arg.contraction_multi_d_kernel_args_[i].a_grid_desc_ak0_m_ak1_.GetLength(I0) *
                    arg.contraction_multi_d_kernel_args_[i].a_grid_desc_ak0_m_ak1_.GetLength(I2);

                if(GridwiseGemm::CalculateHasMainKBlockLoop(K) != has_main_k_block_loop)
                {
                    throw std::runtime_error("wrong! not all gemm has_main_k_block_loop");
                }
            }

            hipGetErrorString(hipMemcpy(arg.p_workspace_,
                                        arg.contraction_multi_d_kernel_args_.data(),
                                        arg.contraction_multi_d_kernel_args_.size() *
                                            sizeof(ContractionMultiDKernelArg),
                                        hipMemcpyHostToDevice));

            float ave_time = 0;

            auto launch_kernel = [&](auto has_main_k_block_loop_) {
                const auto kernel =
                    kernel_grouped_contraction_multiple_d_xdl_cshuffle<GridwiseGemm,
                                                                       ContractionMultiDKernelArg,
                                                                       AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CDEElementwiseOperation,
                                                                       has_main_k_block_loop_>;

                return launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(arg.grid_size_),
                    dim3(BlockSize),
                    0,
                    cast_pointer_to_constant_address_space(arg.p_workspace_),
                    arg.group_count_,
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.cde_element_op_);
            };

            if(has_main_k_block_loop)
            {
                ave_time = launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                ave_time = launch_kernel(integral_constant<bool, false>{});
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a"))
        {
            return false;
        }

        for(std::size_t i = 0; i < arg.group_count_; i++)
        {
            const auto a_grid_desc_m_k_ = arg.contraction_multi_d_device_args_[i].a_grid_desc_m_k_;
            const auto b_grid_desc_n_k_ = arg.contraction_multi_d_device_args_[i].b_grid_desc_n_k_;
            const auto ds_grid_desc_m_n_ =
                arg.contraction_multi_d_device_args_[i].ds_grid_desc_m_n_;
            const auto e_grid_desc_m_n_ = arg.contraction_multi_d_device_args_[i].e_grid_desc_m_n_;
            const auto a_grid_desc_ak0_m_ak1_ =
                arg.contraction_multi_d_kernel_args_[i].a_grid_desc_ak0_m_ak1_;
            const auto b_grid_desc_bk0_n_bk1_ =
                arg.contraction_multi_d_kernel_args_[i].b_grid_desc_bk0_n_bk1_;
            const auto ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                arg.contraction_multi_d_kernel_args_[i]
                    .ds_grid_desc_mblock_mperblock_nblock_nperblock_;
            const auto e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                arg.contraction_multi_d_kernel_args_[i]
                    .e_grid_desc_mblock_mperblock_nblock_nperblock_;

            const auto block_2_etile_map_ =
                arg.contraction_multi_d_kernel_args_[i].block_2_etile_map_;

            const auto a_mz_stride_  = arg.contraction_multi_d_device_args_[i].a_mz_stride_;
            const auto a_kz_stride_  = arg.contraction_multi_d_device_args_[i].a_kz_stride_;
            const auto b_nz_stride_  = arg.contraction_multi_d_device_args_[i].b_nz_stride_;
            const auto b_kz_stride_  = arg.contraction_multi_d_device_args_[i].b_kz_stride_;
            const auto ds_nz_stride_ = arg.contraction_multi_d_device_args_[i].ds_nz_stride_;
            const auto e_nz_stride_  = arg.contraction_multi_d_device_args_[i].e_nz_stride_;

            if(!GridwiseGemm::CheckValidity(a_grid_desc_m_k_,
                                            b_grid_desc_n_k_,
                                            ds_grid_desc_m_n_,
                                            e_grid_desc_m_n_,
                                            block_2_etile_map_))
            {
                return false;
            }

            // check vector access
            static_assert((ABlockTransferSrcVectorDim == 1 || ABlockTransferSrcVectorDim == 2) &&
                              (BBlockTransferSrcVectorDim == 1 || BBlockTransferSrcVectorDim == 2),
                          "wrong!");

            // vector memory access of A: could be on M or AK1 dimension
            if constexpr(ABlockTransferSrcVectorDim == 1)
            {
                if(!(a_mz_stride_ == 1 &&
                     a_grid_desc_ak0_m_ak1_.GetLength(I1) % ABlockTransferSrcScalarPerVector == 0))
                {
                    return false;
                }
            }
            else
            {
                if(!(a_kz_stride_ == 1 &&
                     a_grid_desc_ak0_m_ak1_.GetLength(I2) % ABlockTransferSrcScalarPerVector == 0))
                {
                    return false;
                }
            }

            // vector memory access of B: could be on N or BK1 dimension
            if constexpr(BBlockTransferSrcVectorDim == 1)
            {
                if(!(b_nz_stride_ == 1 &&
                     b_grid_desc_bk0_n_bk1_.GetLength(I1) % BBlockTransferSrcScalarPerVector == 0))
                {
                    return false;
                }
            }
            else
            {
                if(!(b_kz_stride_ == 1 &&
                     b_grid_desc_bk0_n_bk1_.GetLength(I2) % BBlockTransferSrcScalarPerVector == 0))
                {
                    return false;
                }
            }

            // vector memory access of Ds: always on NPerBlock dimension
            bool valid_d_access = true;

            static_for<0, NumDTensor, 1>{}([&](auto j) {
                if(!(ds_nz_stride_[j] == 1 &&
                     ds_grid_desc_mblock_mperblock_nblock_nperblock_[j].GetLength(I3) %
                             CDEBlockTransferScalarPerVector_NPerBlock ==
                         0))
                {
                    valid_d_access = false;
                }
            });

            if(valid_d_access == false)
            {
                return false;
            }

            // vector memory access of E: always on NPerBlock dimension
            if(!(e_nz_stride_ == 1 && e_grid_desc_mblock_mperblock_nblock_nperblock_.GetLength(I3) %
                                              CDEBlockTransferScalarPerVector_NPerBlock ==
                                          0))
            {
                return false;
            }
        }

        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*> p_a_vec,
                             std::vector<const void*> p_b_vec,
                             std::vector<std::array<const void*, NumDTensor>> p_ds_vec,
                             std::vector<void*> p_e_vec,
                             std::vector<ContractionDesc<NumDTensor>> contraction_descs,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a_vec,
                        p_b_vec,
                        p_ds_vec,
                        p_e_vec,
                        contraction_descs,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*> p_a_vec,
                        std::vector<const void*> p_b_vec,
                        std::vector<std::array<const void*, NumDTensor>> p_ds_vec,
                        std::vector<void*> p_e_vec,
                        std::vector<ContractionDesc<NumDTensor>> contraction_descs,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_a_vec,
                                          p_b_vec,
                                          p_ds_vec,
                                          p_e_vec,
                                          contraction_descs,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedContractionMultipleD_Xdl_CShuffle"
            << "<"
            << NumDimM << ", "
            << NumDimN << ", "
            << NumDimK << ", "
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << ABlockTransferSrcVectorDim << ", "
            << BBlockTransferSrcVectorDim
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ *
               sizeof(ContractionMultiDKernelArg);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
