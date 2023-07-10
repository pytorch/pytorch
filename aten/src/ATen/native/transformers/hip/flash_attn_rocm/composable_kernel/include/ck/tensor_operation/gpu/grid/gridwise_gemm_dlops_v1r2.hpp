// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_GRIDWISE_GEMM_DLOPS_V1R2_HPP
#define CK_GRIDWISE_GEMM_DLOPS_V1R2_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_gemm_dlops_v2r2.hpp"
#include "blockwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_set.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AKM0M1GridDesc,
          typename BKN0N1GridDesc,
          typename CM0M10M11N0N10N11GridDesc,
          typename CBlockIdToM0N0BlockClusterAdaptor,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v1r2(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AKM0M1GridDesc a_k_m0_m1_grid_desc,
            const BKN0N1GridDesc b_k_n0_n1_grid_desc,
            const CM0M10M11N0N10N11GridDesc c_m0_m10_m11_n0_n10_n11_grid_desc,
            const CBlockIdToM0N0BlockClusterAdaptor cblockid_to_m0_n0_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_shared_block,
                      a_k_m0_m1_grid_desc,
                      b_k_n0_n1_grid_desc,
                      c_m0_m10_m11_n0_n10_n11_grid_desc,
                      cblockid_to_m0_n0_block_cluster_adaptor,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AKMGridDesc,
          typename BKNGridDesc,
          typename CMNGridDesc,
          index_t MPerBlockM1,
          index_t NPerBlockN1,
          index_t KPerBlock,
          index_t M1PerThreadM111,
          index_t N1PerThreadN111,
          index_t KPerThread,
          index_t M11N11ThreadClusterM1100,
          index_t M11N11ThreadClusterN1100,
          index_t M11N11ThreadClusterM1101,
          index_t M11N11ThreadClusterN1101,
          typename ABlockTransferThreadSliceLengths_K_M0_M1,
          typename ABlockTransferThreadClusterLengths_K_M0_M1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_M1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K_N0_N1,
          typename BBlockTransferThreadClusterLengths_K_N0_N1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_N1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridStepHacks,
          typename BGridStepHacks,
          typename CGridStepHacks,
          typename AGridMoveSliceWindowStepHacks,
          typename BGridMoveSliceWindowStepHacks>
struct GridwiseGemmDlops_km_kn_mn_v1r2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = math::lcm(Number<ABlockTransferDstScalarPerVector_M1>{},
                                                 Number<BBlockTransferDstScalarPerVector_N1>{},
                                                 Number<M1PerThreadM111>{},
                                                 Number<N1PerThreadN111>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlockM1>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<KPerBlock>{}, Number<NPerBlockN1>{}), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size =
            math::integer_least_multiple(a_k_m_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size =
            math::integer_least_multiple(b_k_n_block_desc.GetElementSpaceSize(), max_lds_align);

        return 2 * (a_block_aligned_space_size + b_block_aligned_space_size) * sizeof(FloatAB);
    }

    __host__ __device__ static constexpr bool CheckValidity(const AKMGridDesc& a_k_m_grid_desc,
                                                            const BKNGridDesc& b_k_n_grid_desc,
                                                            const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M = a_k_m_grid_desc.GetLength(I1);
        const auto N = b_k_n_grid_desc.GetLength(I1);
        const auto K = a_k_m_grid_desc.GetLength(I0);

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)

        return (M == c_m_n_grid_desc.GetLength(I0) && N == c_m_n_grid_desc.GetLength(I1) &&
                K == b_k_n_grid_desc.GetLength(I0)) &&
               (M % MPerBlockM1 == 0 && N % NPerBlockN1 == 0 && K % KPerBlock == 0);
    }

    __host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
    {
        const index_t grid_size = (M / MPerBlockM1) * (N / NPerBlockN1);

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const bool has_main_k_block_loop = (K + KPerBlock) / (2 * KPerBlock) > 1;

        return has_main_k_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasDoubleTailKBlockLoop(index_t K)
    {
        const bool has_double_tail_k_block_loop = (K / KPerBlock) % 2 == 0;

        return has_double_tail_k_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeAKM0M1GridDescriptor(const AKMGridDesc& a_k_m_grid_desc)
    {
        const auto K = a_k_m_grid_desc.GetLength(I0);
        const auto M = a_k_m_grid_desc.GetLength(I1);

        const auto M1 = Number<MPerBlockM1>{};
        const auto M0 = M / M1;

        const auto a_k_m0_m1_grid_desc = transform_tensor_descriptor(
            a_k_m_grid_desc,
            make_tuple(make_pass_through_transform(K), make_unmerge_transform(make_tuple(M0, M1))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}));

        return a_k_m0_m1_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeBKN0N1GridDescriptor(const BKNGridDesc& b_k_n_grid_desc)
    {
        const auto K = b_k_n_grid_desc.GetLength(I0);
        const auto N = b_k_n_grid_desc.GetLength(I1);

        const auto N1 = Number<NPerBlockN1>{};
        const auto N0 = N / N1;

        const auto b_k_n0_n1_grid_desc = transform_tensor_descriptor(
            b_k_n_grid_desc,
            make_tuple(make_pass_through_transform(K), make_unmerge_transform(make_tuple(N0, N1))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}));

        return b_k_n0_n1_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCM0M10M11N0N10N11GridDescriptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        constexpr auto M1 = Number<MPerBlockM1>{};
        constexpr auto N1 = Number<NPerBlockN1>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        constexpr auto M11 =
            Number<M11N11ThreadClusterM1100 * M11N11ThreadClusterM1101 * M1PerThreadM111>{};
        constexpr auto N11 =
            Number<M11N11ThreadClusterN1100 * M11N11ThreadClusterN1101 * N1PerThreadN111>{};

        constexpr auto M10 = M1 / M11;
        constexpr auto N10 = N1 / N11;

        const auto c_m0_m10_m11_n0_n10_n11_grid_desc = transform_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(M0, M10, M11)),
                       make_unmerge_transform(make_tuple(N0, N10, N11))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}));

        return c_m0_m10_m11_n0_n10_n11_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockIdToM0N0BlockClusterAdaptor(const CMNGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        constexpr auto M1 = Number<MPerBlockM1>{};
        constexpr auto N1 = Number<NPerBlockN1>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        const auto cblockid_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(make_tuple(make_merge_transform(make_tuple(M0, N0))),
                                             make_tuple(Sequence<0, 1>{}),
                                             make_tuple(Sequence<0>{}));

        return cblockid_to_m0_n0_block_cluster_adaptor;
    }

    using AKM0M1GridDesc            = decltype(MakeAKM0M1GridDescriptor(AKMGridDesc{}));
    using BKN0N1GridDesc            = decltype(MakeBKN0N1GridDescriptor(BKNGridDesc{}));
    using CM0M10M11N0N10N11GridDesc = decltype(MakeCM0M10M11N0N10N11GridDescriptor(CMNGridDesc{}));
    using CBlockIdToM0N0BlockClusterAdaptor =
        decltype(MakeCBlockIdToM0N0BlockClusterAdaptor(CMNGridDesc{}));

    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        FloatAB* __restrict__ p_shared_block,
        const AKM0M1GridDesc& a_k_m0_m1_grid_desc,
        const BKN0N1GridDesc& b_k_n0_n1_grid_desc,
        const CM0M10M11N0N10N11GridDesc& c_m0_m10_m11_n0_n10_n11_grid_desc,
        const CBlockIdToM0N0BlockClusterAdaptor& cblockid_to_m0_n0_block_cluster_adaptor,
        integral_constant<bool, HasMainKBlockLoop>,
        integral_constant<bool, HasDoubleTailKBlockLoop>)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_k_m0_m1_grid_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_k_n0_n1_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_m0_m10_m11_n0_n10_n11_grid_desc.GetElementSpaceSize());

        const auto K = a_k_m0_m1_grid_desc.GetLength(I0);

        // divide block work by [M, N]
        const auto c_m0_n0_block_cluster_idx =
            cblockid_to_m0_n0_block_cluster_adaptor.CalculateBottomIndex(
                make_multi_index(get_block_1d_id()));

        // HACK: this force index data into SGPR
        const index_t im0 = __builtin_amdgcn_readfirstlane(c_m0_n0_block_cluster_idx[I0]);
        const index_t in0 = __builtin_amdgcn_readfirstlane(c_m0_n0_block_cluster_idx[I1]);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(Number<ABlockTransferDstScalarPerVector_M1>{},
                                                 Number<BBlockTransferDstScalarPerVector_N1>{},
                                                 Number<M1PerThreadM111>{},
                                                 Number<N1PerThreadN111>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<KPerBlock>{}, Number<MPerBlockM1>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<KPerBlock>{}, Number<NPerBlockN1>{}), max_lds_align);

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_k_m0_m1_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<KPerBlock>{}, I1, Number<MPerBlockM1>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_k_n0_n1_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<KPerBlock>{}, I1, Number<NPerBlockN1>{}), max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum::Set,
                                            Sequence<KPerBlock, 1, MPerBlockM1>,
                                            ABlockTransferThreadSliceLengths_K_M0_M1,
                                            ABlockTransferThreadClusterLengths_K_M0_M1,
                                            ABlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(a_k_m0_m1_grid_desc),
                                            decltype(a_k_m0_m1_block_desc),
                                            ABlockTransferSrcAccessOrder,
                                            Sequence<0, 1, 2>,
                                            ABlockTransferSrcVectorDim,
                                            2,
                                            ABlockTransferSrcScalarPerVector,
                                            ABlockTransferDstScalarPerVector_M1,
                                            1,
                                            1,
                                            AThreadTransferSrcResetCoordinateAfterRun,
                                            true>(a_k_m0_m1_grid_desc,
                                                  make_multi_index(0, im0, 0),
                                                  a_k_m0_m1_block_desc,
                                                  make_multi_index(0, 0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum::Set,
                                            Sequence<KPerBlock, 1, NPerBlockN1>,
                                            BBlockTransferThreadSliceLengths_K_N0_N1,
                                            BBlockTransferThreadClusterLengths_K_N0_N1,
                                            BBlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(b_k_n0_n1_grid_desc),
                                            decltype(b_k_n0_n1_block_desc),
                                            BBlockTransferSrcAccessOrder,
                                            Sequence<0, 1, 2>,
                                            BBlockTransferSrcVectorDim,
                                            2,
                                            BBlockTransferSrcScalarPerVector,
                                            BBlockTransferDstScalarPerVector_N1,
                                            1,
                                            1,
                                            BThreadTransferSrcResetCoordinateAfterRun,
                                            true>(b_k_n0_n1_grid_desc,
                                                  make_multi_index(0, in0, 0),
                                                  b_k_n0_n1_block_desc,
                                                  make_multi_index(0, 0, 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, MPerBlockM1] is in LDS
        //     b_mtx[KPerBlocl, NPerBlockN1] is in LDS
        //     c_mtx[MPerBlockM1, NPerBlockN1] is distributed among threads, and saved in
        //       register
        const auto blockwise_gemm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v2r2_pipeline_2x2<BlockSize,
                                                                FloatAB,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_k_m_block_desc),
                                                                decltype(b_k_n_block_desc),
                                                                M1PerThreadM111,
                                                                N1PerThreadN111,
                                                                KPerThread,
                                                                M11N11ThreadClusterM1100,
                                                                M11N11ThreadClusterN1100,
                                                                M11N11ThreadClusterM1101,
                                                                M11N11ThreadClusterN1101,
                                                                M1PerThreadM111,
                                                                N1PerThreadN111>{};
        constexpr auto c_m10_m11_n10_n11_thread_tensor_lengths =
            decltype(blockwise_gemm)::GetCM0M1N0N1ThreadTensorLengths();

        constexpr auto c_m10_m11_n10_n11_thread_desc = make_naive_tensor_descriptor_packed(
            sequence_to_tuple_of_number(c_m10_m11_n10_n11_thread_tensor_lengths));

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size =
            math::integer_least_multiple(a_k_m0_m1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size =
            math::integer_least_multiple(b_k_n0_n1_block_desc.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block_double = p_shared_block;
        FloatAB* p_b_block_double = p_shared_block + 2 * a_block_aligned_space_size;

        // register allocation for output
        auto c_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAcc>(
            c_m10_m11_n10_n11_thread_desc.GetElementSpaceSize());

        ThreadwiseTensorSliceSet_v1<FloatAcc,
                                    decltype(c_m10_m11_n10_n11_thread_desc),
                                    decltype(c_m10_m11_n10_n11_thread_tensor_lengths)>{}
            .Run(c_m10_m11_n10_n11_thread_desc,
                 make_tuple(I0, I0, I0, I0),
                 c_thread_buf,
                 FloatAcc{0});

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_k_m0_m1_global_step_hacks = AGridStepHacks{};
        constexpr auto b_k_n0_n1_global_step_hacks = BGridStepHacks{};

        // hack to control index calculation when move slice window for A and B matrix for
        // threadwise copy
        constexpr auto a_k_m0_m1_global_move_slice_window_step_hack =
            AGridMoveSliceWindowStepHacks{};
        constexpr auto b_k_n0_n1_global_move_slice_window_step_hack =
            BGridMoveSliceWindowStepHacks{};

        auto a_block_even_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block_double, a_k_m0_m1_block_desc.GetElementSpaceSize());
        auto b_block_even_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_b_block_double, b_k_n0_n1_block_desc.GetElementSpaceSize());

        auto a_block_odd_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block_double + a_block_aligned_space_size,
            a_k_m0_m1_block_desc.GetElementSpaceSize());
        auto b_block_odd_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_b_block_double + b_block_aligned_space_size,
            b_k_n0_n1_block_desc.GetElementSpaceSize());

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.RunRead(
                a_k_m0_m1_grid_desc, a_global_buf, a_k_m0_m1_global_step_hacks);
            b_blockwise_copy.RunRead(
                b_k_n0_n1_grid_desc, b_global_buf, b_k_n0_n1_global_step_hacks);

            a_blockwise_copy.RunWrite(a_k_m0_m1_block_desc, a_block_even_buf);
            b_blockwise_copy.RunWrite(b_k_n0_n1_block_desc, b_block_even_buf);
        }

        if constexpr(HasMainKBlockLoop)
        {
            index_t k_block_data_begin = 0;

            // LDS double buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                // even iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_k_m0_m1_grid_desc,
                                                    a_block_slice_copy_step,
                                                    a_k_m0_m1_global_move_slice_window_step_hack);
                b_blockwise_copy.MoveSrcSliceWindow(b_k_n0_n1_grid_desc,
                                                    b_block_slice_copy_step,
                                                    b_k_n0_n1_global_move_slice_window_step_hack);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_k_m0_m1_grid_desc, a_global_buf, a_k_m0_m1_global_step_hacks);
                b_blockwise_copy.RunRead(
                    b_k_n0_n1_grid_desc, b_global_buf, b_k_n0_n1_global_step_hacks);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(c_m10_m11_n10_n11_thread_desc,
                                   a_block_even_buf,
                                   b_block_even_buf,
                                   c_thread_buf);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_k_m0_m1_block_desc, a_block_odd_buf);
                b_blockwise_copy.RunWrite(b_k_n0_n1_block_desc, b_block_odd_buf);

                // odd iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_k_m0_m1_grid_desc,
                                                    a_block_slice_copy_step,
                                                    a_k_m0_m1_global_move_slice_window_step_hack);
                b_blockwise_copy.MoveSrcSliceWindow(b_k_n0_n1_grid_desc,
                                                    b_block_slice_copy_step,
                                                    b_k_n0_n1_global_move_slice_window_step_hack);

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_k_m0_m1_grid_desc, a_global_buf, a_k_m0_m1_global_step_hacks);
                b_blockwise_copy.RunRead(
                    b_k_n0_n1_grid_desc, b_global_buf, b_k_n0_n1_global_step_hacks);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(
                    c_m10_m11_n10_n11_thread_desc, a_block_odd_buf, b_block_odd_buf, c_thread_buf);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_k_m0_m1_block_desc, a_block_even_buf);
                b_blockwise_copy.RunWrite(b_k_n0_n1_block_desc, b_block_even_buf);

                k_block_data_begin += 2 * KPerBlock;
            } while(k_block_data_begin < K - 2 * KPerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_k_m0_m1_grid_desc,
                                                a_block_slice_copy_step,
                                                a_k_m0_m1_global_move_slice_window_step_hack);
            b_blockwise_copy.MoveSrcSliceWindow(b_k_n0_n1_grid_desc,
                                                b_block_slice_copy_step,
                                                b_k_n0_n1_global_move_slice_window_step_hack);

            __syncthreads();

            // LDS double buffer: load last data from device mem
            a_blockwise_copy.RunRead(
                a_k_m0_m1_grid_desc, a_global_buf, a_k_m0_m1_global_step_hacks);
            b_blockwise_copy.RunRead(
                b_k_n0_n1_grid_desc, b_global_buf, b_k_n0_n1_global_step_hacks);

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(
                c_m10_m11_n10_n11_thread_desc, a_block_even_buf, b_block_even_buf, c_thread_buf);

            // LDS double buffer: store last data to LDS
            a_blockwise_copy.RunWrite(a_k_m0_m1_block_desc, a_block_odd_buf);
            b_blockwise_copy.RunWrite(b_k_n0_n1_block_desc, b_block_odd_buf);

            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(
                c_m10_m11_n10_n11_thread_desc, a_block_odd_buf, b_block_odd_buf, c_thread_buf);
        }
        else // if has 1 iteration left
        {
            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(
                c_m10_m11_n10_n11_thread_desc, a_block_even_buf, b_block_even_buf, c_thread_buf);
        }

        // output: register to global memory
        {
            constexpr auto c_m0_m10_m11_n0_n10_n11_thread_desc =
                make_naive_tensor_descriptor_packed(
                    make_tuple(I1,
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I0]>{},
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I1]>{},
                               I1,
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I2]>{},
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I3]>{}));

            const auto c_m10_m11_n10_n11_thread_origin_idx_on_block =
                blockwise_gemm.CalculateCM0M1N0N1ThreadOriginOnBlock(get_thread_local_1d_id());

            ThreadwiseTensorSliceTransfer_v1r3<
                FloatAcc,
                FloatC,
                decltype(c_m0_m10_m11_n0_n10_n11_thread_desc),
                decltype(c_m0_m10_m11_n0_n10_n11_grid_desc),
                Sequence<1,
                         c_m10_m11_n10_n11_thread_tensor_lengths[I0],
                         c_m10_m11_n10_n11_thread_tensor_lengths[I1],
                         1,
                         c_m10_m11_n10_n11_thread_tensor_lengths[I2],
                         c_m10_m11_n10_n11_thread_tensor_lengths[I3]>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                true>{c_m0_m10_m11_n0_n10_n11_grid_desc,
                      make_multi_index(im0,
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I0],
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I1],
                                       in0,
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I2],
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I3])}
                .Run(c_m0_m10_m11_n0_n10_n11_thread_desc,
                     make_tuple(I0, I0, I0, I0, I0, I0),
                     c_thread_buf,
                     c_m0_m10_m11_n0_n10_n11_grid_desc,
                     c_grid_buf,
                     CGridStepHacks{});
        }
    }
};

} // namespace ck
#endif
