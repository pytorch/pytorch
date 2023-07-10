// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_GRIDWISE_GEMM_V2_HPP
#define CK_GRIDWISE_GEMM_V2_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_tensor_slice_transfer.hpp"
#include "threadwise_tensor_slice_transfer.hpp"
#include "blockwise_gemm_dlops_v3.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          index_t KPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t EPerBlock,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E_K,
          typename ABlockTransferThreadClusterLengths_E_K,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGlobalStepHacks,
          typename BGlobalStepHacks,
          typename CGlobalStepHacks,
          typename AGlobalMoveSliceWindowStepHacks,
          typename BGlobalMoveSliceWindowStepHacks>
struct GridwiseGemmDlops_km_kn_mn_v3
{
    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto E = EPerBlock * 3 * 3;

        constexpr auto max_lds_align =
            math::lcm(Number<ABlockTransferDstScalarPerVector_K>{}, Number<KPerBlock>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_e_k_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E>{}, Number<KPerBlock>{}), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_e_k_desc.GetElementSpaceSize(), max_lds_align);

        return a_block_space_size * sizeof(FloatAB);
    }

    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const AGlobalDesc& a_e_k_global_desc,
                        const FloatAB* __restrict__ p_a_global,
                        const BGlobalDesc& b_e_n_ho_wo_global_desc,
                        const FloatAB* __restrict__ p_b_global,
                        const CGlobalDesc& c_k_n_ho_wo_global_desc,
                        FloatC* __restrict__ p_c_global,
                        FloatAB* __restrict__ p_shared_block,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_global, a_e_k_global_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_global, b_e_n_ho_wo_global_desc.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_global, c_k_n_ho_wo_global_desc.GetElementSpaceSize());

        constexpr auto E = EPerBlock * 3 * 3;

        // const auto E = a_e_k_global_desc.GetLength(I0);
        const auto K = a_e_k_global_desc.GetLength(I1);

        const auto N  = b_e_n_ho_wo_global_desc.GetLength(I1);
        const auto Ho = b_e_n_ho_wo_global_desc.GetLength(I2);
        const auto Wo = b_e_n_ho_wo_global_desc.GetLength(I3);

// divide block work by [M, N]
#if 0
        const auto ho_block_work_num  = Ho / Number<HoPerBlock>{};
        const auto wo_block_work_num  = Wo / Number<WoPerBlock>{};
        const auto hwo_block_work_num = ho_block_work_num * wo_block_work_num;

        const index_t k_block_work_id   = get_block_1d_id() / hwo_block_work_num;
        const index_t hwo_block_work_id = get_block_1d_id() - k_block_work_id * hwo_block_work_num;

        const index_t ho_block_work_id = hwo_block_work_id / wo_block_work_num;
        const index_t wo_block_work_id = hwo_block_work_id - ho_block_work_id * wo_block_work_num;
#else
        // Hack: this force result into SGPR
        const index_t ho_block_work_num  = __builtin_amdgcn_readfirstlane(Ho / HoPerBlock);
        const index_t wo_block_work_num  = __builtin_amdgcn_readfirstlane(Wo / WoPerBlock);
        const index_t hwo_block_work_num = ho_block_work_num * wo_block_work_num;

        const index_t k_block_work_id =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / hwo_block_work_num);
        const index_t hwo_block_work_id = get_block_1d_id() - k_block_work_id * hwo_block_work_num;

        const index_t ho_block_work_id =
            __builtin_amdgcn_readfirstlane(hwo_block_work_id / wo_block_work_num);
        const index_t wo_block_work_id = hwo_block_work_id - ho_block_work_id * wo_block_work_num;
#endif

        // lds max alignment
        constexpr auto max_lds_align =
            math::lcm(Number<ABlockTransferDstScalarPerVector_K>{}, Number<KPerBlock>{});

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_e_k_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<EPerBlock>{}, Number<KPerBlock>{}), max_lds_align);

        constexpr auto a_e_k_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E>{}, Number<KPerBlock>{}), max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_e_n_ho_wo_block_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<EPerBlock>{}, Number<1>{}, Number<HoPerBlock>{}, Number<WoPerBlock>{}));

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k_n_ho_wo_thread_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<KPerThread>{}, Number<1>{}, Number<HoPerThread>{}, Number<WoPerThread>{}));

        auto blockwise_gemm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v3<BlockSize,
                                                 FloatAB,
                                                 FloatAB,
                                                 FloatAcc,
                                                 decltype(a_e_k_block_desc),
                                                 decltype(b_e_n_ho_wo_block_desc),
                                                 decltype(c_k_n_ho_wo_thread_desc),
                                                 KPerThread,
                                                 HoPerThread,
                                                 WoPerThread,
                                                 EPerThread,
                                                 ABlockTransferSrcScalarPerVector,
                                                 ABlockTransferDstScalarPerVector_K>{};

        auto c_thread_mtx_index = blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        const auto k_thread_id  = c_thread_mtx_index.k;
        const auto ho_thread_id = c_thread_mtx_index.h;
        const auto wo_thread_id = c_thread_mtx_index.w;

        const index_t k_block_data_on_global  = k_block_work_id * KPerBlock;
        const index_t ho_block_data_on_global = ho_block_work_id * HoPerBlock;
        const index_t wo_block_data_on_global = wo_block_work_id * WoPerBlock;

        const index_t ho_thread_data_on_global =
            ho_block_data_on_global + ho_thread_id * HoPerThread;
        const index_t wo_thread_data_on_global =
            wo_block_data_on_global + wo_thread_id * WoPerThread;

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4<BlockSize,
                                            InMemoryDataOperationEnum::Set,
                                            Sequence<E, KPerBlock>,
                                            ABlockTransferThreadSliceLengths_E_K,
                                            ABlockTransferThreadClusterLengths_E_K,
                                            ABlockTransferThreadClusterArrangeOrder,
                                            FloatAB,
                                            FloatAB,
                                            decltype(a_e_k_global_desc),
                                            decltype(a_e_k_desc),
                                            ABlockTransferSrcAccessOrder,
                                            Sequence<0, 1>,
                                            ABlockTransferSrcVectorDim,
                                            1,
                                            ABlockTransferSrcScalarPerVector,
                                            ABlockTransferDstScalarPerVector_K,
                                            1,
                                            1,
                                            AThreadTransferSrcResetCoordinateAfterRun,
                                            true>(a_e_k_global_desc,
                                                  make_multi_index(0, k_block_data_on_global),
                                                  a_e_k_desc,
                                                  make_multi_index(0, 0));

        constexpr auto b_e_n_ho_wo_thread_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<EPerBlock>{}, Number<1>{}, Number<HoPerThread>{}, Number<WoPerThread>{}));

        auto b_threadwise_transfer =
            ThreadwiseTensorSliceTransfer_v2<FloatAB,
                                             FloatAB,
                                             decltype(b_e_n_ho_wo_global_desc),
                                             decltype(b_e_n_ho_wo_thread_desc),
                                             Sequence<EPerBlock, 1, HoPerThread, WoPerThread>,
                                             BBlockTransferSrcAccessOrder,
                                             BBlockTransferSrcVectorDim,
                                             BBlockTransferSrcScalarPerVector,
                                             1,
                                             true>(
                b_e_n_ho_wo_global_desc,
                make_multi_index(0, 0, ho_thread_data_on_global, wo_thread_data_on_global));

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_shared_block, a_e_k_desc.GetElementSpaceSize());

        // register allocation for output
        StaticBuffer<AddressSpaceEnum::Vgpr,
                     FloatAcc,
                     c_k_n_ho_wo_thread_desc.GetElementSpaceSize(),
                     true>
            c_thread_buf;

        // initialize output thread tensor
        ThreadwiseTensorSliceSet_v1<FloatAcc,
                                    decltype(c_k_n_ho_wo_thread_desc),
                                    Sequence<KPerThread, 1, HoPerThread, WoPerThread>>{}
            .Run(c_k_n_ho_wo_thread_desc, make_tuple(I0, I0, I0, I0), c_thread_buf, FloatAcc{0});

        constexpr auto b_thread_slice_copy_step = make_multi_index(EPerBlock, 0, 0, 0);

        // hack to control index calculation when iterating over A and B matrix for threadwise copy
        constexpr auto a_e_k_global_step_hacks       = AGlobalStepHacks{};
        constexpr auto b_e_n_ho_wo_global_step_hacks = BGlobalStepHacks{};

        // hack to control index calculation when move slice window for A and B matrix for
        // threadwise copy
        constexpr auto a_e_k_global_move_slice_window_step_hack = AGlobalMoveSliceWindowStepHacks{};
        constexpr auto b_e_n_ho_wo_global_move_slice_window_step_hack =
            BGlobalMoveSliceWindowStepHacks{};

        // double regsiter buffer for b
        StaticBuffer<AddressSpaceEnum::Vgpr,
                     FloatAB,
                     b_e_n_ho_wo_thread_desc.GetElementSpaceSize(),
                     true>
            b_thread_even_buf, b_thread_odd_buf;

        // LDS double buffer: preload data
        {
            a_blockwise_copy.RunRead(a_e_k_global_desc, a_global_buf, a_e_k_global_step_hacks);

            b_threadwise_transfer.Run(b_e_n_ho_wo_global_desc,
                                      b_global_buf,
                                      b_e_n_ho_wo_thread_desc,
                                      make_tuple(I0, I0, I0, I0),
                                      b_thread_even_buf,
                                      b_e_n_ho_wo_global_step_hacks);

            a_blockwise_copy.RunWrite(a_e_k_desc, a_block_buf);
        }

        __syncthreads();

        if constexpr(HasMainKBlockLoop)
        {
            index_t e_block_data_begin = 0;

            // LDS double buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                // even iteration
                b_threadwise_transfer.MoveSrcSliceWindow(b_e_n_ho_wo_global_desc,
                                                         b_thread_slice_copy_step);

                b_threadwise_transfer.Run(b_e_n_ho_wo_global_desc,
                                          b_global_buf,
                                          b_e_n_ho_wo_thread_desc,
                                          make_tuple(I0, I0, I0, I0),
                                          b_thread_odd_buf,
                                          b_e_n_ho_wo_global_step_hacks);

                // LDS double buffer: GEMM on current data
                // TODO: @Zhang Jing: blockwise gemm should be able to move slice window
                blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                blockwise_gemm.MoveASliceWindow(a_e_k_block_desc, make_tuple(EPerBlock, 0));

                b_threadwise_transfer.MoveSrcSliceWindow(b_e_n_ho_wo_global_desc,
                                                         b_thread_slice_copy_step);

                b_threadwise_transfer.Run(b_e_n_ho_wo_global_desc,
                                          b_global_buf,
                                          b_e_n_ho_wo_thread_desc,
                                          make_tuple(I0, I0, I0, I0),
                                          b_thread_even_buf,
                                          b_e_n_ho_wo_global_step_hacks);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);

                blockwise_gemm.MoveASliceWindow(a_e_k_block_desc, make_tuple(EPerBlock, 0));

                e_block_data_begin += 2 * EPerBlock;

            } while(e_block_data_begin < E - 2 * EPerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            b_threadwise_transfer.MoveSrcSliceWindow(b_e_n_ho_wo_global_desc,
                                                     b_thread_slice_copy_step);

            b_threadwise_transfer.Run(b_e_n_ho_wo_global_desc,
                                      b_global_buf,
                                      b_e_n_ho_wo_thread_desc,
                                      make_tuple(I0, I0, I0, I0),
                                      b_thread_odd_buf,
                                      b_e_n_ho_wo_global_step_hacks);

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

            blockwise_gemm.MoveASliceWindow(a_e_k_block_desc, make_tuple(EPerBlock, 0));

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);
        }
        else // if has 1 iteration left
        {
            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);
        }

        // output: register to global memory
        {
            // hack to control index calculation when iterating over c_k_n_ho_wo_global tensor
            constexpr auto c_k_n_ho_wo_global_tensor_step_hacks = CGlobalStepHacks{};

            const index_t k_thread_data_on_global =
                k_block_data_on_global + k_thread_id * KPerThread;

            ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                               FloatC,
                                               decltype(c_k_n_ho_wo_thread_desc),
                                               decltype(c_k_n_ho_wo_global_desc),
                                               Sequence<KPerThread, 1, HoPerThread, WoPerThread>,
                                               CThreadTransferSrcDstAccessOrder,
                                               CThreadTransferSrcDstVectorDim,
                                               CThreadTransferDstScalarPerVector,
                                               CGlobalMemoryDataOperation,
                                               1,
                                               true>(
                c_k_n_ho_wo_global_desc,
                make_multi_index(
                    k_thread_data_on_global, 0, ho_thread_data_on_global, wo_thread_data_on_global))
                .Run(c_k_n_ho_wo_thread_desc,
                     make_tuple(I0, I0, I0, I0),
                     c_thread_buf,
                     c_k_n_ho_wo_global_desc,
                     c_global_buf,
                     c_k_n_ho_wo_global_tensor_step_hacks);
        }
    }

    // pass tensor descriptor by reference
    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const AGlobalDesc& a_e_k_global_desc,
                        const FloatAB* __restrict__ p_a_global,
                        const BGlobalDesc& b_e_n_ho_wo_global_desc,
                        const FloatAB* __restrict__ p_b_global,
                        const CGlobalDesc& c_k_n_ho_wo_global_desc,
                        FloatC* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        constexpr index_t shared_block_size = GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

        __shared__ FloatAB p_shared_block[shared_block_size];

        Run(a_e_k_global_desc,
            p_a_global,
            b_e_n_ho_wo_global_desc,
            p_b_global,
            c_k_n_ho_wo_global_desc,
            p_c_global,
            p_shared_block,
            integral_constant<bool, HasMainKBlockLoop>{},
            integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }

    // pass tensor descriptors by their pointers
    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const AGlobalDesc* p_a_e_k_global_desc,
                        const FloatAB* __restrict__ p_a_global,
                        const BGlobalDesc* p_b_e_n_ho_wo_global_desc,
                        const FloatAB* __restrict__ p_b_global,
                        const CGlobalDesc* p_c_k_n_ho_wo_global_desc,
                        FloatC* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        const auto a_e_k_global_desc       = *p_a_e_k_global_desc;
        const auto b_e_n_ho_wo_global_desc = *p_b_e_n_ho_wo_global_desc;
        const auto c_k_n_ho_wo_global_desc = *p_c_k_n_ho_wo_global_desc;

        Run(a_e_k_global_desc,
            p_a_global,
            b_e_n_ho_wo_global_desc,
            p_b_global,
            c_k_n_ho_wo_global_desc,
            p_c_global,
            integral_constant<bool, HasMainKBlockLoop>{},
            integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }

    // pass tensor descriptors by void*
    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ void Run(const void* p_a_e_k_global_desc,
                        const FloatAB* __restrict__ p_a_global,
                        const void* p_b_e_n_ho_wo_global_desc,
                        const FloatAB* __restrict__ p_b_global,
                        const void* p_c_k_n_ho_wo_global_desc,
                        FloatC* __restrict__ p_c_global,
                        integral_constant<bool, HasMainKBlockLoop>,
                        integral_constant<bool, HasDoubleTailKBlockLoop>) const
    {
        const auto a_e_k_global_desc = *reinterpret_cast<const AGlobalDesc*>(p_a_e_k_global_desc);
        const auto b_e_n_ho_wo_global_desc =
            *reinterpret_cast<const BGlobalDesc*>(p_b_e_n_ho_wo_global_desc);
        const auto c_k_n_ho_wo_global_desc =
            *reinterpret_cast<const CGlobalDesc*>(p_c_k_n_ho_wo_global_desc);

        Run(a_e_k_global_desc,
            p_a_global,
            b_e_n_ho_wo_global_desc,
            p_b_global,
            c_k_n_ho_wo_global_desc,
            p_c_global,
            integral_constant<bool, HasMainKBlockLoop>{},
            integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }
};

} // namespace ck
#endif
