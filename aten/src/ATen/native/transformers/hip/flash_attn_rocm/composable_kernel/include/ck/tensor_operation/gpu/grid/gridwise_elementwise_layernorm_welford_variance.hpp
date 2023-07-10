// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

// X = Elementwise(input1, input2, input3, ...)
// Y = Normalization(X, beta, gamma)
template <typename InDataTypePointerTuple,
          typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename AccDataType,
          typename XElementwiseOperation,
          typename YElementwiseOperation,
          typename InGrid2dDescTuple,
          typename GridDesc_M_K,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorDim,
          index_t BetaSrcVectorSize,
          index_t YDstVectorDim,
          index_t YDstVectorSize,
          bool SweepOnce>
struct GridwiseElementwiseLayernormWelfordVariance_mk_to_mk
{
    static_assert((XSrcVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert((YDstVectorDim == 0 && MThreadSliceSize % YDstVectorSize == 0) ||
                      (YDstVectorDim == 1 && KThreadSliceSize % YDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr index_t NumInput = InDataTypePointerTuple::Size();

    static constexpr bool reorder_thread_cluster = (XSrcVectorDim == 0);

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using ThreadBufferDimAccessOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<XSrcVectorSize>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelford<AccDataType, ThreadReduceSrcDesc_M_K, ThreadReduceDstDesc_M>;

    using BlockwiseWelford = BlockwiseWelford<AccDataType,
                                              BlockSize,
                                              ThreadClusterLengths_M_K,
                                              ThreadClusterArrangeOrder>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t M_BlockTileSize     = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize     = KThreadClusterSize * KThreadSliceSize;
    static constexpr index_t K_BlockTileStepSize = KThreadClusterSize * XSrcVectorSize;

    static constexpr auto XThreadBufferNumber     = Number<KThreadSliceSize / XSrcVectorSize>{};
    static constexpr auto GammaThreadBufferNumber = Number<KThreadSliceSize / GammaSrcVectorSize>{};
    static constexpr auto BetaThreadBufferNumber  = Number<KThreadSliceSize / BetaSrcVectorSize>{};
    static constexpr auto YThreadBufferNumber     = Number<KThreadSliceSize / YDstVectorSize>{};

    __device__ static int GetKPerThread(const GridDesc_M_K& x_grid_desc_m_k,
                                        int thread_k_cluster_id)
    {
        int kPerBlock = x_grid_desc_m_k.GetTransforms()[I2].GetUpperLengths()[I0];
        int kPerThread =
            kPerBlock < K_BlockTileSize ? 0 : KThreadSliceSize * (kPerBlock / K_BlockTileSize);
        int kPerBlockTail = kPerBlock - kPerThread * KThreadClusterSize;

        if(kPerBlockTail > 0)
        {
            static_for<0, XThreadBufferNumber, 1>{}([&](auto i) {
                int thread_max_len =
                    (thread_k_cluster_id + 1) * XSrcVectorSize + K_BlockTileStepSize * i;
                int delta = thread_max_len - kPerBlockTail;
                delta     = math::clamp(thread_max_len - kPerBlockTail, 0, XSrcVectorSize);
                kPerThread += XSrcVectorSize - delta;
            });
        }

        return kPerThread;
    }

    __device__ static void Run(const InGrid2dDescTuple in_grid_2d_desc_tuple,
                               const GridDesc_M_K& x_grid_desc_m_k,
                               const GridDesc_M_K& gamma_grid_desc_m_k,
                               const GridDesc_M_K& beta_grid_desc_m_k,
                               const GridDesc_M_K& y_grid_desc_m_k,
                               index_t num_k_block_tile_iteration,
                               AccDataType epsilon,
                               const InDataTypePointerTuple p_in_global_tuple,
                               XDataType* const __restrict__ p_x_lds,
                               const GammaDataType* const __restrict__ p_gamma_global,
                               const BetaDataType* const __restrict__ p_beta_global,
                               YDataType* const __restrict__ p_y_global,
                               const XElementwiseOperation x_elementwise_op,
                               const YElementwiseOperation y_elementwise_op)
    {
        if constexpr(SweepOnce)
        {
            num_k_block_tile_iteration = 1;
        }

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t grid_size       = get_grid_size();

        auto in_global_buf_tuple = generate_tuple(
            [&](auto I) {
                static_assert(in_grid_2d_desc_tuple[I].GetNumOfDimension() ==
                              2); // matrix dimension

                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_in_global_tuple[I], in_grid_2d_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumInput>{});

        auto y_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_global, y_grid_desc_m_k.GetElementSpaceSize());

        auto x_lds_val_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_x_lds, x_grid_desc_m_k.GetElementSpaceSize() / grid_size);

        auto in_thread_buf_tuple = generate_tuple(
            [&](auto) {
                return generate_tuple(
                    [&](auto) {
                        return StaticBuffer<AddressSpaceEnum::Vgpr,
                                            AccDataType,
                                            MThreadSliceSize * XSrcVectorSize,
                                            true>{};
                    },
                    Number<NumInput>{});
            },
            Number<XThreadBufferNumber>{});

        auto x_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    AccDataType,
                                    MThreadSliceSize * XSrcVectorSize,
                                    true>{};
            },
            Number<XThreadBufferNumber>{});

        auto gamma_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    AccDataType,
                                    MThreadSliceSize * GammaSrcVectorSize,
                                    true>{};
            },
            Number<GammaThreadBufferNumber>{});

        auto beta_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    AccDataType,
                                    MThreadSliceSize * BetaSrcVectorSize,
                                    true>{};
            },
            Number<BetaThreadBufferNumber>{});

        auto y_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    AccDataType,
                                    MThreadSliceSize * YDstVectorSize,
                                    true>{};
            },
            Number<YThreadBufferNumber>{});

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> var_thread_buf;

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        using ThreadBufferLengths_M_K = Sequence<MThreadSliceSize, XSrcVectorSize>;

        constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<XSrcVectorSize>{}));

        auto in_global_load_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return ThreadwiseTensorSliceTransfer_v2<DataType,
                                                        AccDataType,
                                                        decltype(in_grid_2d_desc_tuple[I]),
                                                        decltype(thread_buffer_desc_m_k),
                                                        ThreadBufferLengths_M_K,
                                                        ThreadBufferDimAccessOrder,
                                                        XSrcVectorDim,
                                                        XSrcVectorSize,
                                                        1,
                                                        false>{
                    in_grid_2d_desc_tuple[I],
                    make_multi_index(block_global_id * M_BlockTileSize +
                                         thread_m_cluster_id * MThreadSliceSize,
                                     thread_k_cluster_id * XSrcVectorSize)};
            },
            Number<NumInput>{});

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  AccDataType,
                                                                  GridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * XSrcVectorSize));

        auto threadwise_gamma_load =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             AccDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             GammaSrcVectorDim,
                                             GammaSrcVectorSize,
                                             1,
                                             true>(
                gamma_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * GammaSrcVectorSize));

        auto threadwise_beta_load =
            ThreadwiseTensorSliceTransfer_v2<BetaDataType,
                                             AccDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             BetaSrcVectorDim,
                                             BetaSrcVectorSize,
                                             1,
                                             true>(
                beta_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * BetaSrcVectorSize));

        using PassThrough = tensor_operation::element_wise::PassThrough;
        PassThrough pass_through_op;
        auto threadwise_x_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               XDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               GridDesc_M_K,
                                               PassThrough,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               XSrcVectorDim,
                                               XSrcVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                x_grid_desc_m_k,
                make_multi_index(thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * XSrcVectorSize),
                pass_through_op);

        auto threadwise_y_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               YDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               GridDesc_M_K,
                                               YElementwiseOperation,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               YDstVectorDim,
                                               YDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                y_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * YDstVectorSize),
                y_elementwise_op);

        // Copy x from Cache
        // one pass: fwd, second pass: bwd
        constexpr auto thread_copy_fwd_step_m_k = make_multi_index(0, K_BlockTileStepSize);
        constexpr auto thread_copy_bwd_step_m_k =
            make_multi_index(0, SweepOnce ? 0 : -K_BlockTileSize);

        const auto gamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_gamma_global, gamma_grid_desc_m_k.GetElementSpaceSize());

        const auto beta_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_beta_global, beta_grid_desc_m_k.GetElementSpaceSize());

        auto threadwise_welford       = ThreadwiseWelford();
        threadwise_welford.max_count_ = GetKPerThread(x_grid_desc_m_k, thread_k_cluster_id);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            mean_thread_buf(I) = type_convert<AccDataType>(0.0f);
            var_thread_buf(I)  = type_convert<AccDataType>(0.0f);
        });

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {
            static_for<0, XThreadBufferNumber, 1>{}([&](auto iK0) {
                static_for<0, NumInput, 1>{}([&](auto I) { // input load loop
                    in_global_load_tuple(I).Run(in_grid_2d_desc_tuple[I],
                                                in_global_buf_tuple[I],
                                                thread_buffer_desc_m_k,
                                                make_tuple(I0, I0),
                                                in_thread_buf_tuple(iK0)(I));

                    in_global_load_tuple(I).MoveSrcSliceWindow(in_grid_2d_desc_tuple[I],
                                                               thread_copy_fwd_step_m_k);
                });

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) { // input add loop
                    static_for<0, XSrcVectorSize, 1>{}([&](auto iK1) {
                        constexpr auto offset_m_k =
                            thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK1));

                        // get reference to in data
                        const auto in_data_refs = generate_tie(
                            // return type should be lvalue
                            [&](auto I) -> const auto& {
                                return in_thread_buf_tuple(iK0)(I)(Number<offset_m_k>{});
                            },
                            Number<NumInput>{});

                        // get reference to dst data
                        auto out_data_refs = generate_tie(
                            // return type should be lvalue
                            [&](auto) -> auto& { return x_thread_buf(iK0)(Number<offset_m_k>{}); },
                            I1);

                        unpack2(x_elementwise_op, out_data_refs, in_data_refs);
                    });
                });
                threadwise_welford.Run(x_thread_buf[iK0], mean_thread_buf, var_thread_buf);

                if constexpr(!SweepOnce)
                {
                    threadwise_x_store.Run(thread_buffer_desc_m_k,
                                           make_tuple(I0, I0),
                                           x_thread_buf(iK0),
                                           x_grid_desc_m_k,
                                           x_lds_val_buf);
                    threadwise_x_store.MoveDstSliceWindow(x_grid_desc_m_k,
                                                          thread_copy_fwd_step_m_k);
                }
            });
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            int count = threadwise_welford.cur_count_;
            BlockwiseWelford::Run(mean_thread_buf(I), var_thread_buf(I), count);
        });

        auto thread_copy_tail_m_k =
            (num_k_block_tile_iteration - 1) * XThreadBufferNumber * thread_copy_fwd_step_m_k;

        if constexpr(!SweepOnce)
            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k, thread_copy_tail_m_k);
        threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_tail_m_k);

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {
            if constexpr(!SweepOnce)
            {
                static_for<0, XThreadBufferNumber, 1>{}([&](auto i) {
                    threadwise_x_load.Run(x_grid_desc_m_k,
                                          x_lds_val_buf,
                                          thread_buffer_desc_m_k,
                                          make_tuple(I0, I0),
                                          x_thread_buf(i));
                    threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);
                });
            }

            static_for<0, GammaThreadBufferNumber, 1>{}([&](auto i) {
                threadwise_gamma_load.Run(gamma_grid_desc_m_k,
                                          gamma_global_val_buf,
                                          thread_buffer_desc_m_k,
                                          make_tuple(I0, I0),
                                          gamma_thread_buf(i));
                threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k,
                                                         thread_copy_fwd_step_m_k);
            });

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                auto divisor = 1 / __builtin_amdgcn_sqrtf(var_thread_buf(iM) + epsilon);
                static_for<0, XThreadBufferNumber, 1>{}([&](auto iK0) {
                    static_for<0, XSrcVectorSize, 1>{}([&](auto iK1) {
                        constexpr auto offset_m_k =
                            thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK1));

                        // normalize
                        y_thread_buf(iK0)(Number<offset_m_k>{}) =
                            (x_thread_buf(iK0)(Number<offset_m_k>{}) - mean_thread_buf(iM)) *
                            divisor;

                        // gamma
                        y_thread_buf(iK0)(Number<offset_m_k>{}) =
                            y_thread_buf(iK0)(Number<offset_m_k>{}) *
                            gamma_thread_buf(iK0)(Number<offset_m_k>{});
                    });
                });
            });

            static_for<0, BetaThreadBufferNumber, 1>{}([&](auto i) {
                threadwise_beta_load.Run(beta_grid_desc_m_k,
                                         beta_global_val_buf,
                                         thread_buffer_desc_m_k,
                                         make_tuple(I0, I0),
                                         beta_thread_buf(i));
                threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k,
                                                        thread_copy_fwd_step_m_k);
            });

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, XThreadBufferNumber, 1>{}([&](auto iK0) {
                    static_for<0, XSrcVectorSize, 1>{}([&](auto iK1) {
                        constexpr auto offset_m_k =
                            thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK1));

                        // beta
                        y_thread_buf(iK0)(Number<offset_m_k>{}) =
                            y_thread_buf(iK0)(Number<offset_m_k>{}) +
                            beta_thread_buf(iK0)(Number<offset_m_k>{});
                    });
                });
            });

            static_for<0, YThreadBufferNumber, 1>{}([&](auto i) {
                threadwise_y_store.Run(thread_buffer_desc_m_k,
                                       make_tuple(I0, I0),
                                       y_thread_buf(i),
                                       y_grid_desc_m_k,
                                       y_global_val_buf);
                threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_fwd_step_m_k);
            });

            if constexpr(!SweepOnce)
                threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, 2 * thread_copy_bwd_step_m_k);
            threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k,
                                                     2 * thread_copy_bwd_step_m_k);
            threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k,
                                                    2 * thread_copy_bwd_step_m_k);
            threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, 2 * thread_copy_bwd_step_m_k);
        }
    }
};

} // namespace ck
