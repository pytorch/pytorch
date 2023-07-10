// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"

namespace ck {

template <typename GridwiseSparseEmbedding,
          typename EmbType,
          typename IndexType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename OutType,
          typename OutGridDesc>
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    __global__ void kernel_sparse_embedding3_forward_layernorm(OutType* p_out,
                                                               const EmbType* p_emb_a,
                                                               const EmbType* p_emb_b,
                                                               const EmbType* p_emb_c,
                                                               const IndexType* p_index_a,
                                                               const IndexType* p_index_b,
                                                               const IndexType* p_index_c,
                                                               const GammaDataType* p_gamma,
                                                               const BetaDataType* p_beta,
                                                               const OutGridDesc out_grid_desc,
                                                               const AccDataType epsilon)
{
    GridwiseSparseEmbedding::Run(p_out,
                                 p_emb_a,
                                 p_emb_b,
                                 p_emb_c,
                                 p_index_a,
                                 p_index_b,
                                 p_index_c,
                                 p_gamma,
                                 p_beta,
                                 out_grid_desc,
                                 epsilon);
}

template <typename EmbType,
          typename IndexType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename OutType,
          typename OutGridDesc,
          ck::index_t BlockSize,
          ck::index_t DimClusterSize,
          ck::index_t RowClusterSize,
          ck::index_t DimPerBlock,   // Row x Dim, along Dim
          ck::index_t RowPerBlock,   // Row x Dim, along Row
          ck::index_t DimThreadSize, // this is actually not vector, but number of registers
          ck::index_t RowVectorSize>
struct GridwiseSparseEmbedding3ForwardLayernorm
{
    static constexpr auto I0          = Number<0>{};
    static constexpr auto I1          = Number<1>{};
    static constexpr auto I2          = Number<2>{};
    static constexpr auto I3          = Number<3>{};
    static constexpr index_t WaveSize = 64;

    static_assert(BlockSize == RowClusterSize * DimClusterSize,
                  "Invalid cluster distribution within block");
    static_assert(RowClusterSize % WaveSize == 0, "need to be wavewise");

    static_assert(DimPerBlock % (DimClusterSize * DimThreadSize) == 0, "");
    static_assert(RowPerBlock % (RowClusterSize * RowVectorSize) == 0, "");

    static constexpr auto DimSubBlocks = DimPerBlock / (DimClusterSize * DimThreadSize);
    static constexpr auto RowSubBlocks = RowPerBlock / (RowClusterSize * RowVectorSize);

    static_assert((DimPerBlock % DimSubBlocks == 0) && (RowPerBlock % RowSubBlocks == 0), "");
    static constexpr auto DimPerSubBlock = DimPerBlock / DimSubBlocks;
    static constexpr auto RowPerSubBlock = RowPerBlock / RowSubBlocks;

    using ThreadwiseWolfordDesc2D = decltype(make_naive_tensor_descriptor_packed(make_tuple(
        Number<DimSubBlocks * DimThreadSize>{}, Number<RowSubBlocks * RowVectorSize>{})));

    using ThreadwiseWolfordDescReduce = decltype(
        make_naive_tensor_descriptor_packed(make_tuple(Number<DimSubBlocks * DimThreadSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelford<AccDataType, ThreadwiseWolfordDesc2D, ThreadwiseWolfordDescReduce>;

    using ThreadClusterLength = Sequence<DimClusterSize, RowClusterSize>;

    using BlockwiseWelford =
        BlockwiseWelford<AccDataType, BlockSize, ThreadClusterLength, Sequence<0, 1>>;

    __device__ static void Run(OutType* p_out,
                               const EmbType* p_emb_a,
                               const EmbType* p_emb_b,
                               const EmbType* p_emb_c,
                               const IndexType* p_index_a,
                               const IndexType* p_index_b,
                               const IndexType* p_index_c,
                               const GammaDataType* p_gamma,
                               const BetaDataType* p_beta,
                               const OutGridDesc,
                               const AccDataType epsilon)
    {
        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        // const auto index_length = out_grid_desc.GetLength(I0);
        // const auto emb_dim      = out_grid_desc.GetLength(I1);

        constexpr auto thread_cluster_desc =
            make_cluster_descriptor(Sequence<DimClusterSize, RowClusterSize>{}, Sequence<0, 1>{});

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_dim_cluster_id = thread_cluster_idx[I0];
        const auto thread_row_cluster_id = thread_cluster_idx[I1];

        const auto wave_dim_id = __builtin_amdgcn_readfirstlane(thread_dim_cluster_id / WaveSize);

        const auto index_start = block_global_id * DimPerBlock + wave_dim_id * DimThreadSize;

        auto threadwise_welford       = ThreadwiseWelford();
        threadwise_welford.max_count_ = RowSubBlocks * RowVectorSize;

        constexpr auto thread_buf_size =
            DimSubBlocks * DimThreadSize * RowSubBlocks * RowVectorSize;
        constexpr auto thread_buf_desc = make_naive_tensor_descriptor_packed(
            make_tuple(DimSubBlocks, DimThreadSize, RowSubBlocks, RowVectorSize));
        constexpr auto mean_var_buf_size = DimSubBlocks * DimThreadSize;
        constexpr auto mean_var_buf_desc =
            make_naive_tensor_descriptor_packed(make_tuple(DimSubBlocks, DimThreadSize));
        constexpr auto gamma_beta_buf_size = RowSubBlocks * RowVectorSize;
        constexpr auto gamma_beta_buf_desc =
            make_naive_tensor_descriptor_packed(make_tuple(RowSubBlocks, RowVectorSize));

        StaticBuffer<AddressSpaceEnum::Vgpr, EmbType, thread_buf_size, true> in_thread_buf_a;
        StaticBuffer<AddressSpaceEnum::Vgpr, EmbType, thread_buf_size, true> in_thread_buf_b;
        StaticBuffer<AddressSpaceEnum::Vgpr, EmbType, thread_buf_size, true> in_thread_buf_c;

        StaticBuffer<AddressSpaceEnum::Sgpr, IndexType, DimPerBlock, true> index_buf_a;
        StaticBuffer<AddressSpaceEnum::Sgpr, IndexType, DimPerBlock, true> index_buf_b;
        StaticBuffer<AddressSpaceEnum::Sgpr, IndexType, DimPerBlock, true> index_buf_c;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, thread_buf_size, true> acc_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, gamma_beta_buf_size, true>
            gamma_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, gamma_beta_buf_size, true>
            beta_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, mean_var_buf_size, true> mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, mean_var_buf_size, true> var_thread_buf;

        auto load_current_sub_row = [&](auto i_dim_sub_, auto i_row_sub_) {
            vector_type_maker_t<EmbType, RowVectorSize> emb_vector_a;
            vector_type_maker_t<EmbType, RowVectorSize> emb_vector_b;
            vector_type_maker_t<EmbType, RowVectorSize> emb_vector_c;

            using src_vector_t = typename decltype(emb_vector_a)::type;
            static_for<0, DimThreadSize, 1>{}([&](auto i_dim_vec_) {
                constexpr auto current_dim = i_dim_sub_ * DimPerSubBlock + i_dim_vec_;
                IndexType index_a          = index_buf_a[Number<current_dim>{}];
                IndexType index_b          = index_buf_b[Number<current_dim>{}];
                IndexType index_c          = index_buf_c[Number<current_dim>{}];

                auto thread_offset = (thread_row_cluster_id + i_row_sub_ * RowClusterSize) *
                                     sizeof(EmbType) * RowVectorSize;

                int32x4_t emb_res_a =
                    make_wave_buffer_resource_with_default_range(p_emb_a + index_a * RowPerBlock);
                int32x4_t emb_res_b =
                    make_wave_buffer_resource_with_default_range(p_emb_b + index_b * RowPerBlock);
                int32x4_t emb_res_c =
                    make_wave_buffer_resource_with_default_range(p_emb_c + index_c * RowPerBlock);
                emb_vector_a.template AsType<src_vector_t>()(I0) =
                    amd_buffer_load_impl<EmbType, RowVectorSize>(emb_res_a, thread_offset, 0);
                emb_vector_b.template AsType<src_vector_t>()(I0) =
                    amd_buffer_load_impl<EmbType, RowVectorSize>(emb_res_b, thread_offset, 0);
                emb_vector_c.template AsType<src_vector_t>()(I0) =
                    amd_buffer_load_impl<EmbType, RowVectorSize>(emb_res_c, thread_offset, 0);

                static_for<0, RowVectorSize, 1>{}([&](auto i_row_vec_) {
                    constexpr auto register_offset = thread_buf_desc.CalculateOffset(
                        make_tuple(i_dim_sub_, i_dim_vec_, i_row_sub_, i_row_vec_));
                    in_thread_buf_a(Number<register_offset>{}) =
                        emb_vector_a.template AsType<EmbType>()[i_row_vec_];
                    in_thread_buf_b(Number<register_offset>{}) =
                        emb_vector_b.template AsType<EmbType>()[i_row_vec_];
                    in_thread_buf_c(Number<register_offset>{}) =
                        emb_vector_c.template AsType<EmbType>()[i_row_vec_];
                });
            });
        };

        auto accumulate_current_sub_row = [&](auto i_dim_sub_, auto i_row_sub_) {
            static_for<0, DimThreadSize, 1>{}([&](auto i_dim_vec_) {
                static_for<0, RowVectorSize, 1>{}([&](auto i_row_vec_) {
                    constexpr auto register_offset = thread_buf_desc.CalculateOffset(
                        make_tuple(i_dim_sub_, i_dim_vec_, i_row_sub_, i_row_vec_));
                    AccDataType va =
                        ck::type_convert<AccDataType>(in_thread_buf_a(Number<register_offset>{}));
                    AccDataType vb =
                        ck::type_convert<AccDataType>(in_thread_buf_b(Number<register_offset>{}));
                    AccDataType vc =
                        ck::type_convert<AccDataType>(in_thread_buf_c(Number<register_offset>{}));

                    acc_thread_buf(Number<register_offset>{}) += va + vb + vc;
                });
            });
        };

        auto threadwise_welford_sub_row = [&](auto i_dim_sub_, auto i_row_sub_) {
            static_for<0, DimThreadSize, 1>{}([&](auto i_dim_vec_) {
                static_for<0, RowVectorSize, 1>{}([&](auto i_row_vec_) {
                    constexpr auto register_offset = thread_buf_desc.CalculateOffset(
                        make_tuple(i_dim_sub_, i_dim_vec_, i_row_sub_, i_row_vec_));
                    constexpr auto mean_var_offset =
                        mean_var_buf_desc.CalculateOffset(make_tuple(i_dim_sub_, i_dim_vec_));

                    threadwise_welford.cur_count_++;
                    threadwise_welford.Update(mean_thread_buf(Number<mean_var_offset>{}),
                                              var_thread_buf(Number<mean_var_offset>{}),
                                              acc_thread_buf(Number<register_offset>{}));
                });
            });
        };

        auto threadwise_normalize_store_out = [&](auto i_dim_sub_, auto i_row_sub_) {
            int32x4_t out_res =
                make_wave_buffer_resource_with_default_range(p_out + index_start * RowPerBlock);
            static_for<0, DimThreadSize, 1>{}([&](auto i_dim_vec_) {
                vector_type_maker_t<OutType, RowVectorSize> out_vector;
                using dst_vector_t = typename decltype(out_vector)::type;

                constexpr auto mean_var_offset =
                    mean_var_buf_desc.CalculateOffset(make_tuple(i_dim_sub_, i_dim_vec_));

                static_for<0, RowVectorSize, 1>{}([&](auto i_row_vec_) {
                    constexpr auto register_offset = thread_buf_desc.CalculateOffset(
                        make_tuple(i_dim_sub_, i_dim_vec_, i_row_sub_, i_row_vec_));
                    constexpr auto gamma_beta_offset =
                        gamma_beta_buf_desc.CalculateOffset(make_tuple(i_row_sub_, i_row_vec_));

                    auto acc_val = acc_thread_buf[Number<register_offset>{}];
                    acc_val      = (acc_val - mean_thread_buf(Number<mean_var_offset>{})) /
                              sqrt(var_thread_buf(Number<mean_var_offset>{}) + epsilon);
                    acc_val = acc_val * gamma_thread_buf[Number<gamma_beta_offset>{}] +
                              beta_thread_buf[Number<gamma_beta_offset>{}];

                    out_vector.template AsType<OutType>()(Number<i_row_vec_>{}) =
                        type_convert<OutType>(acc_val);
                });

                index_t thread_offset = (thread_row_cluster_id + i_row_sub_ * RowClusterSize) *
                                        sizeof(OutType) * RowVectorSize;

                amd_buffer_store_impl<OutType, RowVectorSize>(
                    out_vector.template AsType<dst_vector_t>()[Number<0>{}],
                    out_res,
                    thread_offset,
                    0);
            });
        };

        // first load index
        ck::static_for<0, DimPerBlock, 1>{}([&](auto i_idx_) {
            // prefer use s_load
            index_buf_a(i_idx_) = p_index_a[index_start + i_idx_.value];
            index_buf_b(i_idx_) = p_index_b[index_start + i_idx_.value];
            index_buf_c(i_idx_) = p_index_c[index_start + i_idx_.value];
        });

        // load gamma/beta
        static_for<0, RowSubBlocks, 1>{}([&](auto i_row_sub_) {
            vector_type_maker_t<GammaDataType, RowVectorSize> gamma_vector;
            vector_type_maker_t<BetaDataType, RowVectorSize> beta_vector;

            index_t thread_offset_gamma = (thread_row_cluster_id + i_row_sub_ * RowClusterSize) *
                                          sizeof(GammaDataType) * RowVectorSize;
            index_t thread_offset_beta = (thread_row_cluster_id + i_row_sub_ * RowClusterSize) *
                                         sizeof(BetaDataType) * RowVectorSize;

            int32x4_t gamma_res = make_wave_buffer_resource_with_default_range(p_gamma);
            int32x4_t beta_res  = make_wave_buffer_resource_with_default_range(p_beta);

            gamma_vector.template AsType<typename decltype(gamma_vector)::type>()(I0) =
                amd_buffer_load_impl<GammaDataType, RowVectorSize>(
                    gamma_res, thread_offset_gamma, 0);
            beta_vector.template AsType<typename decltype(beta_vector)::type>()(I0) =
                amd_buffer_load_impl<BetaDataType, RowVectorSize>(beta_res, thread_offset_beta, 0);

            static_for<0, RowVectorSize, 1>{}([&](auto i_row_vec_) {
                constexpr auto offset =
                    gamma_beta_buf_desc.CalculateOffset(make_tuple(i_row_sub_, i_row_vec_));
                gamma_thread_buf(Number<offset>{}) = type_convert<AccDataType>(
                    gamma_vector.template AsType<GammaDataType>()[Number<i_row_vec_>{}]);
                beta_thread_buf(Number<offset>{}) = type_convert<AccDataType>(
                    beta_vector.template AsType<BetaDataType>()[Number<i_row_vec_>{}]);
            });
        });

        static_for<0, thread_buf_size, 1>{}(
            [&](auto I) { acc_thread_buf(I) = type_convert<AccDataType>(0.0f); });

        static_for<0, mean_var_buf_size, 1>{}([&](auto I) {
            mean_thread_buf(I) = type_convert<AccDataType>(0.0f);
            var_thread_buf(I)  = type_convert<AccDataType>(0.0f);
        });

        static_for<0, DimSubBlocks, 1>{}([&](auto i_dim_sub) {
            load_current_sub_row(i_dim_sub, Number<0>{});
            static_for<0, RowSubBlocks - 1, 1>{}([&](auto i_row) {
                load_current_sub_row(i_dim_sub, Number<1>{} + i_row);
                accumulate_current_sub_row(i_dim_sub, i_row);
                threadwise_welford_sub_row(i_dim_sub, i_row);
            });
            accumulate_current_sub_row(i_dim_sub, Number<RowSubBlocks - 1>{});
            threadwise_welford_sub_row(i_dim_sub, Number<RowSubBlocks - 1>{});

            // blockwise welford
            static_for<0, mean_var_buf_size, 1>{}([&](auto I) {
                if constexpr(I > 0)
                    block_sync_lds();

                BlockwiseWelford::Run(
                    mean_thread_buf(I), var_thread_buf(I), threadwise_welford.cur_count_);
            });

            // store
            static_for<0, RowSubBlocks, 1>{}(
                [&](auto i_row) { threadwise_normalize_store_out(i_dim_sub, i_row); });
        });
    }
};

} // namespace ck
