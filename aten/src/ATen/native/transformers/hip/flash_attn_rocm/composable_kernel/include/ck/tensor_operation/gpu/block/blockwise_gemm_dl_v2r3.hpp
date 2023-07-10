// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_contraction_dl.hpp"

namespace ck {

// C[BM0, BM1, BN0, BN1] += transpose(A[K, BM0, BM1]) * B[K, BN0, BN1]
// A and B are visable to the whole block, C is distributed among each thread
// Assume:
//   1. A:
//     1. ABlockDesc_BK0_BM_BK1 is known at compile-time
//     2. ABlockBuffer is DynamicBuffer
//   2. B:
//     1. BBlockDesc_BK0_BN_BK1 is known at compile-time
//     2. BBlockBuffer is DynamicBuffer
//   3. C:
//     1. CThreadDesc_BM0_BM11_BN0_BN11 is known at compile-time
//     2. CThreadBuffer is StaticBuffer
// Also assume:
//   BM10BN10ThreadClusterBM10Xs::Size() = BM10BN10ThreadClusterBN10Xs::Size() == 2
//   BM0 = BN0 = 2. It will do 2x2 pipelined read and fma (ABBA optimization)
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatC,
          typename ABlockDesc_BK0_BM_BK1,
          typename BBlockDesc_BK0_BN_BK1,
          index_t BM1PerThreadBM11,
          index_t BN1PerThreadBN11,
          index_t BK0PerThread,
          typename BM10BN10ThreadClusterBM10Xs, // Sequence<BM10BN10ThreadClusterBM100,
                                                //          BM10BN10ThreadClusterBM101, ...>
          typename BM10BN10ThreadClusterBN10Xs, // Sequence<BM10BN10ThreadClusterBN100,
                                                //          BM10BN10ThreadClusterBN101, ...>
          index_t AThreadCopyScalarPerVector_BM11,
          index_t BThreadCopyScalarPerVector_BN11,
          typename enable_if<ABlockDesc_BK0_BM_BK1::IsKnownAtCompileTime() &&
                                 BBlockDesc_BK0_BN_BK1::IsKnownAtCompileTime(),
                             bool>::type = false>
struct BlockwiseGemmDl_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_pipeline_BM0_2_BN0_2
{
    using AIndex = MultiIndex<3>;
    using BIndex = MultiIndex<3>;
    using CIndex = MultiIndex<4>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr index_t BK0 = ABlockDesc_BK0_BM_BK1{}.GetLength(I0);
    static constexpr index_t BK1 = ABlockDesc_BK0_BM_BK1{}.GetLength(I2);
    static constexpr index_t BM  = ABlockDesc_BK0_BM_BK1{}.GetLength(I1);
    static constexpr index_t BN  = BBlockDesc_BK0_BN_BK1{}.GetLength(I1);

    static constexpr index_t BM100 = BM10BN10ThreadClusterBM10Xs{}[I0];
    static constexpr index_t BN100 = BM10BN10ThreadClusterBN10Xs{}[I0];

    static constexpr index_t BM101 = BM10BN10ThreadClusterBM10Xs{}[I1];
    static constexpr index_t BN101 = BM10BN10ThreadClusterBN10Xs{}[I1];

    static constexpr index_t BM11 = BM1PerThreadBM11;
    static constexpr index_t BN11 = BN1PerThreadBN11;

    static constexpr index_t BM1 = BM100 * BM101 * BM11;
    static constexpr index_t BN1 = BN100 * BN101 * BN11;

    static constexpr index_t BM0 = BM / BM1;
    static constexpr index_t BN0 = BN / BN1;

    __host__ __device__ static constexpr auto
    MakeABlockDescriptor_BK0_BM0_BM1_BK1(const ABlockDesc_BK0_BM_BK1& a_block_desc_bk0_bm_bk1)
    {
        const auto a_block_bk0_bm0_bm1_bk1 = transform_tensor_descriptor(
            a_block_desc_bk0_bm_bk1,
            make_tuple(make_pass_through_transform(Number<BK0>{}),
                       make_unmerge_transform(make_tuple(Number<BM0>{}, Number<BM1>{})),
                       make_pass_through_transform(Number<BK1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return a_block_bk0_bm0_bm1_bk1;
    }

    __host__ __device__ static constexpr auto
    MakeBBlockDescriptor_BK0_BN0_BN1_BK1(const BBlockDesc_BK0_BN_BK1& b_block_desc_bk0_bn_bk1)
    {
        const auto b_block_desc_bk0_bn0_bn1_bk1 = transform_tensor_descriptor(
            b_block_desc_bk0_bn_bk1,
            make_tuple(make_pass_through_transform(Number<BK0>{}),
                       make_unmerge_transform(make_tuple(Number<BN0>{}, Number<BN1>{})),
                       make_pass_through_transform(Number<BK1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        return b_block_desc_bk0_bn0_bn1_bk1;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM_BN()
    {
        // upper: [BM0, BM100, BM101, BM11, BN0, BN100, BN101, BN11]
        // lower: [BM, BN]
        constexpr auto c_block_adaptor_m0_m100_m101_m11_n0_n100_n101_n11_to_m_n =
            make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(
                               Number<BM0>{}, Number<BM100>{}, Number<BM101>{}, Number<BM11>{})),
                           make_unmerge_transform(make_tuple(
                               Number<BN0>{}, Number<BN100>{}, Number<BN101>{}, Number<BN11>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4, 5, 6, 7>{}));

        return c_block_adaptor_m0_m100_m101_m11_n0_n100_n101_n11_to_m_n;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM0_BM1_BN0_BN1()
    {
        // upper: [BM0, BM100, BM101, BM11, BN0, BN100, BN101, BN11]
        // lower: [BM0, BM1, BN0, BN1]
        constexpr auto c_block_adaptor_m0_m100_m101_m11_n0_n100_n101_n11_to_m0_m1_n0_n1 =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(Number<BM0>{}),
                           make_unmerge_transform(
                               make_tuple(Number<BM100>{}, Number<BM101>{}, Number<BM11>{})),
                           make_pass_through_transform(Number<BN0>{}),
                           make_unmerge_transform(
                               make_tuple(Number<BN100>{}, Number<BN101>{}, Number<BN11>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}, Sequence<5, 6, 7>{}));

        return c_block_adaptor_m0_m100_m101_m11_n0_n100_n101_n11_to_m0_m1_n0_n1;
    }

    __host__ __device__ static constexpr auto GetCThreadTensorLengths_BM0_BM1_BN0_BN1()
    {
        return Sequence<BM0, BM11, BN0, BN11>{};
    }

    static constexpr auto a_block_desc_bk0_bm0_bm1_bk1_ =
        MakeABlockDescriptor_BK0_BM0_BM1_BK1(ABlockDesc_BK0_BM_BK1{});

    static constexpr auto b_block_desc_bk0_bn0_bn1_bk1_ =
        MakeBBlockDescriptor_BK0_BN0_BN1_BK1(BBlockDesc_BK0_BN_BK1{});

    public:
    __device__ BlockwiseGemmDl_A_BK0_BM_BK1_B_BK0_BN_BK1_C_BM0_BM1_BN0_BN1_pipeline_BM0_2_BN0_2()
        : c_thread_origin_data_idx_{CalculateCThreadOriginOnBlock_BM0_BM1_BN0_BN1(
              get_thread_local_1d_id())},
          a_thread_copy_{
              make_tuple(0, c_thread_origin_data_idx_[I0], c_thread_origin_data_idx_[I1], 0)},
          b_thread_copy_{
              make_tuple(0, c_thread_origin_data_idx_[I2], c_thread_origin_data_idx_[I3], 0)}
    {
        static_assert(ABlockDesc_BK0_BM_BK1::IsKnownAtCompileTime() &&
                          BBlockDesc_BK0_BN_BK1::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(BlockSize == BM101 * BM100 * BN101 * BN100,
                      "wrong! blocksize and cluster size not consistent");

        static_assert(BM % BM1 == 0 && BN % BN1 == 0, "wrong!");

        static_assert(ABlockDesc_BK0_BM_BK1{}.GetLength(I0) ==
                          BBlockDesc_BK0_BN_BK1{}.GetLength(I0),
                      "wrong! K dimension not consistent");

        // TODO remove this restriction
        static_assert(BM10BN10ThreadClusterBM10Xs::Size() == 2 &&
                          BM10BN10ThreadClusterBN10Xs::Size() == 2,
                      "wrong!");

        // TODO: remove this restriction
        static_assert(BM0 == 2, "wrong");
        static_assert(BM0 == 2 && BN0 == 2, "wrong");
    }

    __device__ static CIndex CalculateCThreadOriginOnBlock_BM0_BM1_BN0_BN1(index_t thread_id)
    {
        // lower: [BM0, BM1, BN0, BN1]
        // upper: [BM0, BM100, BM101, BM11, BN0, BN100, BN101, BN11]
        constexpr auto adaptor0 =
            MakeCBlockAdaptor_BM0_BM100_BM101_BM11_BN0_BN100_BN101_BN11_To_BM0_BM1_BN0_BN1();

        // lower: [BM0, BM100, BM101, BM11, BN0, BN100, BN101, BN11]
        // upper: [Tid, BM0, BM11, BN0, BN11]
        constexpr auto adaptor1 = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(BM100, BN100, BM101, BN101)),
                       make_pass_through_transform(BM0),
                       make_pass_through_transform(BM11),
                       make_pass_through_transform(BN0),
                       make_pass_through_transform(BN11)),
            make_tuple(
                Sequence<1, 5, 2, 6>{}, Sequence<0>{}, Sequence<3>{}, Sequence<4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        constexpr auto adaptor = chain_tensor_adaptors(adaptor0, adaptor1);

        return adaptor.CalculateBottomIndex(make_multi_index(thread_id, 0, 0, 0, 0));
    }

    template <typename CThreadDesc_BM0_BM11_BN0_BN11,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename CThreadBuffer>
    __device__ void Run(const CThreadDesc_BM0_BM11_BN0_BN11&,
                        const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        static_assert(CThreadDesc_BM0_BM11_BN0_BN11::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        // TODO: remove this restriction
        static_assert(BM0 == 2 && BN0 == 2 &&
                          CThreadDesc_BM0_BM11_BN0_BN11{}.GetLength(I0) == BM0 &&
                          CThreadDesc_BM0_BM11_BN0_BN11{}.GetLength(I2) == BN0,
                      "wrong");

        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(
            a_thread_desc_bk0_bm0_bm1_bk1_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatB>(
            b_thread_desc_bk0_bn0_bn1_bk1_.GetElementSpaceSize());

        constexpr auto threadwise_contraction =
            ThreadwiseContractionDl_A_TK0_TM0_TM1_TK1_B_TK0_TN0_TN1_TK1_C_TM0_TM1_TN0_TN1<
                FloatA,
                FloatB,
                FloatC,
                decltype(a_thread_desc_bk0_bm0_bm1_bk1_),
                decltype(b_thread_desc_bk0_bn0_bn1_bk1_),
                CThreadDesc_BM0_BM11_BN0_BN11,
                Sequence<BK0PerThread, BK1>,
                Sequence<1, BM1PerThreadBM11>,
                Sequence<1, BN1PerThreadBN11>>{};

        // read A_sub_0
        a_thread_copy_.Run(a_block_desc_bk0_bm0_bm1_bk1_,
                           make_tuple(I0, I0, I0, I0),
                           a_block_buf,
                           a_thread_desc_bk0_bm0_bm1_bk1_,
                           make_tuple(I0, I0, I0, I0),
                           a_thread_buf);

        // read B_sub_0
        b_thread_copy_.Run(b_block_desc_bk0_bn0_bn1_bk1_,
                           make_tuple(I0, I0, I0, I0),
                           b_block_buf,
                           b_thread_desc_bk0_bn0_bn1_bk1_,
                           make_tuple(I0, I0, I0, I0),
                           b_thread_buf);

        // read B_sub_1
        b_thread_copy_.Run(b_block_desc_bk0_bn0_bn1_bk1_,
                           make_tuple(I0, I1, I0, I0),
                           b_block_buf,
                           b_thread_desc_bk0_bn0_bn1_bk1_,
                           make_tuple(I0, I1, I0, I0),
                           b_thread_buf);

        // read A_sub_1
        a_thread_copy_.Run(a_block_desc_bk0_bm0_bm1_bk1_,
                           make_tuple(I0, I1, I0, I0),
                           a_block_buf,
                           a_thread_desc_bk0_bm0_bm1_bk1_,
                           make_tuple(I0, I1, I0, I0),
                           a_thread_buf);

        // C_sub_00 += transpose(A_sub_0) * B_sub_0
        threadwise_contraction.Run(a_thread_buf,
                                   make_tuple(I0, I0, I0, I0),
                                   b_thread_buf,
                                   make_tuple(I0, I0, I0, I0),
                                   c_thread_buf,
                                   make_tuple(I0, I0, I0, I0));

        // C_sub_01 += transpose(A_sub_0) * B_sub_1
        threadwise_contraction.Run(a_thread_buf,
                                   make_tuple(I0, I0, I0, I0),
                                   b_thread_buf,
                                   make_tuple(I0, I1, I0, I0),
                                   c_thread_buf,
                                   make_tuple(I0, I0, I1, I0));

        // loop over rest of bk0
        static_for<BK0PerThread, BK0, BK0PerThread>{}([&](auto bk0) {
            // read A_sub_0
            a_thread_copy_.Run(a_block_desc_bk0_bm0_bm1_bk1_,
                               make_tuple(bk0, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_bk0_bm0_bm1_bk1_,
                               make_tuple(I0, I0, I0, I0),
                               a_thread_buf);

            // C_sub_10 += transpose(A_sub_1) * B_sub_0
            threadwise_contraction.Run(a_thread_buf,
                                       make_tuple(I0, I1, I0, I0),
                                       b_thread_buf,
                                       make_tuple(I0, I0, I0, I0),
                                       c_thread_buf,
                                       make_tuple(I1, I0, I0, I0));

            // read B_sub_0
            b_thread_copy_.Run(b_block_desc_bk0_bn0_bn1_bk1_,
                               make_tuple(bk0, I0, I0, I0),
                               b_block_buf,
                               b_thread_desc_bk0_bn0_bn1_bk1_,
                               make_tuple(I0, I0, I0, I0),
                               b_thread_buf);

            // C_sub_11 += transpose(A_sub_1) * B_sub_1
            threadwise_contraction.Run(a_thread_buf,
                                       make_tuple(I0, I1, I0, I0),
                                       b_thread_buf,
                                       make_tuple(I0, I1, I0, I0),
                                       c_thread_buf,
                                       make_tuple(I1, I0, I1, I0));

            // read B_sub_1
            b_thread_copy_.Run(b_block_desc_bk0_bn0_bn1_bk1_,
                               make_tuple(bk0, I1, I0, I0),
                               b_block_buf,
                               b_thread_desc_bk0_bn0_bn1_bk1_,
                               make_tuple(I0, I1, I0, I0),
                               b_thread_buf);

            // read A_sub_1
            a_thread_copy_.Run(a_block_desc_bk0_bm0_bm1_bk1_,
                               make_tuple(bk0, I1, I0, I0),
                               a_block_buf,
                               a_thread_desc_bk0_bm0_bm1_bk1_,
                               make_tuple(I0, I1, I0, I0),
                               a_thread_buf);

            // C_sub_00 += transpose(A_sub_0) * B_sub_0
            threadwise_contraction.Run(a_thread_buf,
                                       make_tuple(I0, I0, I0, I0),
                                       b_thread_buf,
                                       make_tuple(I0, I0, I0, I0),
                                       c_thread_buf,
                                       make_tuple(I0, I0, I0, I0));

            // C_sub_01 += transpose(A_sub_0) * B_sub_1
            threadwise_contraction.Run(a_thread_buf,
                                       make_tuple(I0, I0, I0, I0),
                                       b_thread_buf,
                                       make_tuple(I0, I1, I0, I0),
                                       c_thread_buf,
                                       make_tuple(I0, I0, I1, I0));
        });

        // C_sub_10 += transpose(A_sub_1) * B_sub_0
        threadwise_contraction.Run(a_thread_buf,
                                   make_tuple(I0, I1, I0, I0),
                                   b_thread_buf,
                                   make_tuple(I0, I0, I0, I0),
                                   c_thread_buf,
                                   make_tuple(I1, I0, I0, I0));

        // C_sub_11 += transpose(A_sub_1) * B_sub_1
        threadwise_contraction.Run(a_thread_buf,
                                   make_tuple(I0, I1, I0, I0),
                                   b_thread_buf,
                                   make_tuple(I0, I1, I0, I0),
                                   c_thread_buf,
                                   make_tuple(I1, I0, I1, I0));
    }

    private:
    // A[BK0, BM0, BM1, BK1]
    static constexpr auto a_thread_desc_bk0_bm0_bm1_bk1_ =
        make_naive_tensor_descriptor_packed(make_tuple(
            Number<BK0PerThread>{}, Number<BM0>{}, Number<BM1PerThreadBM11>{}, Number<BK1>{}));

    // B[BK0, BN0, BN1, BK1]
    static constexpr auto b_thread_desc_bk0_bn0_bn1_bk1_ =
        make_naive_tensor_descriptor_packed(make_tuple(
            Number<BK0PerThread>{}, Number<BN0>{}, Number<BN1PerThreadBN11>{}, Number<BK1>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4r1<
        FloatA,
        FloatA,
        decltype(a_block_desc_bk0_bm0_bm1_bk1_),
        decltype(a_thread_desc_bk0_bm0_bm1_bk1_),
        Sequence<BK0PerThread, 1, BM1PerThreadBM11, BK1>, // SliceLengths
        Sequence<0, 1, 2, 3>,                             // DimAccessOrder
        Sequence<1, 1, BM1PerThreadBM11, BK1>,            // SrcVectorTensorLengths
        Sequence<0, 1, 2, 3>>;                            // SrcVectorTensorContiguousDimOrder

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4r1<
        FloatB,
        FloatB,
        decltype(b_block_desc_bk0_bn0_bn1_bk1_),
        decltype(b_thread_desc_bk0_bn0_bn1_bk1_),
        Sequence<BK0PerThread, 1, BN1PerThreadBN11, BK1>, // SliceLengths
        Sequence<0, 1, 2, 3>,                             // DimAccessOrder
        Sequence<1, 1, BN1PerThreadBN11, BK1>,            // SrcVectorTensorLengths
        Sequence<0, 1, 2, 3>>;                            // SrcVectorTensorContiguousDimOrder

    CIndex c_thread_origin_data_idx_;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
