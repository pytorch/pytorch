// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseReduction,
          bool OutputIndex,
          bool HaveIndexInput,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InGridDesc_M_K,
          typename OutGridDesc_M,
          typename InElementwiseOperation,
          typename AccElementwiseOperation>
__global__ void kernel_reduce_threadwise(const InGridDesc_M_K in_grid_desc_m_k,
                                         const OutGridDesc_M out_grid_desc_m,
                                         const InElementwiseOperation in_elementwise_op,
                                         const AccElementwiseOperation acc_elementwise_op,
                                         AccDataType alpha,
                                         const InDataType* const __restrict__ p_in_value_global,
                                         const IndexDataType* const __restrict__ p_in_index_global,
                                         AccDataType beta,
                                         OutDataType* const __restrict__ p_out_value_global,
                                         IndexDataType* const __restrict__ p_out_index_global)
{
    if constexpr(!OutputIndex)
    {
        GridwiseReduction::Run(in_grid_desc_m_k,
                               out_grid_desc_m,
                               in_elementwise_op,
                               acc_elementwise_op,
                               alpha,
                               p_in_value_global,
                               beta,
                               p_out_value_global);
    }
    else
    {
        GridwiseReduction::template RunWithIndex<HaveIndexInput>(in_grid_desc_m_k,
                                                                 out_grid_desc_m,
                                                                 in_elementwise_op,
                                                                 acc_elementwise_op,
                                                                 alpha,
                                                                 p_in_value_global,
                                                                 p_in_index_global,
                                                                 beta,
                                                                 p_out_value_global,
                                                                 p_out_index_global);
    };
};

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename IndexDataType,
          typename InGridDesc_M_K,
          typename OutGridDesc_M,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          InMemoryDataOperationEnum OutMemoryDataOperation,
          bool PropagateNan,
          index_t BlockSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct GridwiseReduction_mk_to_m_threadwise
{
    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)) &&
                      (MThreadSliceSize % OutDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    using ThreadBufferDimAccessOrder =
        typename conditional<InSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};

    __device__ static void Run(const InGridDesc_M_K& in_grid_desc_m_k,
                               const OutGridDesc_M& out_grid_desc_m,
                               const InElementwiseOperation& in_elementwise_op,
                               const AccElementwiseOperation& acc_elementwise_op,
                               AccDataType alpha,
                               const InDataType* const __restrict__ p_in_value_global,
                               AccDataType beta,
                               OutDataType* const __restrict__ p_out_value_global)
    {
        using ThreadwiseReduce = ThreadwiseReduction<AccDataType,
                                                     ThreadReduceSrcDesc_M_K,
                                                     ThreadReduceDstDesc_M,
                                                     ReduceOperation,
                                                     PropagateNan>;

        const auto identityVal = ReduceOperation::template GetIdentityValue<AccDataType>();

        const auto in_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_value_global,
            in_grid_desc_m_k.GetElementSpaceSize(),
            ReduceOperation::template GetIdentityValue<InDataType>());
        auto dst_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_value_global, out_grid_desc_m.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> accu_value_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) { accu_value_buf(I) = identityVal; });

        const auto toReduceLength = in_grid_desc_m_k.GetLength(Number<1>{});

        using ThreadBufferLengths         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto thread_buffer_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_val_load =
            ThreadwiseTensorSliceTransfer_v2<InDataType,
                                             AccDataType,
                                             InGridDesc_M_K,
                                             decltype(thread_buffer_desc),
                                             ThreadBufferLengths,
                                             ThreadBufferDimAccessOrder,
                                             InSrcVectorDim,
                                             InSrcVectorSize,
                                             1,
                                             false>(
                in_grid_desc_m_k, make_multi_index(thread_global_1d_id * MThreadSliceSize, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, KThreadSliceSize);

        index_t reducedLength = 0;
        do
        {
            threadwise_src_val_load.Run(in_grid_desc_m_k,
                                        in_global_val_buf,
                                        thread_buffer_desc,
                                        make_tuple(I0, I0),
                                        in_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                // do element-wise pre-reduction operation
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset = thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));
                    in_elementwise_op(in_thread_buf(Number<offset>{}),
                                      in_thread_buf(Number<offset>{}));
                });
            });

            ThreadwiseReduce::Reduce(in_thread_buf, accu_value_buf);

            threadwise_src_val_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

            reducedLength += KThreadSliceSize;
        } while(reducedLength < toReduceLength);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            acc_elementwise_op(accu_value_buf(I), accu_value_buf(I));

            accu_value_buf(I) *= alpha;
        });

        constexpr auto reduced_data_desc = ThreadReduceDstDesc_M{};

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load = ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                                        OutDataType,
                                                                        OutGridDesc_M,
                                                                        decltype(reduced_data_desc),
                                                                        Sequence<MThreadSliceSize>,
                                                                        Sequence<0>,
                                                                        0,
                                                                        1,
                                                                        1,
                                                                        true>(
                out_grid_desc_m, make_multi_index(thread_global_1d_id * MThreadSliceSize));

            StaticBuffer<AddressSpaceEnum::Vgpr, OutDataType, MThreadSliceSize, true>
                priorDstValue_buf;

            threadwise_dst_load.Run(out_grid_desc_m,
                                    dst_global_buf,
                                    reduced_data_desc,
                                    make_tuple(I0),
                                    priorDstValue_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                accu_value_buf(I) += type_convert<AccDataType>(priorDstValue_buf[I]) * beta;
            });
        };

        auto threadwise_dst_store = ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                                       OutDataType,
                                                                       decltype(reduced_data_desc),
                                                                       OutGridDesc_M,
                                                                       PassThroughOp,
                                                                       Sequence<MThreadSliceSize>,
                                                                       Sequence<0>,
                                                                       0,
                                                                       OutDstVectorSize,
                                                                       OutMemoryDataOperation,
                                                                       1,
                                                                       false>(
            out_grid_desc_m,
            make_multi_index(thread_global_1d_id * MThreadSliceSize),
            PassThroughOp{});

        threadwise_dst_store.Run(
            reduced_data_desc, make_tuple(I0), accu_value_buf, out_grid_desc_m, dst_global_buf);
    };

    template <bool HaveIndexInput>
    __device__ static void RunWithIndex(const InGridDesc_M_K& in_grid_desc_m_k,
                                        const OutGridDesc_M& out_grid_desc_m,
                                        const InElementwiseOperation& in_elementwise_op,
                                        const AccElementwiseOperation& acc_elementwise_op,
                                        AccDataType alpha,
                                        const InDataType* const __restrict__ p_in_value_global,
                                        const IndexDataType* const __restrict__ p_in_index_global,
                                        AccDataType beta,
                                        OutDataType* const __restrict__ p_out_value_global,
                                        IndexDataType* const __restrict__ p_out_index_global)
    {
        using ThreadwiseReduceWithIndex = ThreadwiseReductionWithIndex<AccDataType,
                                                                       IndexDataType,
                                                                       ThreadReduceSrcDesc_M_K,
                                                                       ThreadReduceDstDesc_M,
                                                                       ReduceOperation,
                                                                       PropagateNan>;

        (void)acc_elementwise_op;

        const auto identityVal = ReduceOperation::template GetIdentityValue<AccDataType>();

        const auto in_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_value_global,
            in_grid_desc_m_k.GetElementSpaceSize(),
            ReduceOperation::template GetIdentityValue<InDataType>());
        const auto in_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_index_global, in_grid_desc_m_k.GetElementSpaceSize());

        auto out_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_value_global, out_grid_desc_m.GetElementSpaceSize());
        auto out_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_index_global, out_grid_desc_m.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_val_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     IndexDataType,
                     MThreadSliceSize * KThreadSliceSize,
                     true>
            in_thread_idx_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> accu_value_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, IndexDataType, MThreadSliceSize, true> accu_index_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            accu_value_buf(I) = identityVal;
            accu_index_buf(I) = 0;
        });

        const auto toReduceLength = in_grid_desc_m_k.GetLength(Number<1>{});

        using ThreadBufferLengths         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto thread_buffer_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_val_load =
            ThreadwiseTensorSliceTransfer_v2<InDataType,
                                             AccDataType,
                                             InGridDesc_M_K,
                                             decltype(thread_buffer_desc),
                                             ThreadBufferLengths,
                                             ThreadBufferDimAccessOrder,
                                             InSrcVectorDim,
                                             InSrcVectorSize,
                                             1,
                                             false>(
                in_grid_desc_m_k, make_multi_index(thread_global_1d_id * MThreadSliceSize, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, KThreadSliceSize);

        index_t indexStart    = 0;
        index_t reducedLength = 0;
        if constexpr(HaveIndexInput)
        {
            auto threadwise_src_idx_load =
                ThreadwiseTensorSliceTransfer_v2<IndexDataType,
                                                 IndexDataType,
                                                 InGridDesc_M_K,
                                                 decltype(thread_buffer_desc),
                                                 ThreadBufferLengths,
                                                 ThreadBufferDimAccessOrder,
                                                 InSrcVectorDim,
                                                 InSrcVectorSize,
                                                 1,
                                                 false>(
                    in_grid_desc_m_k, make_multi_index(thread_global_1d_id * MThreadSliceSize, 0));

            do
            {
                threadwise_src_val_load.Run(in_grid_desc_m_k,
                                            in_global_val_buf,
                                            thread_buffer_desc,
                                            make_tuple(I0, I0),
                                            in_thread_val_buf);

                threadwise_src_idx_load.Run(in_grid_desc_m_k,
                                            in_global_idx_buf,
                                            thread_buffer_desc,
                                            make_tuple(I0, I0),
                                            in_thread_idx_buf);

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    // do element-wise pre-reduction operation
                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset =
                            thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));

                        in_elementwise_op(in_thread_val_buf(Number<offset>{}),
                                          in_thread_val_buf(Number<offset>{}));
                    });
                });

                ThreadwiseReduceWithIndex::Reduce(
                    in_thread_val_buf, in_thread_idx_buf, accu_value_buf, accu_index_buf);

                threadwise_src_val_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);
                threadwise_src_idx_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

                indexStart += KThreadSliceSize;
                reducedLength += KThreadSliceSize;
            } while(reducedLength < toReduceLength);
        }
        else
        {
            do
            {
                threadwise_src_val_load.Run(in_grid_desc_m_k,
                                            in_global_val_buf,
                                            thread_buffer_desc,
                                            make_tuple(I0, I0),
                                            in_thread_val_buf);

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    // do element-wise pre-reduction operation
                    static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                        constexpr auto offset =
                            thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));

                        in_thread_idx_buf(Number<offset>{}) = indexStart + iK();

                        in_elementwise_op(in_thread_val_buf(Number<offset>{}),
                                          in_thread_val_buf(Number<offset>{}));
                    });
                });

                ThreadwiseReduceWithIndex::Reduce(
                    in_thread_val_buf, in_thread_idx_buf, accu_value_buf, accu_index_buf);

                threadwise_src_val_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

                indexStart += KThreadSliceSize;
                reducedLength += KThreadSliceSize;
            } while(reducedLength < toReduceLength);
        };

        // for indiced operation, acc_elementwise_op shoud do nothing
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            acc_elementwise_op(accu_value_buf(I), accu_value_buf(I));

            accu_value_buf(I) *= alpha;
        });

        constexpr auto reduced_data_desc = ThreadReduceDstDesc_M{};

        if(!float_equal_zero{}(beta))
        {
            auto threadwise_dst_load = ThreadwiseTensorSliceTransfer_v2<OutDataType,
                                                                        OutDataType,
                                                                        OutGridDesc_M,
                                                                        decltype(reduced_data_desc),
                                                                        Sequence<MThreadSliceSize>,
                                                                        Sequence<0>,
                                                                        0,
                                                                        1,
                                                                        1,
                                                                        false>(
                out_grid_desc_m, make_multi_index(thread_global_1d_id * MThreadSliceSize));

            StaticBuffer<AddressSpaceEnum::Vgpr, OutDataType, MThreadSliceSize, true>
                priorDstValue_buf;

            threadwise_dst_load.Run(out_grid_desc_m,
                                    out_global_val_buf,
                                    reduced_data_desc,
                                    make_tuple(I0),
                                    priorDstValue_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                accu_value_buf(I) += type_convert<AccDataType>(priorDstValue_buf[I]) * beta;
            });
        };

        auto threadwise_dst_val_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               OutDataType,
                                               decltype(reduced_data_desc),
                                               OutGridDesc_M,
                                               PassThroughOp,
                                               Sequence<MThreadSliceSize>,
                                               Sequence<0>,
                                               0,
                                               OutDstVectorSize,
                                               OutMemoryDataOperation,
                                               1,
                                               false>(
                out_grid_desc_m,
                make_multi_index(thread_global_1d_id * MThreadSliceSize),
                PassThroughOp{});

        auto threadwise_dst_idx_store =
            ThreadwiseTensorSliceTransfer_v1r3<IndexDataType,
                                               IndexDataType,
                                               decltype(reduced_data_desc),
                                               OutGridDesc_M,
                                               PassThroughOp,
                                               Sequence<MThreadSliceSize>,
                                               Sequence<0>,
                                               0,
                                               OutDstVectorSize,
                                               OutMemoryDataOperation,
                                               1,
                                               false>(
                out_grid_desc_m,
                make_multi_index(thread_global_1d_id * MThreadSliceSize),
                PassThroughOp{});

        threadwise_dst_val_store.Run(
            reduced_data_desc, make_tuple(I0), accu_value_buf, out_grid_desc_m, out_global_val_buf);

        threadwise_dst_idx_store.Run(
            reduced_data_desc, make_tuple(I0), accu_index_buf, out_grid_desc_m, out_global_idx_buf);
    };
};

} // namespace ck
