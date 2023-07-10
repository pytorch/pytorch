// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7.hpp"

namespace ck {

// Thread-group level multi-source, multi-destination tensor slice data movement
// Assume:
//   1. All sources and destinations are DynamicBuffer
//   2. Same VectorDim and ScalerPerVector for all sources and destinations
//   3. DstInMemOps are per destination tensor
//   4. ThreadTransferSrcResetCoordinateAfterRunFlags are per source tensor
//   5. ThreadTransferDstResetCoordinateAfterRunFlags are per destination tensor
//
// Does following things to avoid scratch memory issue
//   1. Pass tensor descritpors by reference (or tuple of references)
//   2. Does not keep reference to tensor descriptor
//   3. Does not construct new tensor coordinate when call Run()
template <typename ThreadGroup,
          typename SrcDatas,
          typename DstDatas,
          typename SrcDescs,
          typename DstDescs,
          typename ElementwiseOperation,
          typename DstInMemOps, // Sequence<InMemoryDataOperationEnum ...>
          typename SliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename DimAccessOrder,
          index_t VectorDim,
          index_t ScalarPerVector,
          typename ThreadTransferSrcResetCoordinateAfterRunFlags,
          typename ThreadTransferDstResetCoordinateAfterRunFlags>
struct ThreadGroupTensorSliceTransfer_v7
{
    static constexpr index_t nDim =
        remove_cvref_t<tuple_element_t<0, SrcDescs>>::GetNumOfDimension();

    static constexpr index_t nSrc = remove_cvref_t<SrcDescs>::Size();
    static constexpr index_t nDst = remove_cvref_t<DstDescs>::Size();

    using Index = MultiIndex<nDim>;

    static constexpr auto thread_slice_lengths = SliceLengths{} / ThreadClusterLengths{};

    __device__ constexpr ThreadGroupTensorSliceTransfer_v7(
        const SrcDescs& src_descs,
        const StaticallyIndexedArray<Index, nSrc>& src_block_slice_origins,
        const DstDescs& dst_descs,
        const StaticallyIndexedArray<Index, nDst>& dst_block_slice_origins,
        const ElementwiseOperation& element_op)
        : threadwise_transfer_(src_descs,
                               StaticallyIndexedArray<Index, nSrc>{},
                               dst_descs,
                               StaticallyIndexedArray<Index, nDst>{},
                               element_op)
    {
        static_assert(nSrc == SrcDatas::Size() && nSrc == SrcDescs::Size() &&
                          nSrc == ThreadTransferSrcResetCoordinateAfterRunFlags::Size() &&
                          nDst == DstDatas::Size() && nDst == DstDescs::Size() &&
                          nDst == ThreadTransferDstResetCoordinateAfterRunFlags::Size(),
                      "wrong!");

        static_for<0, nSrc, 1>{}([&](auto i) {
            static_assert(
                nDim == remove_cvref_t<tuple_element_t<i.value, SrcDescs>>::GetNumOfDimension(),
                "wrong!");
        });

        static_for<0, nDst, 1>{}([&](auto i) {
            static_assert(
                nDim == remove_cvref_t<tuple_element_t<i.value, DstDescs>>::GetNumOfDimension(),
                "wrong!");
        });

        static_assert(nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == DimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<SliceLengths, decltype(thread_slice_lengths * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(ThreadGroup::GetNumOfThread() >= thread_cluster_desc_.GetElementSize(),
                      "wrong! ThreadGroup::GetNumOfThread() too small");

        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(get_thread_local_1d_id()));

            const auto thread_data_idx_begin = thread_cluster_idx * thread_slice_lengths;

            const auto src_thread_slice_origins = generate_tuple(
                [&](auto i) { return src_block_slice_origins[i] + thread_data_idx_begin; },
                Number<nSrc>{});

            const auto dst_thread_slice_origins = generate_tuple(
                [&](auto i) { return dst_block_slice_origins[i] + thread_data_idx_begin; },
                Number<nDst>{});

            threadwise_transfer_.SetSrcSliceOrigins(src_descs, src_thread_slice_origins);
            threadwise_transfer_.SetDstSliceOrigins(dst_descs, dst_thread_slice_origins);
        }
    }

    template <typename SrcBuffers, typename DstBuffers>
    __device__ void Run(const SrcDescs& src_descs,
                        const SrcBuffers& src_bufs,
                        const DstDescs& dst_descs,
                        DstBuffers dst_bufs)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.Run(src_descs, src_bufs, dst_descs, dst_bufs);
        }
    }

    template <index_t ISrc>
    __device__ void
    MoveSrcSliceWindow(const SrcDescs& src_descs, Number<ISrc> iSrc, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(src_descs, iSrc, step);
        }
    }

    template <index_t IDst>
    __device__ void
    MoveDstSliceWindow(const DstDescs& dst_descs, Number<IDst> iDst, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_descs, iDst, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer =
        ThreadwiseTensorSliceTransfer_v7<SrcDatas,
                                         DstDatas,
                                         SrcDescs,
                                         DstDescs,
                                         ElementwiseOperation,
                                         DstInMemOps,
                                         decltype(thread_slice_lengths),
                                         DimAccessOrder,
                                         VectorDim,
                                         ScalarPerVector,
                                         ThreadTransferSrcResetCoordinateAfterRunFlags,
                                         ThreadTransferDstResetCoordinateAfterRunFlags>;

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
