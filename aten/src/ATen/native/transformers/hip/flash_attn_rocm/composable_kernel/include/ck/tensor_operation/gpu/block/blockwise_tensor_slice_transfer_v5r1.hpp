// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v5r1.hpp"

namespace ck {

// this version does following things to avoid scratch memory issue
// 1. Use StaticallyIndexedArray instead of C array for thread buffer
// 2. ThreadwiseTensorSliceTransfer_v3 does not keep reference to tensor descriptor
// 3. ThreadwiseTensorSliceTransfer_v3::Run() does not construct new tensor coordinate
template <index_t BlockSize,
          InMemoryDataOperationEnum DstInMemOp,
          typename BlockSliceLengths,
          typename ThreadSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          typename SrcVectorTensorLengths,
          typename DstVectorTensorLengths,
          typename SrcVectorTensorContiguousDimOrder,
          typename DstVectorTensorContiguousDimOrder,
          bool ThreadTransferSrcResetCoordinateAfterRun,
          bool ThreadTransferDstResetCoordinateAfterRun>
struct BlockwiseTensorSliceTransfer_v5r1
{
    static constexpr index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();

    using Index = MultiIndex<nDim>;

    __device__ constexpr BlockwiseTensorSliceTransfer_v5r1(const SrcDesc& src_desc,
                                                           const Index& src_block_slice_origin,
                                                           const DstDesc& dst_desc,
                                                           const Index& dst_block_slice_origin)
        : threadwise_transfer_(
              src_desc, make_zero_multi_index<nDim>(), dst_desc, make_zero_multi_index<nDim>())

    {
        static_assert(nDim == remove_cvref_t<SrcDesc>::GetNumOfDimension() &&
                          nDim == remove_cvref_t<DstDesc>::GetNumOfDimension() &&
                          nDim == BlockSliceLengths::Size() && nDim == ThreadSliceLengths::Size() &&
                          nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(
            is_same<BlockSliceLengths, decltype(ThreadSliceLengths{} * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(BlockSize >= thread_cluster_desc_.GetElementSize(),
                      "wrong! BlockSize too small");

        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(get_thread_local_1d_id()));

            const auto thread_data_idx_begin = thread_cluster_idx * ThreadSliceLengths{};

            threadwise_transfer_.SetSrcSliceOrigin(src_desc,
                                                   src_block_slice_origin + thread_data_idx_begin);
            threadwise_transfer_.SetDstSliceOrigin(dst_desc,
                                                   dst_block_slice_origin + thread_data_idx_begin);
        }
    }

    template <typename SrcBuffer>
    __device__ void RunRead(const SrcDesc& src_desc, const SrcBuffer& src_buf)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunRead(src_desc, src_buf);
        }
    }

    template <typename DstBuffer>
    __device__ void RunWrite(const DstDesc& dst_desc, DstBuffer& dst_buf)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunWrite(dst_desc, dst_buf);
        }
    }

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(src_desc, step);
        }
    }

    // SrcMoveSliceWindowStepHack to control index calculation move slice window
    template <typename SrcMoveSliceWindowStepHack>
    __device__ void
    MoveSrcSliceWindow(const SrcDesc& src_desc,
                       const Index& step,
                       const SrcMoveSliceWindowStepHack& src_move_slice_window_step_hack)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveSrcSliceWindow(
                src_desc, step, src_move_slice_window_step_hack);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc, const Index& step)
    {
        if(BlockSize == thread_cluster_desc_.GetElementSize() or
           get_thread_local_1d_id() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_desc, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    using ThreadwiseTransfer =
        ThreadwiseTensorSliceTransfer_v5r1<ThreadSliceLengths,
                                           DstInMemOp,
                                           SrcData,
                                           DstData,
                                           SrcDesc,
                                           DstDesc,
                                           SrcDimAccessOrder,
                                           DstDimAccessOrder,
                                           SrcVectorTensorLengths,
                                           DstVectorTensorLengths,
                                           SrcVectorTensorContiguousDimOrder,
                                           DstVectorTensorContiguousDimOrder,
                                           ThreadTransferSrcResetCoordinateAfterRun,
                                           ThreadTransferDstResetCoordinateAfterRun>;

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
