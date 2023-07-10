// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize, typename DataType, typename Grid1dBufferDescType>
__global__ void kernel_buffer_set_value(const Grid1dBufferDescType grid_1d_buffer_desc,
                                        DataType* const __restrict__ p_global,
                                        DataType value)

{

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    constexpr auto I0 = Number<0>{};

    const index_t thread_local_id = get_thread_local_1d_id();
    const index_t block_global_id = get_block_1d_id();

    const index_t thread_global_id = block_global_id * BlockSize + thread_local_id;

    StaticBuffer<AddressSpaceEnum::Vgpr, DataType, 1, true> value_buf;

    value_buf(I0) = value;

    constexpr auto val_buff_desc = make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

    auto global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        p_global, grid_1d_buffer_desc.GetElementSpaceSize());

    if(thread_global_id < grid_1d_buffer_desc.GetElementSize())
    {
        auto threadwise_store = ThreadwiseTensorSliceTransfer_v1r3<DataType,
                                                                   DataType,
                                                                   decltype(val_buff_desc),
                                                                   Grid1dBufferDescType,
                                                                   PassThroughOp,
                                                                   Sequence<1>,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperationEnum::Set,
                                                                   1,
                                                                   true>(
            grid_1d_buffer_desc, make_multi_index(thread_global_id), PassThroughOp{});

        threadwise_store.Run(
            val_buff_desc, make_tuple(I0), value_buf, grid_1d_buffer_desc, global_buf);
    }
};

} // namespace ck
