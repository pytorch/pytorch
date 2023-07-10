// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename Grid1dBufferDescTuple,
          index_t NumBuffer,
          index_t BlockSize,
          typename DataTypePointerTuple,
          typename DataTypeTuple>
__global__ void
kernel_multiple_buffer_set_value(const Grid1dBufferDescTuple grid_1d_buffer_desc_tuple,
                                 DataTypePointerTuple p_global_tuple,
                                 DataTypeTuple value_tuple)

{
    static_assert(NumBuffer == DataTypePointerTuple::Size() && NumBuffer == DataTypeTuple::Size(),
                  "The tuple size should be same as NumBuffer!");

    static_for<0, NumBuffer, 1>{}([&](auto iB) {
        using DataTypePointer     = remove_cvref_t<decltype(DataTypePointerTuple{}[iB])>;
        using DataTypeFromPointer = remove_pointer_t<DataTypePointer>;
        using DataType            = remove_cvref_t<decltype(DataTypeTuple{}[iB])>;

        static_assert(is_same<DataType, DataTypeFromPointer>::value,
                      "Types in tuples does not match!");
    });

    constexpr auto I0 = Number<0>{};

    const index_t thread_global_id = get_thread_global_1d_id();

    auto value_buf_tuple = generate_tuple(
        [&](auto iB) {
            using DataType = remove_cvref_t<decltype(DataTypeTuple{}[iB])>;

            return StaticBuffer<AddressSpaceEnum::Vgpr, DataType, 1, true>{};
        },
        Number<NumBuffer>{});

    static_for<0, NumBuffer, 1>{}([&](auto iB) {
        static_for<0, 1, 1>{}([&](auto J) { value_buf_tuple(iB)(J) = value_tuple[iB]; });
    });

    auto global_buf_tuple = generate_tuple(
        [&](auto iB) {
            return make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_global_tuple(iB), grid_1d_buffer_desc_tuple[iB].GetElementSpaceSize());
        },
        Number<NumBuffer>{});

    constexpr auto val_buff_desc = make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

    static_for<0, NumBuffer, 1>{}([&](auto iB) {
        using DataType      = remove_cvref_t<decltype(DataTypeTuple{}[iB])>;
        using PassThroughOp = tensor_operation::element_wise::PassThrough;

        auto threadwise_store =
            ThreadwiseTensorSliceTransfer_v1r3<DataType,
                                               DataType,
                                               decltype(val_buff_desc),
                                               decltype(Grid1dBufferDescTuple{}[iB]),
                                               PassThroughOp,
                                               Sequence<1>,
                                               Sequence<0>,
                                               0,
                                               1,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                grid_1d_buffer_desc_tuple[iB], make_multi_index(thread_global_id), PassThroughOp{});

        threadwise_store.Run(val_buff_desc,
                             make_tuple(I0),
                             value_buf_tuple(iB),
                             grid_1d_buffer_desc_tuple[iB],
                             global_buf_tuple(iB));
    });
};

} // namespace ck
