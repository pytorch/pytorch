// SPDX-License-Identifier: MIT
// // Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseElementwise2dFunctor,
          typename InGrid2dDescTuple,
          typename OutGrid2dDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename ElementwiseOperation>
__global__ void kernel_elementwise_2d(const InGrid2dDescTuple in_grid_2d_desc_tuple,
                                      const OutGrid2dDescTuple out_grid_2d_desc_tuple,
                                      const InDataTypePointerTuple p_in_global_tuple,
                                      const OutDataTypePointerTuple p_out_global_tuple,
                                      const ElementwiseOperation elementwise_op,
                                      const index_t num_threads_m,
                                      const index_t num_threads_n)
{
    GridwiseElementwise2dFunctor::Run(in_grid_2d_desc_tuple,
                                      out_grid_2d_desc_tuple,
                                      p_in_global_tuple,
                                      p_out_global_tuple,
                                      elementwise_op,
                                      num_threads_m,
                                      num_threads_n);
}

template <typename InGrid2dDescTuple,
          typename OutGrid2dDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename ElementwiseOperation,
          index_t MPerThread,
          index_t NPerThread,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq>
struct GridwiseElementwise_2D
{
    static constexpr index_t NumInput  = InDataTypePointerTuple::Size();
    static constexpr index_t NumOutput = OutDataTypePointerTuple::Size();

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size() &&
                      NumInput == InGrid2dDescTuple::Size() &&
                      NumOutput == OutGrid2dDescTuple::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr auto thread_buffer_desc_mn =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MPerThread>{}, Number<NPerThread>{}));

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    __device__ static void Run(const InGrid2dDescTuple in_grid_2d_desc_tuple,
                               const OutGrid2dDescTuple out_grid_2d_desc_tuple,
                               const InDataTypePointerTuple p_in_global_tuple,
                               const OutDataTypePointerTuple p_out_global_tuple,
                               const ElementwiseOperation elementwise_op,
                               const index_t num_threads_m,
                               const index_t num_threads_n)
    {
        auto in_thread_buf_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    DataType,
                                    MPerThread * NPerThread,
                                    true>{};
            },
            Number<NumInput>{});

        auto out_thread_buf_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(OutDataTypePointerTuple{}[I])>;
                using DataType        = remove_pointer_t<DataTypePointer>;

                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    DataType,
                                    MPerThread * NPerThread,
                                    true>{};
            },
            Number<NumOutput>{});

        auto in_global_buf_tuple = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_in_global_tuple[I], in_grid_2d_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumInput>{});

        auto out_global_buf_tuple = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_out_global_tuple[I], out_grid_2d_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumOutput>{});

        const auto M = in_grid_2d_desc_tuple[I0].GetLength(I0);
        const auto N = in_grid_2d_desc_tuple[I0].GetLength(I1);

        const index_t loop_step_m = num_threads_m * MPerThread;
        const index_t loop_step_n = num_threads_n * NPerThread;

        const index_t thread_1d_id = get_thread_global_1d_id();
        index_t tid_m              = thread_1d_id / num_threads_n;
        index_t tid_n              = thread_1d_id % num_threads_n;

        const auto thread_global_offset = make_multi_index(tid_m * MPerThread, tid_n * NPerThread);

        auto in_global_load_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return ThreadwiseTensorSliceTransfer_v2<
                    DataType,
                    DataType,
                    decltype(in_grid_2d_desc_tuple[I]),
                    decltype(thread_buffer_desc_mn),
                    Sequence<MPerThread, NPerThread>, // SliceLengths
                    Sequence<0, 1>,                   // DimAccessOrder
                    0,                                // SrcVectorDim
                    InScalarPerVectorSeq::At(I),      // ScalarPerVector
                    1,                                // SrcScalarStrideInVector
                    true>{in_grid_2d_desc_tuple[I], thread_global_offset};
            },
            Number<NumInput>{});

        auto out_global_store_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(OutDataTypePointerTuple{}[I])>;
                using DataType        = remove_pointer_t<DataTypePointer>;

                return ThreadwiseTensorSliceTransfer_v1r3<
                    DataType,
                    DataType,
                    decltype(thread_buffer_desc_mn),
                    decltype(out_grid_2d_desc_tuple[I]),
                    PassThroughOp,
                    Sequence<MPerThread, NPerThread>, // SliceLengths
                    Sequence<0, 1>,                   // DimAccessOrder
                    1,                                // SrcVectorDim
                    1,                                // OutScalarPerVectorSeq::At(I),
                    InMemoryDataOperationEnum::Set,
                    1,
                    true>(out_grid_2d_desc_tuple[I], thread_global_offset, PassThroughOp{});
            },
            Number<NumOutput>{});

        index_t num_iter_m = M / (loop_step_m);
        do
        {
            index_t num_iter_n = N / (loop_step_n);
            do
            {
                static_for<0, NumInput, 1>{}([&](auto I) {
                    in_global_load_tuple(I).Run(in_grid_2d_desc_tuple[I],
                                                in_global_buf_tuple[I],
                                                thread_buffer_desc_mn,
                                                make_tuple(I0, I0),
                                                in_thread_buf_tuple(I));

                    in_global_load_tuple(I).MoveSrcSliceWindow(in_grid_2d_desc_tuple[I],
                                                               make_multi_index(0, loop_step_n));
                });

                static_for<0, MPerThread, 1>{}([&](auto iM) {
                    static_for<0, NPerThread, 1>{}([&](auto iN) {
                        constexpr auto offset =
                            thread_buffer_desc_mn.CalculateOffset(make_tuple(iM, iN));
                        // get reference to in data
                        const auto in_data_refs = generate_tie(
                            // return type should be lvalue
                            [&](auto I) -> const auto& {
                                return in_thread_buf_tuple(I)(Number<offset>{});
                            },
                            Number<NumInput>{});

                        // get referenec to dst data
                        auto out_data_refs = generate_tie(
                            // return type should be lvalue
                            [&](auto I) -> auto& {
                                return out_thread_buf_tuple(I)(Number<offset>{});
                            },
                            Number<NumOutput>{});
                        unpack2(elementwise_op, out_data_refs, in_data_refs);
                    });
                });

                static_for<0, NumOutput, 1>{}([&](auto I) {
                    out_global_store_tuple(I).Run(thread_buffer_desc_mn,
                                                  make_tuple(I0, I0),
                                                  out_thread_buf_tuple[I],
                                                  out_grid_2d_desc_tuple[I],
                                                  out_global_buf_tuple(I));

                    out_global_store_tuple(I).MoveDstSliceWindow(out_grid_2d_desc_tuple[I],
                                                                 make_multi_index(0, loop_step_n));
                });

            } while(--num_iter_n);

            static_for<0, NumInput, 1>{}([&](auto I) {
                in_global_load_tuple(I).MoveSrcSliceWindow(
                    in_grid_2d_desc_tuple[I],
                    make_multi_index(loop_step_m, -(N / loop_step_n) * loop_step_n));
            });

            static_for<0, NumOutput, 1>{}([&](auto I) {
                out_global_store_tuple(I).MoveDstSliceWindow(
                    out_grid_2d_desc_tuple[I],
                    make_multi_index(loop_step_m, -(N / loop_step_n) * loop_step_n));
            });
        } while(--num_iter_m);
    }
};

} // namespace ck
