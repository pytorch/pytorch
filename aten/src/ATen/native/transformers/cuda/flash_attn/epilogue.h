/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/layout/layout.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <ATen/native/transformers/cuda/flash_attn/gemm.h>
#include <ATen/native/transformers/cuda/flash_attn/epilogue_predicated_tile_iterator.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MmaCore>
struct FMHAEpilogue {

    using ThreadblockShape = typename MmaCore::Shape;
    using WarpMma = typename MmaCore::MmaTensorOp;
    using LayoutC = typename MmaCore::LayoutC;
    using Element = typename MmaCore::ElementA;
    using ElementC = typename MmaCore::ElementC;

    static constexpr int kPartitionsK = ThreadblockShape::kK / MmaCore::WarpShape::kK;

    using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
                                    typename WarpMma::Shape,
                                    typename WarpMma::Policy::Operator::Shape,
                                    typename WarpMma::Policy::Operator::ElementC,
                                    typename WarpMma::Policy::Operator::FragmentC,
                                    LayoutC>;
    using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;
    static constexpr int kIterationsStore = AccumulatorFragmentIterator::kIterations;

    // Maybe elementsPerAccess should vary: 4 for d=64, 2 for d=32?
    using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
        ThreadblockShape, typename WarpMma::Shape, kPartitionsK, Element, /*ElementsPerAccess=*/4>::Type;
    using OutputTileThreadMapAccum = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
        ThreadblockShape, typename WarpMma::Shape, kPartitionsK, ElementC, /*ElementsPerAccess=*/4>::Type;

    using GmemIterator = fmha::EpiloguePredicatedTileIterator<
        OutputTileThreadMap,
        Element
    >;
    // which ThreadMap should we use?
    using GmemIteratorAccum = fmha::EpiloguePredicatedTileIterator<
        // OutputTileThreadMapAccum,
        OutputTileThreadMap,
        ElementC
    >;


    using DefaultIterators = cutlass::epilogue::threadblock::detail::DefaultIteratorsTensorOp<
        Element, ElementC, /*ElementsPerAccess=*/4, ThreadblockShape, typename WarpMma::Shape,
        typename WarpMma::Policy::Operator::Shape, typename OutputTileThreadMap::CompactedThreadMap>;
    using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
    static_assert(WarpTileIterator::kIterations == kIterationsStore, "");
    using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;
    using OutputFragment = typename SharedLoadIterator::Fragment;

    // using Padding = cutlass::MatrixShape<0, 0>;
    using Padding = cutlass::MatrixShape<0, 64 / cutlass::sizeof_bits<ElementC>::value * 4>;
    static constexpr int kFragmentsPerIteration = kIterationsStore;  // TODO: could be 1 for Volta?
    /*Using kIterationsStore here so that we get the right storage size*/
    using EpilogueBase = typename cutlass::epilogue::threadblock::EpilogueBase<
        ThreadblockShape, typename WarpMma::Shape, kPartitionsK, AccumulatorFragmentIterator, WarpTileIterator,
        Padding, kIterationsStore>;

    using SharedStorage = typename EpilogueBase::SharedStorage;
    static constexpr int kSmemTiles = EpilogueBase::kFragmentsPerIteration;
    static constexpr int kSmemPointerOffset = SharedStorage::StorageShape::kCount / kSmemTiles;
    static constexpr int kSmemPointerOffsetPerWarp = SharedStorage::StorageShape::kCount / (kSmemTiles * kPartitionsK);

    SharedStorage *shared_storage;
    WarpTileIterator warp_tile_iterator;

    inline __device__ FMHAEpilogue(void *smem, const int tidx)
        : shared_storage(reinterpret_cast<SharedStorage *>(smem))
        , warp_tile_iterator(shared_storage->reference(), threadIdx.x % 32) {

        // const int warp_idx = tidx / 32;
        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        // https://github.com/NVIDIA/cutlass/blob/e66bfcb1f880792caa46b1e983c4114e23afa5f3/include/cutlass/gemm/kernel/gemm_with_fused_epilogue.h#L520
        const int warp_idx = __shfl_sync(0xffffffff, tidx / 32, 0);

        cutlass::MatrixCoord warp_offset{kIterationsStore * warp_idx, 0};

        warp_tile_iterator.add_tile_offset(warp_offset);
    }

    // Store the accumulators.
    inline __device__ void store(const AccumulatorTile &acc) {
        AccumulatorFragmentIterator accum_fragment_iterator(acc);
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < kIterationsStore; ++p) {
            typename AccumulatorFragmentIterator::Fragment accum_fragment;
            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            warp_tile_iterator.store(accum_fragment);
            if (p < kIterationsStore - 1) {
                warp_tile_iterator.add_pointer_offset(kSmemPointerOffsetPerWarp);
            }
        }
        if (kIterationsStore > 1) {
            warp_tile_iterator.add_pointer_offset((1 - kIterationsStore) * kSmemPointerOffsetPerWarp);
        }
    }

    // Load the accumulators
    template<bool zero_init=true>
    inline __device__ void load(OutputFragment (&out)[kFragmentsPerIteration],
                                const int tidx) {
        SharedLoadIterator shared_load_iterator(shared_storage->reference(), tidx);
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < EpilogueBase::kFragmentsPerIteration; ++p) {
            OutputFragment aligned_accum_fragment[kPartitionsK];
            shared_load_iterator.load(aligned_accum_fragment[0]);
            cutlass::plus<OutputFragment> add_fragments;
            if (kPartitionsK > 1) {
                CUTLASS_PRAGMA_UNROLL
                for ( int i = 1; i < kPartitionsK; ++i) {
                    shared_load_iterator.add_pointer_offset(kSmemPointerOffsetPerWarp * kIterationsStore);
                    shared_load_iterator.load(aligned_accum_fragment[i]);
                    aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
                }
                shared_load_iterator.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffsetPerWarp * kIterationsStore);
            }
            if (p < EpilogueBase::kFragmentsPerIteration - 1) {
                shared_load_iterator.add_pointer_offset(kSmemPointerOffsetPerWarp);
            }

            out[p] = zero_init ? aligned_accum_fragment[0] : add_fragments(out[p], aligned_accum_fragment[0]);
        }
    }

};

}  // namespace fmha
