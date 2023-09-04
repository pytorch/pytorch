/***************************************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include <ATen/native/transformers/cuda/flash_attn/block_info.h>
#include <ATen/native/transformers/cuda/flash_attn/kernel_traits.h>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>
#include <ATen/native/transformers/cuda/flash_attn/softmax.h>
#include <ATen/native/transformers/cuda/flash_attn/philox.cuh>

namespace pytorch_flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_B_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = decltype(size<1>(TileShape_MNK{}))::value / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    // This gives the correct layout, idk why.
    // auto t = make_tile(Layout<Shape<Shape<_8, _2>, _2>,
    //                           Stride<Stride<_1, _64>, _8> >{},
    // auto t = make_tile(Layout<Shape<_8, _2, _2>,
    //                           Stride<_1, _64, _8> >{},
    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{},       // (1, 64, 8) or (1, 32, 8)
                       make_layout(size<2>(TileShape_MNK{})));
    // if (cute::thread0()) {printf("make_tiled_copy_B_warpcontiguousN "); print(t); printf("\n");  }
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = decltype(size<1>(TileShape_MNK{}))::value / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(make_layout(size<0>(TileShape_MNK{})),
                       Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{});       // (1, 64, 8) or (1, 32, 8)
    // if (cute::thread0()) {printf("make_tiled_copy_C_warpcontiguousN "); print(t); printf("\n");  }
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_ROW, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
inline __device__ void dot_do_o(Tensor<Engine0, Layout0> const &do_, Tensor<Engine0, Layout0> const &o,
                                Tensor<Engine1, Layout1> &dP_sum, Tensor<Engine2, Layout2> &sdPsum,
                                const int gdP_col_stride, const float scale) {
    static_assert(Layout0::rank == 3, "Only support 3D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(do_.layout() == o.layout());
    // Reshape do_ and o from (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, 8 * kHeadDim / 64)
    // The last coordinate is the "page".
    Tensor do_reshaped = make_tensor(do_.data(), make_layout(get<1>(do_.layout()),
                                                             make_layout(get<0>(do_.layout()),
                                                                         get<2>(do_.layout()))));
    Tensor o_reshaped = make_tensor(o.data(), do_reshaped.layout());
    Tensor do_fp32 = pytorch_flash::convert_type<float>(do_reshaped);
    Tensor o_fp32 = pytorch_flash::convert_type<float>(o_reshaped);
    #pragma unroll
    for (int mi = 0; mi < size<0>(do_reshaped); ++mi) {
        float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
        #pragma unroll
        for (int ni = 1; ni < size<1>(do_reshaped); ni++) {
            dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
        }
        pytorch_flash::SumOp<float> sum_op;
        dP_sum_cur = pytorch_flash::Allreduce<THREADS_PER_ROW>::run(dP_sum_cur, sum_op) * scale;
        if (threadIdx.x % THREADS_PER_ROW == 0) {
            dP_sum(mi * gdP_col_stride + threadIdx.x / THREADS_PER_ROW) = dP_sum_cur;
            // recast<float>(sdPsum)(mi * gdP_col_stride + threadIdx.x / THREADS_PER_ROW) = dP_sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Just compute dot(do, o) and write the result (softmax_d) to global memory as a separate kernel.
// This is used in the case where we want to parallelize the backward across seqlen_k.
template<bool Clear_dQaccum=true, typename Kernel_traits, typename Params>
inline __device__ void compute_dot_do_o(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const BlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq_accum = ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM;

    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
    Tensor dP_sum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    // TODO: careful, we're zeroing out dQaccum with type float4, but when
    // we do atomicAdds, we use type float. The layouts are different. Check this.
    typename Kernel_traits::GmemTiledCopydQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    Tensor cdO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tdOcdO = gmem_thr_copy_dO.partition_S(cdO);

    // Allocate predicate tensors for k
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdOgdO)));
    // Set predicates for k bounds
    #pragma unroll
    for (int k = 0; k < size(tdOpdO); ++k) {tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d;}

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgO, tdOrO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    // By right we need to scale dP up by 1/p_dropout, but instead we don't and only scale the final
    // results (dQ and dK) by 1/p_dropout. So we need to keep dP_sum scaled down by p_dropout here,
    // so that (dP - dP_sum) is on the same scale.
    dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, dP_sum, dP_sum,
                                                Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    if (Clear_dQaccum) {
        Tensor zero = make_fragment_like(tdQgdQaccum);
        clear(zero);
        cute::copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void clear_dKVaccum(const Params &params) {
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int n_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const BlockInfo binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    const index_t row_offset_dkv_accum = ((bidb * params.h_k + bidh) * params.seqlen_k_rounded + n_block * kBlockN) * params.d_rounded;

    Tensor gdKaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dk_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
    Tensor gdVaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dv_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});

    typename Kernel_traits::GmemTiledCopydQaccum gmem_tiled_copy_dKVaccum;
    auto gmem_thr_copy_dKVaccum = gmem_tiled_copy_dKVaccum.get_thread_slice(tidx);
    Tensor tdKgdKaccum = gmem_thr_copy_dKVaccum.partition_D(gdKaccum);
    Tensor tdVgdVaccum = gmem_thr_copy_dKVaccum.partition_D(gdVaccum);
    Tensor zero = make_fragment_like(tdKgdKaccum);
    clear(zero);
    cute::copy(gmem_tiled_copy_dKVaccum, zero, tdKgdKaccum);
    cute::copy(gmem_tiled_copy_dKVaccum, zero, tdVgdVaccum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert dQ from dQaccum (in float) to fp16/bf16.
// This is used in the case where we want to parallelize the backward across seqlen_k.
template<typename Kernel_traits, typename Params>
inline __device__ void convert_dQ(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const BlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = ((bidb * params.h + bidh) * params.seqlen_q_rounded
                                         + m_block * kBlockM) * params.d_rounded;

    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  Stride<Int<kHeadDim>, _1>{});

    Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                             typename Kernel_traits::SmemLayoutdQ{});

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));

    Tensor tdQrdQaccum = make_fragment_like(tdQgdQaccum);
    cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) {
        acc_dq(i) = tdQrdQaccum(i) * params.scale_softmax_rp_dropout;
    }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = pytorch_flash::convert_type<Element>(acc_dq);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);

    Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    #pragma unroll
    for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert dK and dV from dKaccum and dVaccum (in float) to fp16/bf16.
// This is used in the case where we want to parallelize the backward across seqlen_q.
template<typename Kernel_traits, typename Params>
inline __device__ void convert_dKV(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    const int n_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const BlockInfo binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    const index_t row_offset_dkv_accum = ((bidb * params.h_k + bidh) * params.seqlen_k_rounded
                                          + n_block * kBlockN) * params.d_rounded;

    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));
    Tensor gdKaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dk_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  Stride<Int<kHeadDim>, _1>{});
    Tensor gdVaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dv_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  Stride<Int<kHeadDim>, _1>{});

    Tensor sdK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                             typename Kernel_traits::SmemLayoutdKV{});
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dKVaccum;
    auto gmem_thr_copy_dKVaccum = gmem_tiled_copy_dKVaccum.get_thread_slice(tidx);

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);  // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
    Tensor tdKgdKaccum = gmem_thr_copy_dKVaccum.partition_S(gdKaccum);
    Tensor tdVgdVaccum = gmem_thr_copy_dKVaccum.partition_S(gdVaccum);

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    CUTE_STATIC_ASSERT_V(size(acc_dk) == size(tdKgdKaccum));
    CUTE_STATIC_ASSERT_V(size(acc_dv) == size(tdVgdVaccum));

    Tensor tdKrdKaccum = make_fragment_like(tdKgdKaccum);
    Tensor tdVrdVaccum = make_fragment_like(tdVgdVaccum);
    cute::copy(gmem_tiled_copy_dKVaccum, tdKgdKaccum, tdKrdKaccum);
    cute::copy(gmem_tiled_copy_dKVaccum, tdVgdVaccum, tdVrdVaccum);
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) {
        acc_dk(i) = tdKrdKaccum(i) * params.scale_softmax_rp_dropout;
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dv); ++i) {
        acc_dv(i) = tdVrdVaccum(i) * params.rp_dropout;
    }
    // Convert acc_dk from fp32 to fp16
    Tensor rdK = pytorch_flash::convert_type<Element>(acc_dk);
    Tensor rdV = pytorch_flash::convert_type<Element>(acc_dv);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);  // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);  // ((Atom,AtomNum), MMA_N, MMA_N)
    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);

    Tensor cdKV = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    pytorch_flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    // constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{}))::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k || binfo.actual_seqlen_q == 0) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = ((bidb * params.h + bidh) * params.seqlen_q_rounded
                                         + (m_block_max - 1) * kBlockM) * params.d_rounded;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q
        + (m_block_max - 1) * kBlockM;
    const index_t row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded
        + (m_block_max - 1) * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  Stride<Int<kHeadDim>, _1>{});
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    // Double buffer for sQ
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                                                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    // sP and sdQ share the same memory so be careful
    Tensor sdQ = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutdQ{});
    Tensor sdPsum = make_tensor(make_smem_ptr(reinterpret_cast<float2 *>((sP.data() + cute::max(size(sP), size(sdQ))).get())),
                                Shape<Int<Kernel_traits::kSmemdPsumCount / 2>>{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = std::conditional_t<
        Is_first,
        typename Kernel_traits::GmemTiledCopydO,
        typename Kernel_traits::GmemTiledCopyQKV
    >;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = std::conditional_t<
        !Seq_parallel,
        typename Kernel_traits::GmemTiledCopydQaccum,
        typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd
    >;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
    // if (cute::thread0()) { print(tdQgdQaccum.layout()); printf("\n"); }
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx < 64) {
    //     printf("tidx = %d, tdQgdQaccum = 0x%p\n", tidx, tdQgdQaccum.data());
    // }

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle); // (MMA, MMA_N, MMA_N)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);   // (MMA, MMA_K, MMA_N)
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);   // (MMA, MMA_N, MMA_N)
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle); // (MMA, MMA_K, MMA_N)

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);                      // (MMA, MMA_N, MMA_N)
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);    // (MMA, MMA_K, MMA_N)

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // auto smem_thr_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp).get_thread_slice(tidx);
    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    // if (cute::thread(0, 0) && n_block == 0) { printf("sK layout: "); print(sK.layout()); printf("\n"); }
    // if (cute::thread(0, 0) && n_block == 0) { print(tSsK.layout()); printf("\n"); }
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    // This has to be tiled_mma_sdp, not tiled_mma_dkv
    // auto smem_thr_copy_PdS = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp).get_thread_slice(tidx);
    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    // if (cute::thread(0, 0) && n_block == 0) { printf("sP layout: "); print(sP.layout()); printf("\n"); }
    // if (cute::thread(0, 0) && n_block == 0) { print(tPsP.layout()); printf("\n"); }
    // if (n_block == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tidx < 64) {
    //     printf("tidx=%d, tPsP = 0x%p\n", tidx, tPsP.data());
    // }
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We'll advance gdQ and gdQaccum before the 1st read/write.
    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.d_rounded;

    int m_block = m_block_max - 1;
    int m_block_min = !Is_causal ? 0 : (n_block * kBlockN) / kBlockM;

    // We might need to exit early and write 0 to dK and dV.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // TODO: what if we're not parallelizing, do we need to compute dot_do_o?
    if (Is_causal && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
          + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
          + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }

    if (Double_buffer && m_block % 2 == 1) {  // Double buffer for sQ
        tQsQ.data() = tQsQ.data() + size(sQ);
        tSsQ.data() = tSsQ.data() + size(sQ);
        tdKsQt.data() = tdKsQt.data() + size(sQ);
    }

    if (!Is_first && !Seq_parallel) { __syncthreads(); }

    if (Kernel_traits::Is_V_in_regs) {
        // Clear the smem tiles to account for predicated off loads
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        pytorch_flash::cp_async_fence();
    }

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    if (!Is_first) {
        // Clear the smem tiles to account for predicated off loads
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((2, 2), MMA_N, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        // Using uint32_t row makes it 10us slower on d=128, not sure why.
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : 0;
    }

    // Tensor tKrK = make_fragment_like(tKsK);
    // // cute::copy(gmem_thr_copy_QKV, tKgK(_, _, _, 0), tKrK);
    // cute::copy(gmem_thr_copy_QKV, tKgK, tKrK);
    // // if (cute::thread(1, 0)) { print(tKrK); }

    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    if (!Kernel_traits::Is_V_in_regs) {
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
    }
    pytorch_flash::cp_async_fence();

    // if (cute::thread0()) { print(tdOgdO.layout()); printf("\n"); print(tdOrdO); print(tdOrO); }
    if (Is_first) {
        cute::copy(tdOrdO, tdOsdO);
        dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, gdPsum, sdPsum,
                                                    Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    }

    if (Kernel_traits::Is_V_in_regs) {
        cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    auto seeds = at::cuda::philox::unpack(params.philox_args);
    unsigned long long seed = std::get<0>(seeds);
    unsigned long long offset = std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32;

    clear(acc_dv);
    clear(acc_dk);

    for (; m_block >= m_block_min; --m_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

        // if (cute::thread0()) { print(sK); }
        // Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
        // #pragma unroll
        // for (int k = 0; k < size<2>(tSrK_copy_view); ++k) {
        //     cute::copy(smem_tiled_copy_KV, tSsK(_, _, k), tSrK_copy_view(_, _, k));
        // }
        // if (cute::thread0()) { print(tSrK); }
        pytorch_flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO,
                            smem_thr_copy_KV);

        // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), pytorch_flash::convert_layout_acc_rowcol(acc_s.layout()));
        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.
        if (!Is_causal) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                pytorch_flash::apply_mask(scores, binfo.actual_seqlen_k,
                                  n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16);
            }
        } else {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements not beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                pytorch_flash::apply_mask_causal(scores, n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                                         binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                         // binfo.actual_seqlen_k, m_block * kBlockM + (tidx / 32) % AtomLayoutMS * 16 + (tidx % 32) / 4,
                                         AtomLayoutMS * 16);
            }
        }
        // if (cute::thread(32, 0)) { print(scores); }
        // Compute the exponential value.
        pytorch_flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        if (Is_dropout) {
            uint32_t warp_id = tidx / 32;
            uint32_t block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            static_assert(MMA_N_SdP % 2 == 0);
            uint32_t block_col_idx = n_block * (kBlockN / 32) + (warp_id / AtomLayoutMS) * (MMA_N_SdP / 2);
            Tensor scores_dropped = make_tensor(scores.data(), pytorch_flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMmaSdP>(scores.layout()));
            pytorch_flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                scores_dropped, params.p_dropout_in_uint8_t, seed, offset,
                block_row_idx, block_col_idx, AtomLayoutMS
            );
        }
        // Convert scores from fp32 to fp16/bf16
        Tensor rP = !Is_dropout
            ? pytorch_flash::convert_type<Element>(scores)
            : pytorch_flash::convert_type_relu<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_N), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_N, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_N, MMA_N) if using m16n8k8.
        Tensor tPrP = make_tensor(rP.data(), pytorch_flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMmaSdP>(rP.layout()));
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);
        // if (cute::thread0()) { print(tPaP); }
        // __syncthreads();
        // if (cute::thread0()) { print(sP); }

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);
        // Tensor acc_dp_reshaped = make_tensor(acc_dp.data(), pytorch_flash::convert_layout_acc_rowcol(acc_dp.layout()));
        // #pragma unroll
        // for (int mi = 0; mi < size<0>(acc_dp_reshaped); ++mi) {
        //     #pragma unroll
        //     for (int ni = 0; ni < size<1>(acc_dp_reshaped); ++ni) {
        //         acc_dp_reshaped(mi, ni) = -dP_sum(mi);
        //     }
        // }

        // if (cute::thread0()) { print(dP_sum); }

        pytorch_flash::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
            acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
        );

        // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                dS(mi, ni) = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
            }
        }
        // if (cute::thread0()) { print(dS); }

        Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
        tdQgdQaccum.data() = tdQgdQaccum.data() + (-int(kBlockM * params.d_rounded));
        if (Is_first || Seq_parallel) {
            clear(acc_dq);
        } else {
            // Reshape acc_dq from (4, 1, 2) to (4, 2, 1) to write to gdQaccum
            Tensor acc_dq_reshaped = make_tensor(acc_dq.data(),
                                                 make_layout(get<0>(acc_dq.layout()),
                                                             get<2>(acc_dq.layout()),
                                                             get<1>(acc_dq.layout())));
            cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, acc_dq_reshaped);
        }

        if (Double_buffer && m_block > m_block_min) {
            // Double buffer for sQ
            const int sQ_offset = m_block % 2 == 0 ? size(sQ) : -size(sQ);
            tQsQ.data() = tQsQ.data() + sQ_offset;
            tSsQ.data() = tSsQ.data() + sQ_offset;
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            pytorch_flash::cp_async_fence();
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        // Convert dS from fp32 to fp16
        Tensor tdSrdS = pytorch_flash::convert_type<Element>(dS_reshaped);
        // if (cute::thread0()) { print(tPrP); }
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);                                          // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        __syncthreads();

        // Layout p_l = tPrP.layout();
        // Tensor tdVrPt = make_tensor(tPrP.data(), make_layout(get<0>(p_l), get<2>(p_l), get<1>(p_l)));
        // pytorch_flash::gemm_A_in_regs(acc_dv, tdVrPt, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_thr_copy_QdOt);
        // Tensor tdKrdSt = make_tensor(tdSrdS.data(), tdVrPt.layout());
        // pytorch_flash::gemm_A_in_regs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_thr_copy_QdOt);
        pytorch_flash::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        // if (cute::thread0() && n_block == 0 && m_block == 0) { print(tdVrPt); }
        // if (cute::thread0()) { print(acc_dv); }

        __syncthreads(); // Need syncthreads since we're writing to the same sdO location

        if (m_block > m_block_min) {
            // Advance gdO
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            if (Is_first) {
                tdOgO.data() = tdOgO.data() + (-int(kBlockM * params.o_row_stride));
                pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ);
                pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ);
            } else {
                pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
                pytorch_flash::cp_async_fence();
            }
        }

        pytorch_flash::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
                    smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);
        // if (cute::thread0()) { print(acc_dq); }

        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            // if (!Is_first && tidx < kBlockM / 2) {
            //     sdPsum(tidx) = recast<float2>(gdPsum)(tidx);
            // if (!Is_first && tidx < kBlockM) {
            //     recast<float>(sdPsum)(tidx) = gdPsum(tidx);
            // }
        }

        if (!Is_last) {
            // Reshape acc_dq from (4, 1, 2) to (4, 2, 1) to write to gdQaccum
            Tensor acc_dq_reshaped = make_tensor(acc_dq.data(),
                                                 make_layout(get<0>(acc_dq.layout()),
                                                             get<2>(acc_dq.layout()),
                                                             get<1>(acc_dq.layout())));
            if (!Seq_parallel) {
                cute::copy(gmem_tiled_copy_dQaccum, acc_dq_reshaped, tdQgdQaccum);
            } else {
                // if (cute::thread0()) { print(acc_dq.layout()); printf("\n"); print(acc_dq_reshaped.layout()); printf("\n"); print(tdQgdQaccum.layout()); printf("\n"); }
                CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
                #pragma unroll
                for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
            // Convert acc_dq from fp32 to fp16
            Tensor rdQ = pytorch_flash::convert_type<Element>(acc_dq);
            Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
        }

        pytorch_flash::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        // if (cute::thread0()) { print(acc_dk); }
        if (Double_buffer) {  // Double buffer for sQ
            tdKsQt.data() = tdKsQt.data() + (m_block % 2 == 0 ? size(sQ) : -size(sQ));
        }
        if (!Double_buffer && m_block > m_block_min) {
            __syncthreads();
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            pytorch_flash::cp_async_fence();
        }

        if (Is_first && m_block > m_block_min) {
            cute::copy(tdOrdO, tdOsdO);
            dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, gdPsum, sdPsum,
                                                        Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
        }

        if (Is_last) {
            __syncthreads();
            Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
            cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
            tdQgdQ.data() = tdQgdQ.data() + (-int(kBlockM * params.dq_row_stride));
            Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
            Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
            #pragma unroll
            for (int m = 0; m < size<1>(tdQgdQ); ++m) {
                if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM) {
                    cute::copy(gmem_tiled_copy_dQ, tdQrdQ(_, m, _), tdQgdQ(_, m, _));
                }
            }
        }

    }

    // Epilogue

    if (Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = pytorch_flash::convert_type<Element>(acc_dk);
    Tensor rdV = pytorch_flash::convert_type<Element>(acc_dv);

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    if (!Is_last) { __syncthreads(); }
    if (Kernel_traits::Is_V_in_regs && !Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
       + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
       + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_N, bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    // constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{}))::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;

    const BlockInfo</*Varlen=*/!Is_even_N> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM, kBlockN));
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    // We'll advance gdKaccum and gdVaccum before the first write.
    const index_t row_offset_dkv_accum = ((bidb * params.h_k + (bidh / params.h_h_k_ratio)) * params.seqlen_k_rounded
                                          + n_block_max * kBlockN) * params.d_rounded;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    // We assume that params.d == kHeadDim for now
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdKaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dk_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  Stride<Int<kHeadDim>, _1>{});
    Tensor gdVaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dv_accum_ptr) + row_offset_dkv_accum),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  Stride<Int<kHeadDim>, _1>{});
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sdO = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                                                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    // Double buffer for sK
    Tensor sV = make_tensor(sK.data() + 2 * size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sdS = make_tensor(sV.data() + size(sV), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sdPsum = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(sdS.data().get())),
                                Shape<Int<kBlockM>>{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dKVaccum;
    auto gmem_thr_copy_dKVaccum = gmem_tiled_copy_dKVaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdKgdKaccum = gmem_thr_copy_dKVaccum.partition_D(gdKaccum);
    Tensor tdVgdVaccum = gmem_thr_copy_dKVaccum.partition_D(gdVaccum);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle); // (MMA, MMA_N, MMA_N)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);   // (MMA, MMA_K, MMA_N)
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);   // (MMA, MMA_N, MMA_N)
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle); // (MMA, MMA_K, MMA_N)

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);                      // (MMA, MMA_N, MMA_N)
    Tensor tdQrKt  = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);    // (MMA, MMA_K, MMA_N)

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M_SdP, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    // This has to be tiled_mma_sdp, not tiled_mma_dkv
    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);

    // TODO: Might need to exit early and write 0 to gdQ.

    pytorch_flash::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    pytorch_flash::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );

    Tensor tQrQ = make_fragment_like(tQgQ);
    pytorch_flash::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );

    int n_block = n_block_max - 1;
    if (n_block % 2 == 1) {
        tKsK.data() = tKsK.data() + size(sK);
        tSsK.data() = tSsK.data() + size(sK);
        tdQsKt.data() = tdQsKt.data() + size(sK);
    }

    pytorch_flash::copy<Is_even_N, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    pytorch_flash::copy<Is_even_N, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((2, 2), MMA_N, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        // Using uint32_t row makes it 10us slower on d=128, not sure why.
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : 0;
    }

    cute::cp_async_fence();

    Tensor dP_sum = make_fragment_like(lse);
    cute::copy(tdOrdO, tdOsdO);
    dot_do_o<Kernel_traits::kGmemThreadsPerRow>(
        tdOrdO, tdOrO, sdPsum, sdPsum,
        Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout
    );
    __syncthreads();
    #pragma unroll
    for (int mi = 0; mi < size(dP_sum); ++mi) { dP_sum(mi) = sdPsum(get<0>(taccScS_row(mi))); }

    auto seeds = at::cuda::philox::unpack(params.philox_args);
    unsigned long long seed = std::get<0>(seeds);
    unsigned long long offset = std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32;

    clear(acc_dq);

    for (; n_block >= 0; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M_SdP, MMA_N)
        clear(acc_s);
        pytorch_flash::cp_async_wait<0>();
        __syncthreads();

        pytorch_flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);
        // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), pytorch_flash::convert_layout_acc_rowcol(acc_s.layout()));
        // We don't need to mask out the elements beyond actual_seqlen_k, because acc_s would
        // be some finite value for those indices. In the end when we multiply with K to get dQ,
        // the corresponding values of K would be 0, so the result would still be correct.
        if (Is_causal && m_block * kBlockM < (n_block + 1) * kBlockN) {
            pytorch_flash::apply_mask_causal(scores, n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                                     binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                     // binfo.actual_seqlen_k, m_block * kBlockM + (tidx / 32) % AtomLayoutMS * 16 + (tidx % 32) / 4,
                                     AtomLayoutMS * 16);
        }
        // Compute the exponential value.
        pytorch_flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        if (Is_dropout) {
            uint32_t warp_id = tidx / 32;
            uint32_t block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            static_assert(MMA_N_SdP % 2 == 0);
            uint32_t block_col_idx = n_block * (kBlockN / 32) + (warp_id / AtomLayoutMS) * (MMA_N_SdP / 2);
            Tensor scores_dropped = make_tensor(scores.data(), pytorch_flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMmaSdP>(scores.layout()));
            pytorch_flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                scores_dropped, params.p_dropout_in_uint8_t, seed, offset,
                block_row_idx, block_col_idx, AtomLayoutMS
            );
        }
        // Convert scores from fp32 to fp16/bf16
        Tensor rP = !Is_dropout
            ? pytorch_flash::convert_type<Element>(scores)
            : pytorch_flash::convert_type_relu<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_N), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_N, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_N, MMA_N) if using m16n8k8.
        Tensor tPrP = make_tensor(rP.data(), pytorch_flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMmaSdP>(rP.layout()));
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);
        pytorch_flash::gemm(acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

        // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                dS(mi, ni) = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
            }
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        // Convert dS from fp32 to fp16
        Tensor tdSrdS = pytorch_flash::convert_type<Element>(dS_reshaped);
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);                                          // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        __syncthreads();

        if (n_block > 0) {
            // Double buffer for sK
            const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
            tKsK.data() = tKsK.data() + sK_offset;
            tSsK.data() = tSsK.data() + sK_offset;
            // Advance gK, gV
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
        clear(acc_dv);
        pytorch_flash::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(acc_dv); }
        tdVgdVaccum.data() = tdVgdVaccum.data() + (-int(kBlockN * params.d_rounded));
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { atomicAdd(&tdVgdVaccum(i), acc_dv(i)); }

        __syncthreads();
        Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
        clear(acc_dk);
        pytorch_flash::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        tdKgdKaccum.data() = tdKgdKaccum.data() + (-int(kBlockN * params.d_rounded));
        #pragma unroll
        for (int i = 0; i < size(acc_dk); ++i) { atomicAdd(&tdKgdKaccum(i), acc_dk(i)); }

        pytorch_flash::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
                    smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);
        // Double buffer for sK
        tdQsKt.data() = tdQsKt.data() + (n_block % 2 == 0 ? size(sK) : -size(sK));

    }

    // Epilogue

    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = pytorch_flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    __syncthreads();

    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);

    Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    pytorch_flash::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_M, bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv(const Params &params) {

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    const int n_block_max = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    if (n_block_max == 1) {
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_even_M, Is_even_K, true, true>(params, bidb, bidh, 0);
    } else {
        // Iterating backward from n_block_max - 1 to 0 might save 1 register
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_even_M, Is_even_K, true, false>(params, bidb, bidh, n_block_max - 1);
        for (int n_block = n_block_max - 2; n_block > 0; n_block--) {
            compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_even_M, Is_even_K, false, false>(params, bidb, bidh, n_block);
        }
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_even_M, Is_even_K, false, true>(params, bidb, bidh, 0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv_seqk_parallel(const Params &params) {

    const int n_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_even_MN, Is_even_K, false, false, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_even_N, bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv_seqq_parallel(const Params &params) {

    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    compute_dq_dk_dv_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_even_N, Is_even_K>(params, bidb, bidh, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace pytorch_flash
