#pragma once


#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_multistage.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma_pipelined.h>

#include <cutlass/gemm/threadblock/mma_multistage.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>

template <typename Mma, int kMaxK>
struct MakeCustomMma;

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB,
    typename SmemIteratorB,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int Stages,
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaMultistage<
        Shape,
        IteratorA,
        SmemIteratorA,
        CacheOpA,
        IteratorB,
        SmemIteratorB,
        CacheOpB,
        ElementC,
        LayoutC,
        Policy,
        Stages,
        SharedMemoryClear>,
    kMaxK> {
  // Reduce the number of stages if we don't need that many
  static int constexpr kStages = kMaxK == std::numeric_limits<int>::max()
      ? Stages
      : std::min(Stages, (kMaxK + int(Shape::kK) - 1) / int(Shape::kK));
  using Mma = cutlass::gemm::threadblock::CustomMmaMultistage<
      Shape,
      IteratorA,
      SmemIteratorA,
      CacheOpA,
      IteratorB,
      SmemIteratorB,
      CacheOpB,
      ElementC,
      LayoutC,
      Policy,
      kStages,
      SharedMemoryClear,
      kMaxK>;
};

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    typename IteratorB,
    typename SmemIteratorB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaPipelined<
        Shape,
        IteratorA,
        SmemIteratorA,
        IteratorB,
        SmemIteratorB,
        ElementC,
        LayoutC,
        Policy>,
    kMaxK> {
  using Mma = cutlass::gemm::threadblock::CustomMmaPipelined<
      Shape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      Policy>;
};
