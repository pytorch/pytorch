#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

#include <cuda_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/vector.h>
#include <cutlass/numeric_types.h>

#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/threadblock/epilogue_smem_accumulator.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/platform/platform.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/vector_iterator.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/epilogue_pipelined.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/epilogue_predicated_tile_iterator.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/find_default_mma.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/mma_from_smem.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h>

#include <cinttypes>

using namespace gemm_kernel_utils;

namespace {
template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSm() {
  bool is_half = !std::is_same<scalar_t, float>::value;
  if (Arch::kMinComputeCapability >= 80) {
    return is_half ? 12 : 8;
  }
  return 8;
}
} // namespace

template <
    // which arch we target (eg `cutlass::arch::Sm80`)
    typename ArchTag_,
    // input/output type
    typename scalar_t_,
    // run optimized kernel because memory accesses will be aligned
    bool kIsAligned_,
    // upperbound on `max(value.shape[-1], query.shape[-1])`
    int kMaxK = std::numeric_limits<int>::max()>
struct AttentionBackwardKernel {
  using scalar_t = scalar_t_;
  using output_t = scalar_t;
  using lse_scalar_t = float;
  using accum_t = float;
  using ArchTag = ArchTag_;
  static constexpr bool kIsAligned = kIsAligned_;

  struct Params {
    // Input tensors
    scalar_t* query_ptr; // [num_queries, head_dim]
    scalar_t* key_ptr; // [num_keys, head_dim]
    scalar_t* value_ptr; // [num_keys, head_dim_value]
    lse_scalar_t* logsumexp_ptr; // [num_queries]
    scalar_t* output_ptr; // [num_queries, head_dim_value]
    scalar_t* grad_output_ptr; // [num_queries, head_dim_value]
    accum_t* delta_ptr; // [num_queries]

    // Output tensors
    scalar_t* grad_query_ptr; // [num_queries, head_dim]
    scalar_t* grad_key_ptr; // [num_keys, head_dim]
    scalar_t* grad_value_ptr; // [num_keys, head_dim_value]

    // Dimensions/strides
    int32_t head_dim;
    int32_t head_dim_value;
    int32_t num_queries;
    int32_t num_keys;
    int32_t num_batches;
    bool causal;

    __device__ void advance_batches(int32_t batch_id) {
      constexpr int32_t kAlignLSE = 32; // block size of backward
      auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

      query_ptr += batch_id * head_dim * num_queries;
      key_ptr += batch_id * head_dim * num_keys;
      value_ptr += batch_id * head_dim_value * num_keys;
      logsumexp_ptr += batch_id * lse_dim;
      output_ptr += batch_id * head_dim_value * num_queries;
      grad_output_ptr += batch_id * head_dim_value * num_queries;
      delta_ptr += batch_id * num_queries;

      grad_query_ptr += batch_id * head_dim * num_queries;
      grad_key_ptr += batch_id * head_dim * num_keys;
      grad_value_ptr += batch_id * head_dim_value * num_keys;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(1, 1, num_batches);
    }
    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  // Blocks & grid
  static constexpr bool kSupports64x128 =
      ArchTag::kMinComputeCapability >= 80 ||
      (ArchTag::kMinComputeCapability >= 70 &&
       cutlass::sizeof_bits<scalar_t>::value <= 16);
  static constexpr int64_t kWarpSize = 32;
  static constexpr int64_t kBlockSizeI =
      kSupports64x128 && kMaxK > 64 ? 128 : 64;
  static constexpr int64_t kBlockSizeJ = 64;
  static constexpr int64_t kNumWarpsPerBlock =
      (kBlockSizeI * kBlockSizeJ) / (32 * 32);

  // If this is true, we store and accumulate dK/dV in RF
  // rather than going back to gmem everytime
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value <= 16;
  static constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;
  static constexpr bool kPreloadMmas =
      kIsHalf && ArchTag::kMinComputeCapability >= 80 && kOutputInRF;
  static constexpr bool kPrologueQK = kPreloadMmas;
  static constexpr bool kPrologueGV = kPreloadMmas;
  static constexpr bool kPrologueDOV = kPreloadMmas;
  static constexpr bool kPrologueGQ = kPreloadMmas;
  static constexpr bool kPrologueGK = kPreloadMmas;

  // Compute delta for the f16 kernels
  // TODO: Figure out why it's slower on the f32 kernels
  // (something due to RF pressure?)
  // TODO: Remove condition on `kOutputInRF` - this is needed to work
  // around a compiler bug on V100, not exactly sure why but I spent
  // too much time on this already. Reproducible with
  // (B, Mq, Mkv, K) = (1, 1, 1, 136) for instance
  static constexpr bool kKernelComputesDelta =
      kIsHalf && (kOutputInRF || ArchTag::kMinComputeCapability != 70);

  // Launch bounds
  static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int64_t kMinBlocksPerSm =
      getWarpsPerSm<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  using GemmType = DefaultGemmType<ArchTag, scalar_t>;
  using DefaultConfig =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          typename GemmType::OpClass,
          ArchTag,
          scalar_t,
          scalar_t,
          scalar_t, // ElementC
          accum_t // ElementAccumulator
          >;
  static constexpr auto kOptimalAlignement =
      std::max(DefaultConfig::kAlignmentA, DefaultConfig::kAlignmentB);
  static constexpr auto kMinimumAlignment = GemmType::kMinimumAlignment;

  struct MatmulQK {
    /*
    attn_T = k_j @ q_i.transpose(-2, -1) # matmul
    attn_T = (attn_T - logsumexp[i_start:i_end].unsqueeze(1).transpose(-2,
    -1)).exp() # epilogue

    with attn_T.shape = (kBlockSizeJ, kBlockSizeI)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        accum_t, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator,
        false, // AccumulatorsInRowMajor = false,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma =
        typename MakeCustomMma<typename DefaultMma::ThreadblockMma, kMaxK>::Mma;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using ScalingCoefsUpdater = typename DefaultAttentionScalingCoefsUpdater<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Updater;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradV {
    /*
    grad_v[j_start:j_end] += attn_T @ do_i # matmul

    Dimensions: (kBlockSizeJ * kNumWarpsPerBlock, kBlockSizeI, K)
    (we might need to iterate multiple times on K)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            typename MatmulQK::AccumulatorSharedStorage>;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
  };

  struct MatmulDOIVJ {
    /*
    doi_t_vj = do_i @ v_j.transpose(-2, -1) # matmul
    tmp = (doi_t_vj - Di.unsqueeze(1)) * attn # inplace / epilogue?
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        accum_t, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator,
        false, // AccumulatorsInRowMajor = false,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma =
        typename MakeCustomMma<typename DefaultMma::ThreadblockMma, kMaxK>::Mma;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradQ {
    // grad_q <- tmp @ k_j
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            typename MatmulDOIVJ::AccumulatorSharedStorage>;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
  };
  struct MatmulGradK {
    // grad_k <- tmp.transpose(-2, -1) @ q_i
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            typename MatmulQK::AccumulatorSharedStorage>;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
  };

  // See https://fburl.com/gsheet/l5bltspl
  // for an illustration of how smem is used
  struct SharedStoragePrologue {
    struct {
      cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
      typename MatmulQK::Mma::SharedStorageA mm_qk_k;
    } persistent;
    union {
      struct {
        // p1 - after Q.K / dV / dO.V
        typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;

        union {
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };

        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
      } p1;

      struct {
        // p2 - dQ
        typename MatmulQK::AccumulatorSharedStorage
            attn_shared_storage; // (from p1)
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::Mma::SharedStorage mm_gradQ; // (preload)
        typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;

        typename MatmulDOIVJ::AccumulatorSharedStorage doivj_shared_storage;
      } p2;

      struct {
        // p3 - after last iteration on dQ's epilogue / dK
        typename MatmulQK::AccumulatorSharedStorage
            attn_shared_storage; // (from p1)
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::DefaultEpilogue::SharedStorage
            gradQ_epilogue_lastIter;

        typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
      } p3;

      struct {
        // p4 - after last iteration on dK's epilogue / preload next K.Q_t
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;

        // If we reach end of current key, dump RF->gmem with "final" epilogues
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } p4;
    };
    static void print_size() {
      // Field size
#define FSZ(f) int((sizeof(((SharedStorage*)0)->f)))

      printf("Total smem: %d bytes\n", int(sizeof(SharedStorage)));
      printf("  persistent: %db\n", FSZ(persistent));
      printf("    mm_qk_k: %db\n", FSZ(persistent.mm_qk_k));
      printf("  p1: %db\n", FSZ(p1));
      printf("    attn_shared_storage: %db\n", FSZ(p1.attn_shared_storage));
      printf("    mm_gradV: %db\n", FSZ(p1.mm_gradV));
      printf("    gradV_epilogue: %db\n", FSZ(p1.gradV_epilogue));
      printf("    mm_doivj: %db\n", FSZ(p1.mm_doivj));
      printf("  p2: %db\n", FSZ(p2));
      printf("    mm_gradK: %db\n", FSZ(p2.mm_gradK));
      printf("    mm_gradQ: %db\n", FSZ(p2.mm_gradQ));
      printf("    gradQ_epilogue: %db\n", FSZ(p2.gradQ_epilogue));
      printf("    doivj_shared_storage: %db\n", FSZ(p2.doivj_shared_storage));
      printf("  p3: %db\n", FSZ(p3));
      printf("  p4: %db\n", FSZ(p4));
      printf("    mm_qk_q: %db\n", FSZ(p4.mm_qk_q));
      printf("    gradK_epilogue_final: %db\n", FSZ(p4.gradK_epilogue_final));
      printf("    gradV_epilogue_final: %db\n", FSZ(p4.gradV_epilogue_final));
    }
// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(persistent, di)
    FIELD(persistent, mm_qk_k)
    FIELD(p1, attn_shared_storage)
    FIELD(p1, mm_gradV)
    FIELD(p1, gradV_epilogue)
    FIELD(p1, mm_doivj)
    FIELD(p2, mm_gradK)
    FIELD(p2, mm_gradQ)
    FIELD(p2, gradQ_epilogue)
    FIELD(p2, doivj_shared_storage)
    FIELD(p3, gradQ_epilogue_lastIter)
    FIELD(p3, gradK_epilogue)
    FIELD(p4, mm_qk_q)
    FIELD(p4, gradK_epilogue_final)
    FIELD(p4, gradV_epilogue_final)
  };

  struct SharedStorageNoPrologue {
    struct {
      cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
    } persistent;
    union {
      struct {
        // p1 - Q.K matmul
        typename MatmulQK::Mma::SharedStorageA mm_qk_k;
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;
      } p1;

      struct {
        // p2 - compute gradV
        typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        union {
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };
      } p2;

      struct {
        // p3 - DO.V matmul
        typename MatmulQK::AccumulatorSharedStorage
            attn_shared_storage; // (from p2)
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
      } p3;

      struct {
        // p4 - compute gradQ
        typename MatmulQK::AccumulatorSharedStorage
            attn_shared_storage; // (from p2)
        typename MatmulDOIVJ::AccumulatorSharedStorage doivj_shared_storage;
        union {
          typename MatmulGradQ::Mma::SharedStorage mm_gradQ;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage
              gradQ_epilogue_lastIter;
        };
      } p4;

      struct {
        // p5 - compute gradK
        typename MatmulQK::AccumulatorSharedStorage
            attn_shared_storage; // (from p2)
        typename MatmulDOIVJ::AccumulatorSharedStorage
            doivj_shared_storage; // (from p4)
        union {
          typename MatmulGradK::Mma::SharedStorage mm_gradK;
          typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
        };
      } p5;

      struct {
        // p6 - store RF accumulated into gmem
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } p6;
    };
    static void print_size() {
#define FIELD_SIZEOF(f) int((sizeof(((SharedStorage*)0)->f)))
      printf("Total smem: %d bytes\n", int(sizeof(SharedStorage)));
      printf("  persistent: %db\n", FIELD_SIZEOF(persistent));
      printf("  p1: %db\n", FIELD_SIZEOF(p1));
      printf("  p2: %db\n", FIELD_SIZEOF(p2));
      printf("  p3: %db\n", FIELD_SIZEOF(p3));
      printf("  p4: %db\n", FIELD_SIZEOF(p4));
      printf("  p5: %db\n", FIELD_SIZEOF(p5));
      printf("  p6: %db\n", FIELD_SIZEOF(p6));
    }
// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(persistent, di)
    FIELD(p1, mm_qk_k)
    FIELD(p1, mm_qk_q)
    FIELD(p2, attn_shared_storage)
    FIELD(p2, mm_gradV)
    FIELD(p2, gradV_epilogue)
    FIELD(p3, mm_doivj)
    FIELD(p4, doivj_shared_storage)
    FIELD(p4, mm_gradQ)
    FIELD(p4, gradQ_epilogue)
    FIELD(p4, gradQ_epilogue_lastIter)
    FIELD(p5, mm_gradK)
    FIELD(p5, gradK_epilogue)
    FIELD(p6, gradK_epilogue_final)
    FIELD(p6, gradV_epilogue_final)
  };

  using SharedStorage = typename std::conditional<
      kPreloadMmas,
      SharedStoragePrologue,
      SharedStorageNoPrologue>::type;

  struct OutputFragments {
    typename MatmulGradV::Mma::FragmentC gradV;
    typename MatmulGradK::Mma::FragmentC gradK;

    __device__ __forceinline__ void clear() {
      gradV.clear();
      gradK.clear();
    }
  };

  static void __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.key_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.value_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.output_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.grad_output_ptr, kMinimumAlignment);
    TORCH_CHECK(
        p.head_dim % kMinimumAlignment == 0,
        "query/key is not correctly aligned");
    TORCH_CHECK(
        p.head_dim_value % kMinimumAlignment == 0,
        "value is not correctly aligned");
  }

  static __device__ void kernel(Params& p_) {
    // Hint to nvcc to store points & tensor shapes in registers
    // as we use them a lot
    register const Params p = p_;

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);

    auto getQueryStart = [&](int32_t key_start) {
      if (p.causal) {
        return key_start;
      }
      return 0;
    };

    if (kPrologueQK) {
      prologueQkNextIteration<true>(shared_storage, p, 0, 0);
    }

    // Computes (dO*out).sum(-1) and writes it to `p.delta_ptr`
    if (kKernelComputesDelta) {
      constexpr int kOptimalElements =
          128 / cutlass::sizeof_bits<scalar_t>::value;
      if (p.head_dim_value % kOptimalElements == 0) {
        for (int query_start = 0; query_start < p.num_queries;
             query_start += kBlockSizeI) {
          computeDelta<kOptimalElements>(p, query_start);
        }
      } else {
        for (int query_start = 0; query_start < p.num_queries;
             query_start += kBlockSizeI) {
          computeDelta<1>(p, query_start);
        }
      }
      __syncthreads();
    }

    OutputFragments output_frags;
    int32_t key_start = 0;
    int32_t key_end = p.num_keys / kBlockSizeJ * kBlockSizeJ;
    for (; key_start < key_end; key_start += kBlockSizeJ) {
      output_frags.clear();
      int32_t query_start = getQueryStart(key_start);
      int32_t query_end = query_start +
          (p.num_queries - query_start) / kBlockSizeI * kBlockSizeI;
      for (; query_start < query_end; query_start += kBlockSizeI) {
        processBlockIJ<true>(
            shared_storage, output_frags, p, query_start, key_start);
      }
      // last (partial) query
      if (query_start < p.num_queries) {
        processBlockIJ<false>(
            shared_storage, output_frags, p, query_start, key_start);
      }
      if (kOutputInRF) {
        writeFragsToGmem<true>(shared_storage, output_frags, p, key_start);
      }
      __syncthreads();
    }
    // Last (partial) key
    if (key_start != p.num_keys) {
      output_frags.clear();
      for (int32_t query_start = getQueryStart(key_start);
           query_start < p.num_queries;
           query_start += kBlockSizeI) {
        processBlockIJ<false>(
            shared_storage, output_frags, p, query_start, key_start);
      }
      if (kOutputInRF) {
        writeFragsToGmem<false>(shared_storage, output_frags, p, key_start);
      }
    }
  }

  static __device__ __forceinline__ void loadDi(
      cutlass::Array<accum_t, kBlockSizeI>& di,
      Params const& p,
      int32_t query_start) {
    int32_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    if (thread_id < kBlockSizeI) {
      accum_t di_rf = accum_t(0);
      if (query_start + thread_id < p.num_queries) {
        di_rf = p.delta_ptr[query_start + thread_id];
      }
      di[thread_id] = di_rf;
    }
  }

  template <bool skipBoundsChecks>
  static __device__ __forceinline__ void processBlockIJ(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params const& p,
      int32_t query_start,
      int32_t key_start) {
    cutlass::MatrixCoord no_offset{0, 0};
    accum_t scale = accum_t(1.0 / std::sqrt(float(p.head_dim)));
    int32_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;
    __syncthreads();
    loadDi(shared_storage.di(), p, query_start);

    int32_t num_queries_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kN
        : std::min(
              (int32_t)MatmulQK::Mma::Shape::kN, p.num_queries - query_start);
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : std::min((int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start);

    auto prologueGradV = [&](int col) {
      typename MatmulGradV::Mma::IteratorB iterator_dO(
          {int32_t(p.head_dim_value)},
          p.grad_output_ptr + query_start * p.head_dim_value + col,
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);
      MatmulGradV::Mma::prologue(
          shared_storage.mm_gradV(),
          iterator_dO,
          thread_id,
          num_queries_in_block);
    };
    auto prologueGradQ = [&](int col) {
      typename MatmulGradQ::Mma::IteratorB iterator_K(
          {int32_t(p.head_dim)},
          p.key_ptr + key_start * p.head_dim + col,
          {num_keys_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      MatmulGradQ::Mma::prologue(
          shared_storage.mm_gradQ(), iterator_K, thread_id, num_keys_in_block);
    };
    auto prologueGradK = [&](int col) {
      typename MatmulGradK::Mma::IteratorB iterator_Q(
          {int32_t(p.head_dim)},
          p.query_ptr + query_start * p.head_dim + col,
          {num_queries_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      MatmulGradK::Mma::prologue(
          shared_storage.mm_gradK(),
          iterator_Q,
          thread_id,
          num_queries_in_block);
    };
    auto prologueDOV = [&]() {
      typename MatmulDOIVJ::Mma::IteratorA iterator_A(
          {int32_t(p.head_dim_value)},
          p.grad_output_ptr + query_start * p.head_dim_value,
          {num_queries_in_block, p.head_dim_value},
          thread_id,
          no_offset);
      typename MatmulDOIVJ::Mma::IteratorB iterator_B(
          {int32_t(p.head_dim_value)},
          p.value_ptr + key_start * p.head_dim_value,
          {p.head_dim_value, num_keys_in_block},
          thread_id,
          no_offset);
      MatmulDOIVJ::Mma::prologue(
          shared_storage.mm_doivj(),
          iterator_A,
          iterator_B,
          thread_id,
          p.head_dim_value);
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulQK
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulQK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block,
          num_queries_in_block,
          p.head_dim // k
      );

      // k_j
      typename Mma::IteratorA iterator_A(
          {int32_t(p.head_dim)},
          p.key_ptr + key_start * p.head_dim,
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // q_i.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim)},
          p.query_ptr + query_start * p.head_dim,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.mm_qk_k(),
          shared_storage.mm_qk_q(),
          thread_id,
          warp_id,
          lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma.set_prologue_done(kPrologueQK);
      mma.set_zero_outside_bounds(!skipBoundsChecks);
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      accum = cutlass::multiplies<typename Mma::FragmentC>()(scale, accum);

      // Epilogue: add LSE + exp and store that to our shared memory buffer
      // shmem <- (matmul_result -
      // logsumexp[i_start:i_end].unsqueeze(1)).exp()
      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};
      // Apply mask
      if (p.causal) {
        auto lane_offset = MatmulQK::ScalingCoefsUpdater::get_lane_offset(
            lane_id, warp_id, output_tile_coords);
        MatmulQK::ScalingCoefsUpdater::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              // (don't forget we are transposed!)
              if (accum_m > accum_n + query_start - key_start) {
                accum[idx] = -std::numeric_limits<accum_t>::infinity();
              }
            },
            [&](int accum_m) {});
      }

      __syncthreads();
      if (kPrologueGV) {
        prologueGradV(0);
      }
      if (kPrologueDOV) {
        prologueDOV();
      }
      MatmulQK::B2bGemm::accumApplyLSEToSmem(
          shared_storage.attn_shared_storage(),
          accum,
          p.logsumexp_ptr + query_start,
          problem_size.n(),
          thread_id,
          warp_id,
          lane_id,
          output_tile_coords);
      __syncthreads();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradV matmul
    //
    // grad_v[j_start:j_end] += attn_T @ do_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    for (int col = 0; col < (kOutputInRF ? 1 : p.head_dim_value);
         col += MatmulGradV::ThreadblockShape::kN) {
      using Mma = typename MatmulGradV::Mma;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block, p.head_dim_value - col, num_queries_in_block);
      auto createEpilogueIter = [&]() {
        return typename MatmulGradV::OutputTileIterator(
            typename MatmulGradV::OutputTileIterator::Params{p.head_dim_value},
            p.grad_value_ptr + key_start * p.head_dim_value + col,
            {num_keys_in_block, p.head_dim_value - col},
            thread_id);
      };
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim_value)},
          p.grad_output_ptr + query_start * p.head_dim_value + col,
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.mm_gradV(),
          shared_storage.attn_shared_storage(),
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      if (!kOutputInRF) {
        createEpilogueIter().prefetch_all();
        output_frags.gradV.clear();
      }
      mma.set_prologue_done(kPrologueGV);

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();

      mma(gemm_k_iterations,
          output_frags.gradV,
          iterator_B,
          output_frags.gradV);
      __syncthreads();
      if (kPrologueGV &&
          col + MatmulGradV::ThreadblockShape::kN < p.head_dim_value) {
        prologueGradV(col + MatmulGradV::ThreadblockShape::kN);
      }

      if (!kOutputInRF) {
        accumulateInGmem<MatmulGradV>(
            shared_storage.gradV_epilogue(),
            output_frags.gradV,
            createEpilogueIter(),
            query_start == 0 || (p.causal && query_start == key_start));
      }
    }
    __syncthreads();
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulDOIVJ
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulDOIVJ::Mma;
      // do_i
      typename Mma::IteratorA iterator_A(
          {int32_t(p.head_dim_value)},
          p.grad_output_ptr + query_start * p.head_dim_value,
          {num_queries_in_block, p.head_dim_value},
          thread_id,
          no_offset);

      // v_j.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim_value)},
          p.value_ptr + key_start * p.head_dim_value,
          {p.head_dim_value, num_keys_in_block},
          thread_id,
          no_offset);

      Mma mma(shared_storage.mm_doivj(), thread_id, warp_id, lane_id);
      mma.set_prologue_done(kPrologueDOV);
      mma.set_zero_outside_bounds(!skipBoundsChecks);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (p.head_dim_value + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();
      if (kPrologueGQ) {
        prologueGradQ(0);
      }
      if (kPrologueGK) {
        prologueGradK(0);
      }

      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};
      // TODO: This must be terribly inefficient. There must be a better way
      // tmp [RF] <- (accum [RF] - Di [smem] ) * attn_T.T [smem]
      // attn_shared_storage  [smem] <- tmp.T
      // doivj_shared_storage [smem] <- tmp
      {
        using RegistersIter = typename DefaultAttentionScalingCoefsUpdater<
            typename Mma::Operator::IteratorC,
            typename MatmulDOIVJ::DefaultMma::MmaCore::ElementC,
            kWarpSize>::Updater;
        auto lane_offset = RegistersIter::get_lane_offset(
            lane_id, warp_id, output_tile_coords);
        auto attn_T = shared_storage.attn_shared_storage().accum_ref();
        accum_t current_di;
        typename Mma::FragmentC fragment_attn, fragment_di;
        RegistersIter::iterateRows(
            lane_offset,
            [&](int accum_m) { current_di = shared_storage.di()[accum_m]; },
            [&](int accum_m, int accum_n, int idx) {
              // TODO: Otherwise we can get nans as we
              // might have infs here (only seen on f16 tho)
              if (skipBoundsChecks ||
                  (accum_m < num_queries_in_block &&
                   accum_n < num_keys_in_block)) {
                fragment_attn[idx] = attn_T.at({accum_n, accum_m});
              } else {
                fragment_attn[idx] = 0;
              }
              fragment_di[idx] = current_di;
            },
            [&](int accum_m) {

            });
        accum = (accum - fragment_di) * fragment_attn * scale;
        // attn <- attn_T.T
        RegistersIter::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              // How does this even work?! We need to change the layout
              attn_T.at({accum_n, accum_m}) = scalar_t(accum[idx]);
            },
            [&](int accum_m) {});
      }

      MatmulDOIVJ::B2bGemm::accumToSmem(
          shared_storage.doivj_shared_storage(),
          accum,
          lane_id,
          output_tile_coords);
      __syncthreads();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradQ matmul
    //
    // grad_q[i_start:i_end] += tmp @ k_j
    /////////////////////////////////////////////////////////////////////////////////////////////////
    for (int col = 0; col < p.head_dim;
         col += MatmulGradQ::ThreadblockShape::kN) {
      using Mma = typename MatmulGradQ::Mma;

      cutlass::gemm::GemmCoord problem_size(
          num_queries_in_block,
          false ? MatmulGradQ::ThreadblockShape::kN : p.head_dim - col,
          num_keys_in_block);
      auto createEpilogueIter = [&]() {
        return typename MatmulGradQ::OutputTileIterator(
            typename MatmulGradQ::OutputTileIterator::Params{p.head_dim},
            p.grad_query_ptr + query_start * p.head_dim + col,
            {problem_size.m(), problem_size.n()},
            thread_id);
      };

      // k_j
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim)},
          p.key_ptr + key_start * p.head_dim + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.mm_gradQ(),
          shared_storage.doivj_shared_storage(),
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Start prefetching output tile now to make the epilogue faster
      createEpilogueIter().prefetch_all();
      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();
      mma.set_prologue_done(kPrologueGQ);
      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();
      bool isLastColumn = col + MatmulGradQ::ThreadblockShape::kN >= p.head_dim;
      if (kPrologueGQ && !isLastColumn) {
        prologueGradQ(col + MatmulGradQ::ThreadblockShape::kN);
      }

      // Output results
      typename MatmulGradQ::OutputTileIterator output_it = createEpilogueIter();
      DISPATCH_BOOL(
          key_start == 0, kIsFirst, ([&]() {
            using DefaultEpilogue = typename MatmulGradQ::DefaultEpilogue;
            using DefaultOutputOp = typename MatmulGradQ::DefaultOutputOp;
            static constexpr auto ScaleType = kIsFirst
                ? cutlass::epilogue::thread::ScaleType::Nothing
                : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
            using EpilogueOutputOp =
                typename cutlass::epilogue::thread::LinearCombination<
                    typename DefaultOutputOp::ElementOutput,
                    DefaultOutputOp::kCount,
                    typename DefaultOutputOp::ElementAccumulator,
                    typename DefaultOutputOp::ElementCompute,
                    ScaleType>;
            using Epilogue =
                typename cutlass::epilogue::threadblock::EpiloguePipelined<
                    typename DefaultEpilogue::Shape,
                    typename Mma::Operator,
                    DefaultEpilogue::kPartitionsK,
                    typename MatmulGradQ::OutputTileIterator,
                    typename DefaultEpilogue::AccumulatorFragmentIterator,
                    typename DefaultEpilogue::WarpTileIterator,
                    typename DefaultEpilogue::SharedLoadIterator,
                    EpilogueOutputOp,
                    typename DefaultEpilogue::Padding,
                    DefaultEpilogue::kFragmentsPerIteration,
                    true // IterationsUnroll
                    >;
            EpilogueOutputOp rescale({1, 1});
            Epilogue epilogue(
                isLastColumn ? shared_storage.gradQ_epilogue_lastIter()
                             : shared_storage.gradQ_epilogue(),
                thread_id,
                warp_id,
                lane_id);
            epilogue(rescale, output_it, accum, output_it);
          }));
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradK matmul
    //
    // grad_k[i_start:i_end] += tmp.transpose(-2, -1) @ q_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    for (int col = 0; col < (kOutputInRF ? 1 : p.head_dim);
         col += MatmulGradK::ThreadblockShape::kN) {
      using Mma = typename MatmulGradK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block,
          false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col,
          num_queries_in_block);
      auto createEpilogueIter = [&]() {
        return typename MatmulGradK::OutputTileIterator(
            typename MatmulGradK::OutputTileIterator::Params{p.head_dim},
            p.grad_key_ptr + key_start * p.head_dim + col,
            {num_keys_in_block,
             false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col},
            thread_id);
      };

      // q_i
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim)},
          p.query_ptr + query_start * p.head_dim + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.mm_gradK(),
          shared_storage.attn_shared_storage(), // storing tmp.T
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      if (!kOutputInRF) {
        output_frags.gradK.clear();
        createEpilogueIter().prefetch_all();
      }

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();
      mma.set_prologue_done(kPrologueGK);
      mma(gemm_k_iterations,
          output_frags.gradK,
          iterator_B,
          output_frags.gradK);
      __syncthreads();
      bool isLastColumn = col + MatmulGradK::ThreadblockShape::kN >= p.head_dim;
      if (kPrologueGK && !isLastColumn) {
        prologueGradK(col + MatmulGradK::ThreadblockShape::kN);
      }

      if (kPrologueQK && isLastColumn) {
        int32_t next_query = query_start + kBlockSizeI;
        int32_t next_key = key_start;
        if (next_query >= p.num_queries) {
          next_key = key_start + kBlockSizeJ;
          next_query = p.causal ? next_key : 0;
        }
        DISPATCH_BOOL(next_key != key_start, kForceReloadK, ([&]() {
                        prologueQkNextIteration<kForceReloadK>(
                            shared_storage, p, next_query, next_key);
                      }));
      }

      // Output results
      if (!kOutputInRF) {
        accumulateInGmem<MatmulGradK>(
            isLastColumn ? shared_storage.gradK_epilogue_final()
                         : shared_storage.gradK_epilogue(),
            output_frags.gradK,
            createEpilogueIter(),
            query_start == 0 || (p.causal && query_start == key_start));
      }
    }
  }

  template <bool kForceReloadK>
  static CUTLASS_DEVICE void prologueQkNextIteration(
      SharedStorage& shared_storage,
      Params const& p,
      int32_t query_start,
      int32_t key_start) {
    if (query_start >= p.num_queries || key_start >= p.num_keys) {
      return;
    }

    static constexpr bool kReloadK =
        kForceReloadK || !MatmulQK::Mma::kSmemContainsEntireMat;
    auto thread_id = get_thread_id();
    typename MatmulQK::Mma::IteratorA iterator_A(
        {int32_t(p.head_dim)},
        p.key_ptr + key_start * p.head_dim,
        {p.num_keys - key_start, p.head_dim},
        thread_id,
        cutlass::MatrixCoord{0, 0});

    typename MatmulQK::Mma::IteratorB iterator_B(
        {int32_t(p.head_dim)},
        p.query_ptr + query_start * p.head_dim,
        {p.head_dim, p.num_queries - query_start},
        thread_id,
        cutlass::MatrixCoord{0, 0});

    MatmulQK::Mma::prologue<kReloadK, true>(
        shared_storage.mm_qk_k(),
        shared_storage.mm_qk_q(),
        iterator_A,
        iterator_B,
        thread_id,
        p.head_dim);
  }

  template <bool skipBoundsChecks>
  static __device__ __forceinline__ void writeFragsToGmem(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params const& p,
      int32_t key_start) {
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : std::min((int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start);
    typename MatmulGradV::OutputTileIterator outputV_it(
        typename MatmulGradV::OutputTileIterator::Params{p.head_dim_value},
        p.grad_value_ptr + key_start * p.head_dim_value,
        {num_keys_in_block, p.head_dim_value},
        get_thread_id());
    accumulateInGmem<MatmulGradV>(
        shared_storage.gradV_epilogue_final(),
        output_frags.gradV,
        outputV_it,
        true);

    typename MatmulGradK::OutputTileIterator outputK_it(
        typename MatmulGradK::OutputTileIterator::Params{p.head_dim},
        p.grad_key_ptr + key_start * p.head_dim,
        {num_keys_in_block,
         false ? MatmulGradK::ThreadblockShape::kN : p.head_dim},
        get_thread_id());
    accumulateInGmem<MatmulGradK>(
        shared_storage.gradK_epilogue_final(),
        output_frags.gradK,
        outputK_it,
        true);
  }

  template <typename MatmulT>
  static __device__ __forceinline__ void accumulateInGmem(
      typename MatmulT::DefaultEpilogue::SharedStorage& epilogue_smem,
      typename MatmulT::Mma::FragmentC const& accum,
      typename MatmulT::OutputTileIterator output_it,
      bool first) {
    using DefaultEpilogue = typename MatmulT::DefaultEpilogue;
    using DefaultOutputOp = typename MatmulT::DefaultOutputOp;
    using Mma = typename MatmulT::Mma;
    DISPATCH_BOOL(
        first, kIsFirst, ([&]() {
          static constexpr auto ScaleType = kIsFirst
              ? cutlass::epilogue::thread::ScaleType::Nothing
              : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
          using EpilogueOutputOp =
              typename cutlass::epilogue::thread::LinearCombination<
                  typename DefaultOutputOp::ElementOutput,
                  DefaultOutputOp::kCount,
                  typename DefaultOutputOp::ElementAccumulator,
                  typename DefaultOutputOp::ElementCompute,
                  ScaleType>;
          using Epilogue =
              typename cutlass::epilogue::threadblock::EpiloguePipelined<
                  typename DefaultEpilogue::Shape,
                  typename Mma::Operator,
                  DefaultEpilogue::kPartitionsK,
                  typename MatmulT::OutputTileIterator,
                  typename DefaultEpilogue::AccumulatorFragmentIterator,
                  typename DefaultEpilogue::WarpTileIterator,
                  typename DefaultEpilogue::SharedLoadIterator,
                  EpilogueOutputOp,
                  typename DefaultEpilogue::Padding,
                  DefaultEpilogue::kFragmentsPerIteration,
                  true // IterationsUnroll
                  >;
          EpilogueOutputOp rescale({1, 1});
          Epilogue epilogue(
              epilogue_smem, get_thread_id(), get_warp_id(), get_lane_id());
          epilogue(rescale, output_it, accum, output_it);
        }));
  }

  template <int kElementsPerAccess>
  static __device__ void computeDelta(Params const& p, int32_t query_start) {
    // Each thread computes one value for Delta
    // Depending on warp configuration, we might have multiple
    // threads of the same warp working on the same row
    using AccessType = cutlass::Array<scalar_t, kElementsPerAccess>;
    static_assert(kNumThreads >= kBlockSizeI, "");
    static constexpr int kNumThreadsPerLine = kNumThreads / kBlockSizeI;
    int16_t thread_id = get_thread_id();

    int16_t laneFirstCol =
        kElementsPerAccess * (get_lane_id() % kNumThreadsPerLine);
    int16_t laneRow = thread_id / kNumThreadsPerLine;
    bool rowPred = (query_start + laneRow) < p.num_queries;
    bool pred = rowPred;

    const __restrict__ AccessType* grad_output_ptr =
        reinterpret_cast<const __restrict__ AccessType*>(
            p.grad_output_ptr + (query_start + laneRow) * p.head_dim_value +
            laneFirstCol);
    const __restrict__ AccessType* output_ptr =
        reinterpret_cast<const __restrict__ AccessType*>(
            p.output_ptr + (query_start + laneRow) * p.head_dim_value +
            laneFirstCol);

    static constexpr int64_t kMaxIters =
        kMaxK / (kElementsPerAccess * kNumThreadsPerLine);
    constexpr int kPipelineStages = 2;
    accum_t delta_value = accum_t(0);
    using GlobalLoad =
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>;
    AccessType frag_grad_output[kPipelineStages];
    AccessType frag_output[kPipelineStages];

    auto loadAndIncrement = [&](int ld_pos, bool is_valid) {
      frag_grad_output[ld_pos].clear();
      frag_output[ld_pos].clear();
      GlobalLoad(frag_grad_output[ld_pos], grad_output_ptr, is_valid);
      GlobalLoad(frag_output[ld_pos], output_ptr, is_valid);
      grad_output_ptr += kNumThreadsPerLine;
      output_ptr += kNumThreadsPerLine;
    };

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < kPipelineStages - 1; ++iter) {
      int ld_pos = iter % kPipelineStages;
      pred = pred &&
          (laneFirstCol + iter * kElementsPerAccess * kNumThreadsPerLine) <
              p.head_dim_value;
      loadAndIncrement(ld_pos, pred);
    }
    auto columnIteration = [&](int iter) {
      // Load for next iter
      int ld_pos = (iter + kPipelineStages - 1) % kPipelineStages;
      pred = pred &&
          (laneFirstCol +
           (iter + kPipelineStages - 1) * kElementsPerAccess *
               kNumThreadsPerLine) < p.head_dim_value;
      loadAndIncrement(ld_pos, pred);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < AccessType::kElements; ++i) {
        delta_value += accum_t(frag_output[iter % kPipelineStages][i]) *
            accum_t(frag_grad_output[iter % kPipelineStages][i]);
      }
    };

    // If we have a small lower-bound for K, we can unroll the loop
    if (kMaxK <= 256) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kMaxIters; ++iter) {
        columnIteration(iter);
      }
    } else {
      int num_iters =
          ceil_div(p.head_dim_value, kElementsPerAccess * kNumThreadsPerLine) *
          (kElementsPerAccess * kNumThreadsPerLine);
      for (int iter = 0; iter < num_iters; ++iter) {
        columnIteration(iter);
      }
    }

    // Reduce between workers
    static_assert(
        kNumThreadsPerLine == 1 || kNumThreadsPerLine == 2 ||
            kNumThreadsPerLine == 4,
        "");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kNumThreadsPerLine; i *= 2) {
      delta_value = delta_value + __shfl_xor_sync(0xffffffff, delta_value, i);
    }

    // Store in gmem
    if (rowPred) {
      p.delta_ptr[query_start + laneRow] = delta_value;
    }
  }

  static __device__ __forceinline__ int8_t get_lane_id() {
    return threadIdx.x;
  }
  static __device__ __forceinline__ int8_t get_warp_id() {
    return threadIdx.y;
  }
  static __device__ __forceinline__ int16_t get_thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_backward_batched(typename AK::Params params);

#define _ATTENTION_KERNEL_BACKWARD_BEGIN(...)                 \
  template <>                                                 \
  __global__ void __launch_bounds__(                          \
      __VA_ARGS__::kNumThreads, __VA_ARGS__::kMinBlocksPerSm) \
      attention_kernel_backward_batched<__VA_ARGS__>(         \
          typename __VA_ARGS__::Params p) {                   \
    using Kernel = __VA_ARGS__;
#define _ATTENTION_KERNEL_BACKWARD_END() }

#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD(ARCH, ...)             \
  _ATTENTION_KERNEL_BACKWARD_BEGIN(                                  \
      AttentionBackwardKernel<cutlass::arch::Sm##ARCH, __VA_ARGS__>) \
  auto batch_id = blockIdx.z;                                        \
  p.advance_batches(batch_id);                                       \
  Kernel::kernel(p);                                                 \
  _ATTENTION_KERNEL_BACKWARD_END();

#ifdef __CUDA_ARCH__
#define __CUDA_ARCH_OR_ZERO__ __CUDA_ARCH__
#else
#define __CUDA_ARCH_OR_ZERO__ 0
#endif

#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(ARCH, ...)                \
  _ATTENTION_KERNEL_BACKWARD_BEGIN(                                              \
      AttentionBackwardKernel<cutlass::arch::Sm##ARCH, __VA_ARGS__>)             \
  printf(                                                                        \
      "FATAL: this function is for sm%d, but was built with __CUDA_ARCH__=%d\n", \
      int(ARCH),                                                                 \
      int(__CUDA_ARCH_OR_ZERO__));                                               \
  _ATTENTION_KERNEL_BACKWARD_END();

// All kernels are disabled by default
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(50, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(70, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(75, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(80, __VA_ARGS__)

// Enable the right one based on __CUDA_ARCH__
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 500
// "Need cuda arch at least 5.0"
#elif __CUDA_ARCH__ < 700
#undef INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD(50, __VA_ARGS__)
#elif __CUDA_ARCH__ < 750
#undef INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD(70, __VA_ARGS__)
#elif __CUDA_ARCH__ < 800
#undef INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD(75, __VA_ARGS__)
#elif __CUDA_ARCH__ >= 800
#undef INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80
#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_BACKWARD(80, __VA_ARGS__)
#endif
