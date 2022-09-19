#pragma once

#include <ATen/ATen.h>
// #include <torch/library.h>
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

#include <ATen/native/transformers/cuda/mem_eff_attention/find_default_mma.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/mma_from_smem.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>

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

template <typename scalar_t_, bool kIsAligned_, typename ArchTag>
struct AttentionBackwardKernel {
  using scalar_t = scalar_t_;
  using output_t = scalar_t;
  using lse_scalar_t = float;
  using accum_t = float;
  static constexpr bool kIsAligned = kIsAligned_;

  struct Params {
    // Input tensors
    scalar_t* query_ptr; // [num_queries, head_dim]
    scalar_t* key_ptr; // [num_keys, head_dim]
    scalar_t* value_ptr; // [num_keys, head_dim_value]
    lse_scalar_t* logsumexp_ptr; // [num_queries]
    scalar_t* output_ptr; // [num_queries, head_dim_value]
    scalar_t* grad_output_ptr; // [num_queries, head_dim_value]

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
  static constexpr int64_t kWarpSize = 32;
  static constexpr int64_t kNumWarpsPerBlock = 4;
  static constexpr int64_t kBlockSizeI = 64;
  static constexpr int64_t kBlockSizeJ = 64;

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
        ArchTag::kMinComputeCapability >= 80
            ? cutlass::gemm::SharedMemoryClearOption::kZfill
            : cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma = typename DefaultMma::ThreadblockMma;

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
    using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

    struct SharedStorage {
      union {
        // Storing parts of `V` during the matmul
        typename Mma::SharedStorage mm;
        // Used by the Epilogue (so we can reuse the same memory space)
        typename DefaultEpilogue::SharedStorage epilogue;
      };
    };
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
        ArchTag::kMinComputeCapability >= 80
            ? cutlass::gemm::SharedMemoryClearOption::kZfill
            : cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma = typename DefaultMma::ThreadblockMma;

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
    using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

    struct SharedStorage {
      union {
        // Storing parts of `V` during the matmul
        typename Mma::SharedStorage mm;
        // Used by the Epilogue (so we can reuse the same memory space)
        typename DefaultEpilogue::SharedStorage epilogue;
      };
    };
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
    using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

    struct SharedStorage {
      union {
        typename Mma::SharedStorage mm;
        typename DefaultEpilogue::SharedStorage epilogue;
      };
    };
  };

  struct SharedStorage {
    struct AfterDOIJV {
      typename MatmulDOIVJ::AccumulatorSharedStorage doivj_shared_storage;
      union {
        typename MatmulGradQ::SharedStorage mm_gradQ;
        typename MatmulGradK::SharedStorage mm_gradK;
      };
    };
    struct AfterQK {
      typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
      union {
        typename MatmulGradV::SharedStorage mm_gradV;
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
        AfterDOIJV after_doivj;
      };
    };
    cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
    union {
      typename MatmulQK::Mma::SharedStorage qk;
      AfterQK after_qk;
    };
  };

  // OLD VERSION - a3f257389709
  template <int kElementsPerAccess>
  static __device__ void _computeDi(
      cutlass::Array<accum_t, kBlockSizeI>& di,
      Params const& p,
      int32_t query_start) {
    __syncthreads();
    using AccessType = cutlass::Array<scalar_t, kElementsPerAccess>;
    static constexpr int kNumThreadsPerLine = 4;
    static constexpr int kParallelRowsPerWarp = kWarpSize / kNumThreadsPerLine;

    int32_t laneCol = (get_lane_id() % kNumThreadsPerLine);
    int32_t laneRow = (get_lane_id() / kNumThreadsPerLine) +
        get_warp_id() * kBlockSizeI / kNumWarpsPerBlock;

    int32_t dO_s0 = p.head_dim_value / AccessType::kElements;
    int32_t out_s0 = p.head_dim_value / AccessType::kElements;
    cutlass::
        Array<accum_t, kBlockSizeI / kParallelRowsPerWarp / kNumWarpsPerBlock>
            di_frag;
    di_frag.clear();
    assert(p.head_dim_value % AccessType::kElements == 0);
    CUTLASS_PRAGMA_UNROLL
    for (int firstCol = 0; firstCol < p.head_dim_value;
         firstCol += kNumThreadsPerLine * AccessType::kElements) {
      const __restrict__ AccessType* dO =
          reinterpret_cast<const __restrict__ AccessType*>(
              p.grad_output_ptr + (query_start + laneRow) * p.head_dim_value +
              firstCol);
      const __restrict__ AccessType* out =
          reinterpret_cast<const __restrict__ AccessType*>(
              p.output_ptr + (query_start + laneRow) * p.head_dim_value +
              firstCol);
      int32_t rowEnd = (p.num_queries - query_start);
      int32_t colEnd = p.head_dim_value / AccessType::kElements;

      AccessType frag_dO;
      AccessType frag_out;
      AccessType result;
      frag_dO.clear();
      frag_out.clear();
      dO += laneCol;
      out += laneCol;

      bool withinBounds =
          firstCol + laneCol * AccessType::kElements < p.head_dim_value;

      CUTLASS_PRAGMA_UNROLL
      for (int frag_idx = 0; frag_idx < di_frag.size(); ++frag_idx) {
        int32_t fetching_index = laneRow + frag_idx * kParallelRowsPerWarp;
        if (fetching_index >= rowEnd) {
          break;
        }
        if (withinBounds) {
          frag_dO = *dO;
          frag_out = *out;
          dO += dO_s0 * kParallelRowsPerWarp;
          out += out_s0 * kParallelRowsPerWarp;
          cutlass::multiplies<AccessType> multiply;
          result = multiply(frag_dO, frag_out);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < AccessType::kElements; ++i) {
            di_frag[frag_idx] = di_frag[frag_idx] + accum_t(result[i]);
          }
        }
      }
    }
    // Store everything in smem
    CUTLASS_PRAGMA_UNROLL
    for (int frag_idx = 0; frag_idx < di_frag.size(); ++frag_idx) {
      int32_t fetching_index = laneRow + frag_idx * kParallelRowsPerWarp;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kNumThreadsPerLine; i *= 2) {
        di_frag[frag_idx] = di_frag[frag_idx] +
            __shfl_xor_sync(0xffffffff, di_frag[frag_idx], i);
      }
      di[fetching_index] = di_frag[frag_idx];
    }
    __syncthreads();
  }

  static __device__ void computeDi(
      cutlass::Array<accum_t, kBlockSizeI>& di,
      Params const& p,
      int32_t query_start) {
    constexpr int kOptimalElements =
        128 / cutlass::sizeof_bits<scalar_t>::value;
    if (p.head_dim_value % kOptimalElements == 0) {
      _computeDi<kOptimalElements>(di, p, query_start);
    } else {
      _computeDi<1>(di, p, query_start);
    }
  }

  static __device__ void kernel(Params& p_) {
    // Hint to nvcc to store points & tensor shapes in registers
    // as we use them a lot
    register const Params p = p_;

    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    int32_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    auto clearSmem = [&]() {
      // Initialize shared-memory. It can contain `nans` otherwise that screw up
      // everything (only seens on Sm75+ tho)
      uint32_t* smem = (uint32_t*)smem_buffer;
      for (int i = 0; i < sizeof(SharedStorage) / sizeof(uint32_t) -
               kWarpSize * kNumWarpsPerBlock;
           i += kWarpSize * kNumWarpsPerBlock) {
        smem[i + thread_id] = 0;
      }
    };

    auto getNumKeys = [&](int32_t query_start) {
      if (p.causal) {
        return std::min(int32_t(query_start + kBlockSizeI), p.num_keys);
      }
      return p.num_keys;
    };

    int32_t query_end = p.num_queries / kBlockSizeI * kBlockSizeI;
    int32_t query_start = 0;
    for (; query_start < query_end; query_start += kBlockSizeI) {
      clearSmem();
      computeDi(shared_storage.di, p, query_start);

      int32_t key_start = 0;
      int32_t key_end = getNumKeys(query_start) / kBlockSizeJ * kBlockSizeJ;
      for (; key_start < key_end; key_start += kBlockSizeJ) {
        processBlockIJ<true>(shared_storage, p, query_start, key_start);
      }
      // last (partial) key
      if (key_start != getNumKeys(query_start)) {
        processBlockIJ<false>(shared_storage, p, query_start, key_start);
      }
    }
    // Last (partial) query block
    if (query_start != p.num_queries) {
      computeDi(shared_storage.di, p, query_start);
      for (int32_t key_start = 0; key_start < getNumKeys(query_start);
           key_start += kBlockSizeJ) {
        processBlockIJ<false>(shared_storage, p, query_start, key_start);
      }
    }
  }

  // Compute threadblock location
  template <bool skipBoundsChecks>
  static __device__ __forceinline__ void processBlockIJ(
      SharedStorage& shared_storage,
      Params const& p,
      int32_t query_start,
      int32_t key_start) {
    cutlass::MatrixCoord no_offset{0, 0};
    accum_t scale = accum_t(1.0 / std::sqrt(float(p.head_dim)));
    int32_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t warp_id = threadIdx.y;
    int32_t lane_id = threadIdx.x;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulQK
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulQK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks ? MatmulQK::ThreadblockShape::kM
                           : p.num_keys - key_start,
          skipBoundsChecks ? MatmulQK::ThreadblockShape::kN
                           : p.num_queries - query_start,
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

      Mma mma(shared_storage.qk, thread_id, warp_id, lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
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
        int32_t last_col;
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
      MatmulQK::B2bGemm::accumApplyLSEToSmem(
          shared_storage.after_qk.attn_shared_storage,
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
    for (int col = 0; col < p.head_dim_value;
         col += MatmulGradV::ThreadblockShape::kN) {
      using Mma = typename MatmulGradV::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks ? MatmulGradV::ThreadblockShape::kM
                           : p.num_keys - key_start,
          p.head_dim_value - col,
          skipBoundsChecks ? MatmulQK::Mma::Shape::kN
                           : std::min(
                                 (int32_t)MatmulQK::Mma::Shape::kN,
                                 p.num_queries - query_start));

      // q_i.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim_value)},
          p.grad_output_ptr + query_start * p.head_dim_value + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.mm_gradV.mm,
          shared_storage.after_qk.attn_shared_storage,
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();

      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();

      // Output results
      typename MatmulGradV::OutputTileIterator output_read_it(
          typename MatmulGradV::OutputTileIterator::Params{p.head_dim_value},
          p.grad_value_ptr + key_start * p.head_dim_value + col,
          {skipBoundsChecks ? MatmulGradV::ThreadblockShape::kM
                            : p.num_keys - key_start,
           p.head_dim_value - col},
          thread_id);
      typename MatmulGradV::OutputTileIterator output_write_it = output_read_it;

      DISPATCH_BOOL(
          query_start == 0, kIsFirst, ([&]() {
            using DefaultEpilogue = typename MatmulGradV::DefaultEpilogue;
            using DefaultOutputOp = typename MatmulGradV::DefaultOutputOp;
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
            using Epilogue = typename cutlass::epilogue::threadblock::Epilogue<
                typename DefaultEpilogue::Shape,
                typename Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename DefaultEpilogue::OutputTileIterator,
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
                shared_storage.after_qk.mm_gradV.epilogue,
                thread_id,
                warp_id,
                lane_id);
            epilogue(rescale, output_write_it, accum, output_read_it);
          }));
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulDOIVJ
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulDOIVJ::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks ? MatmulDOIVJ::ThreadblockShape::kM
                           : p.num_queries - query_start,
          skipBoundsChecks ? MatmulDOIVJ::ThreadblockShape::kN
                           : p.num_keys - key_start,
          p.head_dim_value // k
      );

      // do_i
      typename Mma::IteratorA iterator_A(
          {int32_t(p.head_dim_value)},
          p.grad_output_ptr + query_start * p.head_dim_value,
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // v_j.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim_value)},
          p.value_ptr + key_start * p.head_dim_value,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(shared_storage.after_qk.mm_doivj, thread_id, warp_id, lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

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
        auto attn_T = shared_storage.after_qk.attn_shared_storage.accum_ref();
        accum_t current_di;
        typename Mma::FragmentC fragment_attn, fragment_di, fragment_pos;
        RegistersIter::iterateRows(
            lane_offset,
            [&](int accum_m) { current_di = shared_storage.di[accum_m]; },
            [&](int accum_m, int accum_n, int idx) {
              // TODO: Otherwise we can get nans as we
              // might have infs here (only seen on f16 tho)
              if (skipBoundsChecks ||
                  (accum_m < problem_size.m() && accum_n < problem_size.n())) {
                fragment_attn[idx] = attn_T.at({accum_n, accum_m});
              } else {
                fragment_attn[idx] = 0;
              }
              fragment_di[idx] = current_di;
              fragment_pos[idx] = 100 * accum_m + accum_n;
            },
            [&](int accum_m) {

            });
        accum = (accum - fragment_di) * fragment_attn * scale;
        __syncthreads();
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
          shared_storage.after_qk.after_doivj.doivj_shared_storage,
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
          skipBoundsChecks
              ? MatmulGradQ::ThreadblockShape::kM
              : std::min((int32_t)Mma::Shape::kM, p.num_queries - query_start),
          false ? MatmulGradQ::ThreadblockShape::kN : p.head_dim - col,
          skipBoundsChecks
              ? MatmulQK::Mma::Shape::kM
              : std::min(
                    (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start));

      // k_j
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim)},
          p.key_ptr + key_start * p.head_dim + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.after_doivj.mm_gradQ.mm,
          shared_storage.after_qk.after_doivj.doivj_shared_storage,
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();
      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();

      // Output results
      typename MatmulGradQ::OutputTileIterator output_read_it(
          typename MatmulGradQ::OutputTileIterator::Params{p.head_dim},
          p.grad_query_ptr + query_start * p.head_dim + col,
          {problem_size.m(), problem_size.n()},
          thread_id);
      typename MatmulGradQ::OutputTileIterator output_write_it = output_read_it;

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
            using Epilogue = typename cutlass::epilogue::threadblock::Epilogue<
                typename DefaultEpilogue::Shape,
                typename Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename DefaultEpilogue::OutputTileIterator,
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
                shared_storage.after_qk.after_doivj.mm_gradQ.epilogue,
                thread_id,
                warp_id,
                lane_id);
            epilogue(rescale, output_write_it, accum, output_read_it);
          }));
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradK matmul
    //
    // grad_k[i_start:i_end] += tmp.transpose(-2, -1) @ q_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    for (int col = 0; col < p.head_dim;
         col += MatmulGradK::ThreadblockShape::kN) {
      using Mma = typename MatmulGradK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          skipBoundsChecks
              ? MatmulGradK::ThreadblockShape::kM
              : std::min((int32_t)Mma::Shape::kM, p.num_keys - key_start),
          false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col,
          skipBoundsChecks ? MatmulQK::Mma::Shape::kN
                           : std::min(
                                 (int32_t)MatmulQK::Mma::Shape::kN,
                                 p.num_queries - query_start));

      // q_i
      typename Mma::IteratorB iterator_B(
          {int32_t(p.head_dim)},
          p.query_ptr + query_start * p.head_dim + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.after_qk.after_doivj.mm_gradK.mm,
          shared_storage.after_qk.attn_shared_storage, // storing tmp.T
          thread_id,
          warp_id,
          lane_id,
          problem_size.k());

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();
      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();

      // Output results
      typename MatmulGradK::OutputTileIterator output_read_it(
          typename MatmulGradK::OutputTileIterator::Params{p.head_dim},
          p.grad_key_ptr + key_start * p.head_dim + col,
          {skipBoundsChecks
               ? MatmulGradK::ThreadblockShape::kM
               : std::min((int32_t)Mma::Shape::kM, p.num_keys - key_start),
           false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col},
          thread_id);
      typename MatmulGradK::OutputTileIterator output_write_it = output_read_it;

      DISPATCH_BOOL(
          query_start == 0, kIsFirst, ([&]() {
            using DefaultEpilogue = typename MatmulGradK::DefaultEpilogue;
            using DefaultOutputOp = typename MatmulGradK::DefaultOutputOp;
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
            using Epilogue = typename cutlass::epilogue::threadblock::Epilogue<
                typename DefaultEpilogue::Shape,
                typename Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename DefaultEpilogue::OutputTileIterator,
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
                shared_storage.after_qk.after_doivj.mm_gradK.epilogue,
                thread_id,
                warp_id,
                lane_id);
            epilogue(rescale, output_write_it, accum, output_read_it);
          }));
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

#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD(ARCH, SCALAR_T, IS_ALIGNED)      \
  template <>                                                                  \
  __global__ void __launch_bounds__(                                           \
      AttentionBackwardKernel<SCALAR_T, IS_ALIGNED, cutlass::arch::Sm##ARCH>:: \
          kNumThreads,                                                         \
      AttentionBackwardKernel<SCALAR_T, IS_ALIGNED, cutlass::arch::Sm##ARCH>:: \
          kMinBlocksPerSm)                                                     \
      attention_kernel_backward_batched<AttentionBackwardKernel<               \
          SCALAR_T,                                                            \
          IS_ALIGNED,                                                          \
          cutlass::arch::Sm##ARCH>>(                                           \
          AttentionBackwardKernel<                                             \
              SCALAR_T,                                                        \
              IS_ALIGNED,                                                      \
              cutlass::arch::Sm##ARCH>::Params params) {                       \
    auto batch_id = blockIdx.z;                                                \
    params.advance_batches(batch_id);                                          \
    AttentionBackwardKernel<SCALAR_T, IS_ALIGNED, cutlass::arch::Sm##ARCH>::   \
        kernel(params);                                                        \
  }

#ifdef __CUDA_ARCH__
#define __CUDA_ARCH_OR_ZERO__ __CUDA_ARCH__
#else
#define __CUDA_ARCH_OR_ZERO__ 0
#endif

#define INSTANTIATE_ATTENTION_KERNEL_BACKWARD_DISABLED(                        \
    ARCH, SCALAR_T, IS_ALIGNED)                                                \
  template <>                                                                  \
  __global__ void __launch_bounds__(                                           \
      AttentionBackwardKernel<SCALAR_T, IS_ALIGNED, cutlass::arch::Sm##ARCH>:: \
          kNumThreads,                                                         \
      AttentionBackwardKernel<SCALAR_T, IS_ALIGNED, cutlass::arch::Sm##ARCH>:: \
          kMinBlocksPerSm)                                                     \
      attention_kernel_backward_batched<AttentionBackwardKernel<               \
          SCALAR_T,                                                            \
          IS_ALIGNED,                                                          \
          cutlass::arch::Sm##ARCH>>(                                           \
          AttentionBackwardKernel<                                             \
              SCALAR_T,                                                        \
              IS_ALIGNED,                                                      \
              cutlass::arch::Sm##ARCH>::Params params) {                       \
    printf(                                                                    \
        "FATAL: this function is for sm%d, but was built for sm%d\n",          \
        int(ARCH),                                                             \
        int(__CUDA_ARCH_OR_ZERO__));                                           \
  }

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
#ifndef __CUDA_ARCH__
#elif __CUDA_ARCH__ < 500
#error "Need cuda arch at least 5.0"
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
