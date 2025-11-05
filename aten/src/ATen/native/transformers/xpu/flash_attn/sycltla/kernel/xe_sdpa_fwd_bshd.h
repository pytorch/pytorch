#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "../collective/xe_flash_attn_prefill_mma_bshd.h"

namespace cutlass::flash_attention::kernel {

template <class ProblemShape, class CollectiveMainloop,
          class CollectiveSoftmaxEpilogue_, class CollectiveEpilogue,
          class TileScheduler_ = void>
class FMHAPrefill;

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_,
          class CollectiveSoftmaxEpilogue_, class CollectiveEpilogue_,
          class TileScheduler_>
class FMHAPrefill {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 7,
                "ProblemShape{} should be <batch, num_heads_q, num_head_kv, "
                "seq_len_qo, seq_len_kv, head_size_qk, head_size_vo>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using TiledMmaQK = typename CollectiveMainloop::TiledMmaQK;
  using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementQ = typename CollectiveMainloop::ElementQ;
  using StrideQ = typename CollectiveMainloop::StrideQ;
  using ElementK = typename CollectiveMainloop::ElementK;
  using StrideK = typename CollectiveMainloop::StrideK;
  using ElementV = typename CollectiveMainloop::ElementV;
  using StrideV = typename CollectiveMainloop::StrideV;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using CollectiveSoftmaxEpilogue = CollectiveSoftmaxEpilogue_;
  using SoftmaxArguments = typename CollectiveSoftmaxEpilogue::Arguments;
  using SoftmaxParams = typename CollectiveSoftmaxEpilogue::Params;

  static_assert(cute::is_void_v<TileScheduler_> or
                    cute::is_same_v<TileScheduler_, PersistentScheduler> or
                    cute::is_same_v<TileScheduler_, IndividualScheduler>,
                "Unsupported TileScheduler for Intel Xe.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler =
      typename detail::TileSchedulerSelector<TileScheduler_,
                                             ArchTag>::Scheduler;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementO = typename CollectiveEpilogue::ElementO;
  using StrideO = typename CollectiveEpilogue::StrideO;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileShapeOutput = typename CollectiveEpilogue::TileShapeOutput;
  using TiledMmaOutput = typename CollectiveEpilogue::TiledMmaOutput;

  static_assert(
      cute::is_same_v<ElementAccumulator,
                      typename CollectiveEpilogue::ElementAccumulator>,
      "Mainloop and epilogue do not agree on accumulator value type.");
  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = 0;

  static constexpr bool CausalMask = CollectiveMainloop::CausalMask;
  static constexpr int SubgroupSize =
      CollectiveMainloop::SubgroupSize; // sub_group size
  static constexpr uint32_t MaxThreadsPerBlock =
      CollectiveMainloop::MaxThreadsPerBlock;
  using MmaAtomShape = typename CollectiveMainloop::MmaAtomShape; // 8,16,16

  static constexpr int QK_BLK_M = CollectiveMainloop::QK_BLK_M;
  static constexpr int QK_BLK_N = CollectiveMainloop::QK_BLK_N;
  static constexpr int QK_BLK_K = CollectiveMainloop::QK_BLK_K;

  static constexpr int QK_ATOM_N = CollectiveMainloop::QK_ATOM_N;
  static constexpr int QK_ATOM_K = CollectiveMainloop::QK_ATOM_K;

  static constexpr int QK_SG_M = CollectiveMainloop::QK_SG_M;

  static constexpr int Epilogue_BLK_N = get<1>(TileShapeOutput{});
  static constexpr int Epilogue_BLK_K = get<2>(TileShapeOutput{});

  static constexpr int PV_ATOM_M = CollectiveMainloop::PV_ATOM_M;
  static constexpr int PV_ATOM_N = CollectiveMainloop::PV_ATOM_N;
  static constexpr int PV_ATOM_K = CollectiveMainloop::PV_ATOM_K;

  static constexpr auto Num_SGs = PV_ATOM_N * PV_ATOM_M * PV_ATOM_K;
  static constexpr int Vec = CollectiveMainloop::Vec;
  static constexpr int FragsM = CollectiveMainloop::FragsM;
  // The FragsN here used for Creation of S matrix so we use the FragsN for S
  // shape
  static constexpr int FragsN = CollectiveMainloop::FragsNS;

  static constexpr int VSlicer =
      get<1>(TileShapeOutput{}) /
      (get<1>(TileShapePV{}) * PV_ATOM_N); // ceil_div(FragsNOut,FragsNS);
  using AccumeShape = decltype(make_shape(
      Int<Vec>{}, Int<FragsM>{}, get<1>(TileShapePV{}) / get<1>(MmaAtomShape()),
      Int<VSlicer>{}));

  static constexpr bool is_var_len = CollectiveMainloop::is_var_len;
  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  // Device side arguments
  struct Arguments {
    gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    SoftmaxArguments softmax{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    float softmax_scale;
  };

  // Kernel entry point API
  struct Params {
    gemm::GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    SoftmaxParams softmax;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
    float softmax_scale;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the
  // aliased type.
  static Params to_underlying_arguments(Arguments const &args,
                                        void *workspace) {
    (void)workspace;
    return {args.mode,
            args.problem_shape,
            CollectiveMainloop::to_underlying_arguments(
                args.problem_shape, args.mainloop, workspace),
            CollectiveSoftmaxEpilogue::to_underlying_arguments(args.softmax),
            CollectiveEpilogue::to_underlying_arguments(
                args.problem_shape, args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(
                args.problem_shape, args.hw_info, TileShapeOutput{}), args.softmax_scale};
  }

  static bool can_implement(Arguments const &args) {
    bool mode_implementable = args.mode == gemm::GemmUniversalMode::kGemm or
                              (args.mode == gemm::GemmUniversalMode::kBatched &&
                               rank(ProblemShape{}) == 4);
    return mode_implementable;
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status
  initialize_workspace(Arguments const &args, void *workspace = nullptr,
                       cudaStream_t stream = nullptr,
                       CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<Num_SGs>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(ProblemShape const &problem_shape,
                                            int const &batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(
          select<3, 4>(problem_shape), batch);
    } else {
      return select<3, 4>(problem_shape);
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {
    SharedStorage &shared_storage =
        *reinterpret_cast<SharedStorage *>(smem_buf);
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShapeQK>::value);
    CUTE_STATIC_ASSERT(is_static<TileShapePV>::value);
    // Separate out problem shape for convenience
    auto &batch = get<0>(params.problem_shape);
    auto &num_heads_q = get<1>(params.problem_shape);
    auto &num_head_kv = get<2>(params.problem_shape);
    auto group_heads_q = num_heads_q / num_head_kv;
    auto &head_size_qk = get<5>(params.problem_shape);
    auto &head_size_vo = get<6>(params.problem_shape);
    auto &softmax_scale = params.softmax_scale;

    // Preconditions
    static_assert(cute::rank(StrideQ{}) == 3,
                  "StrideQ must be rank-3: [seq_len_qo, head_size_qk, batch * "
                  "num_heads_q].");
    static_assert(cute::rank(StrideK{}) == 3,
                  "StrideK must be rank-3: [head_size_qk, seq_len_kv, batch * "
                  "num_heads_kv].");
    static_assert(cute::rank(StrideV{}) == 3,
                  "StrideV must be rank-3: [seq_len_kv, head_size_vo, batch * "
                  "num_heads_kv].");

    int thread_idx = int(ThreadIdxX());
    int sub_group_id = thread_idx / SubgroupSize;

    TileScheduler tile_scheduler{params.scheduler};
    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto blk_coord =
          tile_scheduler
              .get_block_coord(); // head_size_blk_idx, seq_len_blk_idx,
                                  // batch_blk_idx, num_heads_blk_idx

      auto blk_m_coord = get<0>(blk_coord);  // seq_len_blk_idx
      auto q_head_coord = get<1>(blk_coord); // q_heads_idx
      auto batch_coord = get<2>(blk_coord);  // batch_blk_idx
      auto blk_n_coord = 0; // nums_head_blk_idx - not defined in TileScheduler

      // For variable sequence length case, batch is considered to be 1 (same as
      // group gemm). For fixed sequence length case, the l_coord is the
      // weighted sum of both batch_coord and num_heads_coord. Flash Attention
      // implementation combines batch and num_heads to calculate the total
      // batch_size. iff is_var_len: batch_size = num_heads (as each batch would
      // have it's own seq_len_qo and seq_len_kv) iff !is_var_len: batch_size =
      // batch * num_heads
      // auto blk_l_coord = is_var_len ? num_heads_coord : batch_coord *
      // num_heads_q + num_heads_coord;

      // Get problem shape for the current batch_blk_idx. For variable sequence
      // length, it loads the sequence length from Global memory for the given
      // batch_blk_idx and returns the appropriate problem_shape. For fixed
      // sequence length, sequence_length_shape == select<3,
      // 4>(params.problem_shape). sequence_length_shape = [seq_len_qo,
      // seq_len_kv]
      auto sequence_length_shape =
          get_sequence_length_shape(params.problem_shape, batch_coord);

      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;

      // Calculate the seq_len_idx (blk_m_coord * get<0>(TileShapeOutput{})) and
      // check if it is still within bounds of the actual seq_len_qo
      // (get<0>(sequence_length_shape)).
      if (blk_m_coord * get<0>(TileShapeOutput{}) >= seq_len_qo) {
        continue;
      }

      auto offset = cute::min(seq_len_qo, seq_len_kv); //(2048, 1024)
      auto discard_seq_coord = seq_len_qo - offset;    // 1024
      auto full_tile_offset = seq_len_kv - offset;     // 0
      const int seq_coord =
          cute::min(seq_len_qo, (blk_m_coord * QK_BLK_M +
                                 (sub_group_id / PV_ATOM_N) * QK_SG_M) %
                                    seq_len_qo);
      const int seq_len =
          CausalMask
              ? full_tile_offset +
                    cute::min(seq_len_kv, seq_coord - discard_seq_coord) +
                    QK_SG_M
              : seq_len_kv;
      const int nblock_limit = cute::ceil_div(seq_len, QK_BLK_N);
      if (CausalMask && seq_coord < discard_seq_coord) { // 1024 =0
        continue;
      }

      Tensor mQ_mkl = cute::get_xe_tensor(
          make_shape(seq_len_qo, head_size_qk, 1)); //(m,k,l)
      Tensor mK_nkl = cute::get_xe_tensor(
          make_shape(seq_len_kv, head_size_qk, 1)); //(n,k,l)
      Tensor mV_nkl = cute::get_xe_tensor(
          make_shape(head_size_vo, seq_len_kv, 1)); //(n,k,l)
      Tensor mQ_mk = mQ_mkl(_, _, 0);
      Tensor mK_nk = mK_nkl(_, _, 0); // (n,k)
      Tensor mV_nk = mV_nkl(_, _, 0);

      auto gQ = local_tile(mQ_mk, TileShapeQK{}, make_coord(blk_m_coord, _, _),
                           Step<_1, X, _1>{});
      auto gK = local_tile(mK_nk, TileShapeQK{}, make_coord(_, _, _),
                           Step<X, _1, _1>{});
      auto gV = local_tile(mV_nk, TileShapeOutput{},
                           make_coord(_, blk_n_coord, _), Step<X, _1, _1>{});

      auto mainloop_params = CollectiveMainloop::get_updated_copies(
          params.mainloop, params.problem_shape, sequence_length_shape,
          batch_coord, q_head_coord);
      // we limit the horisontal size to two subgroup, the empirical resutls
      // show that reading the two cacheline side by side in gives better
      // performance and anything after that does not have an effect on
      // performance. // (64 here for float b float when possible and loop over
      // to cover all the data needed)
      auto tiled_prefetch_q = cute::prefetch_selector<
          Shape<Int<QK_BLK_M>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>,
          Num_SGs>(mainloop_params.gmem_tiled_copy_q);
      auto tiled_prefetch_k = cute::prefetch_selector<
          Shape<Int<QK_BLK_N>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>,
          Num_SGs>(mainloop_params.gmem_tiled_copy_k);
      auto tiled_prefetch_v = cute::prefetch_selector<
          Shape<Int<cute::max(cute::gcd(Epilogue_BLK_N, 64), 32)>,
                Int<Epilogue_BLK_K>>,
          Num_SGs>(mainloop_params.gmem_tiled_copy_v);
      auto thr_prefetch_Q = tiled_prefetch_q.get_slice(thread_idx);
      auto thr_prefetch_K = tiled_prefetch_k.get_slice(thread_idx);
      auto thr_prefetch_V = tiled_prefetch_v.get_slice(thread_idx);
      auto pQgQ = thr_prefetch_Q.partition_S(gQ);
      auto pKgK = thr_prefetch_K.partition_S(gK);
      auto pVgV = thr_prefetch_V.partition_S(gV);

      for (int i = 0; i < size<3>(pQgQ); i++) {
        prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
      }
      for (int j = 0; j < size<4>(pKgK); j++) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < DispatchPolicy::Stages; i++) {
          prefetch(tiled_prefetch_k, pKgK(_, _, _, i, j));
        }
      }

      // Allocate the tiled_mma and the accumulators for the (M,N)
      // workgroup_shape
      Tensor out_reg = make_tensor<ElementAccumulator>(AccumeShape{});

      // There are 16 workitem and 16 max per subgroup, each worktime containt 1
      // max and cumulatively, they calculate the max per subgroup
      ElementAccumulator max_reg{-INFINITY};

      // The sum reg each contains a 2d tesnor for 8 x 2 This is number of
      // sequence lenght process per subgroup
      Tensor sum_reg =
          make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>>{});

      clear(sum_reg);
      clear(out_reg);

      // Perform the collective scoped MMA
      CollectiveMainloop collective_mma;
      // when causal mask is true. It is not possible to set the scope
      // of the barrier to workgroup level as the number n block is
      // different for each subgroup due to triangular nature of causal based
      // operation
      static constexpr int barrier_scope = CausalMask ? 3 : 2;
      // MAIN LOOP: loop over K and V, perform fused attention + online softmax
      for (int nblock = 0; nblock < nblock_limit - static_cast<int>(CausalMask);
           nblock++) {
        barrier_arrive(barrier_scope);
        // 1) Load K (performed inside mmaQK)
        // 2) Create Tensor S
        Tensor tSr = make_tensor<ElementAccumulator>(
            Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
        clear(tSr);

        // 3) Perform GEMM S = Q*K
        collective_mma.mmaQK(tSr, gQ, gK(_, _, nblock, _), tSr,
                             ceil_div(head_size_qk, QK_BLK_K), mainloop_params);

        // we only need one block ahead, there is enough gap to prefetch it
        // while doing softmax. because the gap between the two MMA is big,
        // prefetching it the same way as cutlass K matrix does not make sense
        for (int i = 0; i < size<1>(pVgV); i++) {
          prefetch(tiled_prefetch_v, pVgV(_, i, _, nblock));
        }

        CollectiveSoftmaxEpilogue softmax(params.softmax);
        softmax(nblock == 0, tSr, max_reg, sum_reg, out_reg);

        collective_mma.template mmaPV<VSlicer>(out_reg, tSr, gV(_, _, nblock),
                                               out_reg, mainloop_params);

        // Prefetch the next K tile
        // there is no need to gaurd it with if statememt as prefetch will
        // ignore out of bound reading
        for (int j = 0; j < size<4>(pKgK); j++) {
          prefetch(tiled_prefetch_k,
                   pKgK(_, _, _, nblock + DispatchPolicy::Stages, j));
        }
        barrier_wait(barrier_scope);
      }

      if constexpr (CausalMask) {
        // BAND Matrix
        // 1) Load K (performed inside mmaQK)
        // 2) Create Tensor S
        Tensor tSr = make_tensor<ElementAccumulator>(
            Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
        clear(tSr);
        // 3) Perform GEMM S = Q*K
        collective_mma.mmaQK(tSr, gQ, gK(_, _, nblock_limit - 1, _), tSr,
                             ceil_div(head_size_qk, QK_BLK_K), mainloop_params);
        // we only need one block ahead, there is enough gap to prefetch it
        // while doing softmax. because the gap between the two MMA is big,
        // prefetching it the same way as cutlass K matrix does not make sense
        for (int i = 0; i < size<1>(pVgV); i++) {
          prefetch(tiled_prefetch_v, pVgV(_, i, _, nblock_limit - 1));
        }
        // mask the elements of each tile where j > i
        const int item_id = thread_idx % SubgroupSize;
        int col_idx = item_id + (nblock_limit - 1) * QK_BLK_N;
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < FragsN;
             n++, col_idx += get<1>(MmaAtomShape())) { // 4
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < FragsM; m++) { // 2
            int row_idx = m * Vec + seq_coord;
            CUTLASS_PRAGMA_UNROLL
            for (int row = 0; row < Vec; row++, row_idx++) { // 8
              if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
                tSr(row, m, n) = ElementAccumulator{-INFINITY};
              }
            }
          }
        }

        CollectiveSoftmaxEpilogue softmax(params.softmax);
        softmax((nblock_limit - 1) == 0, tSr, max_reg, sum_reg, out_reg);

        collective_mma.template mmaPV<VSlicer>(
            out_reg, tSr, gV(_, _, nblock_limit - 1), out_reg, mainloop_params);
      }

      auto epilogue_params =
          CollectiveEpilogue::template get_updated_copies<is_var_len>(
              params.epilogue, params.problem_shape, sequence_length_shape,
              batch_coord, q_head_coord);
      CollectiveEpilogue epilogue{epilogue_params, shared_storage.epilogue};
      auto blk_coord_mnkl = make_coord(blk_m_coord, blk_n_coord, batch_coord, 0);
      epilogue(params.problem_shape, sequence_length_shape, blk_coord_mnkl,
               out_reg, max_reg, sum_reg, q_head_coord, softmax_scale);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::flash_attention::kernel
