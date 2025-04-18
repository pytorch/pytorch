#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/epilogue/collective/detail.hpp>

#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>

// TODO remove *BroadcastPtrArrays and replace with just Broadcast
// when  https://github.com/NVIDIA/cutlass/pull/2120/ is in the tagged cutlass version


namespace cutlass::epilogue::fusion {
  using namespace cute;
  using namespace detail;
  // Row vector broadcast with grouping.
template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput,
  class ElementCompute = ElementInput,
  class StrideMNL_ = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementInput>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct Sm90RowBroadcastPtrArray {
  using StrideMNL = StrideMNL_;
  static_assert(Stages == 0, "Row broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<1>(StrideMNL{}))>, bool>; // row vector or scalar broadcast
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{} || IsDynamicBroadcast);

  struct SharedStorage {
    array_aligned<ElementInput, size<1>(CtaTileShapeMNK{})> smem;
  };

  struct Arguments {
    ElementInput const* const* ptr_row_array = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dRow = {};
  };

  struct Params {
    ElementInput const* const* ptr_row_array = nullptr;
    ElementCompute null_default = ElementCompute(0);
    StrideMNL dRow = {};
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {args.ptr_row_array, ElementCompute(args.null_default), args.dRow};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90RowBroadcastPtrArray() { }

  CUTLASS_HOST_DEVICE
  Sm90RowBroadcastPtrArray(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false),
        smem(const_cast<ElementInput*>(shared_storage.smem.data())) {
    auto const& [stride_M, stride_N, stride_L] = params.dRow;
    // Nullptr default
    if (EnableNullptr && params.ptr_row_array == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    // Dynamic non-batched scalar broadcast
    else if (IsDynamicBroadcast && stride_N == bool(0) && stride_L == repeat_like(stride_L, 0)) {
      is_zero_ = params.ptr_row_array[0][0] == ElementInput(0);
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class GS_GTensor, class GS_STensor, class GS_CTensor, class Tiled_G2S, class SR_STensor, class SR_RTensor, class Residue, class ThrNum>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        GS_GTensor tGS_gRow_, GS_STensor tGS_sRow_,
        GS_CTensor tGS_cRow_, Tiled_G2S tiled_g2s_,
        SR_STensor tSR_sRow_, SR_RTensor tSR_rRow_,
        Residue residue_cRow_, ThrNum thr_num_, Params const& params_)
      : tGS_gRow(tGS_gRow_)
      , tGS_sRow(tGS_sRow_)
      , tGS_cRow(tGS_cRow_)
      , tiled_G2S(tiled_g2s_)
      , tSR_sRow(tSR_sRow_)
      , tSR_rRow(tSR_rRow_)
      , residue_cRow(residue_cRow_)
      , params(params_)
      , is_nullptr(EnableNullptr && params_.ptr_row_array == nullptr) {
      if (is_nullptr) {
        fill(tSR_rRow, params.null_default);
      }
    }

    GS_GTensor tGS_gRow;                                                         // (CPY,CPY_M,CPY_N)
    GS_STensor tGS_sRow;                                                         // (CPY,CPY_M,CPY_N)
    GS_CTensor tGS_cRow;                                                         // (CPY,CPY_M,CPY_N)
    Tiled_G2S tiled_G2S;

    SR_STensor tSR_sRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    SR_RTensor tSR_rRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    Residue residue_cRow;                                                        // (m, n)
    ThrNum thr_num;
    Params const& params;
    bool is_nullptr;

    CUTLASS_DEVICE void
    begin() {
      if (is_nullptr) {
        return;
      }

      auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(thr_num, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };
      Tensor tGS_gRow_flt = filter_zeros(tGS_gRow);
      Tensor tGS_sRow_flt = filter_zeros(tGS_sRow);
      Tensor tGS_cRow_flt = filter_zeros(tGS_cRow, tGS_gRow.stride());

      for (int i = 0; i < size(tGS_gRow_flt); ++i) {
        if (get<1>(tGS_cRow_flt(i)) >= size<1>(CtaTileShapeMNK{})) {
          continue; // OOB of SMEM,
        }
        if (elem_less(tGS_cRow_flt(i), residue_cRow)) {
          tGS_sRow_flt(i) = tGS_gRow_flt(i);
        }
        else {
          tGS_sRow_flt(i) = ElementInput(0); // Set to Zero when OOB so LDS can be issued without any preds.
        }
      }
      synchronize();
    }

    CUTLASS_DEVICE void
    begin_loop(int epi_m, int epi_n) {
      if (epi_m == 0 and not is_nullptr) { // Assumes M-major subtile loop
        Tensor tSR_sRow_flt = filter_zeros(tSR_sRow(_,_,_,epi_m,epi_n));
        Tensor tSR_rRow_flt = make_tensor_like<ElementInput>(tSR_sRow_flt);
        copy_aligned(tSR_sRow_flt, tSR_rRow_flt);

        constexpr int FrgSize = size(tSR_rRow_flt);
        using FrgInput = Array<ElementInput, FrgSize>;
        using FrgCompute = Array<ElementCompute, FrgSize>;
        using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FrgSize>;

        Tensor tSR_rRow_input_frg = recast<FrgInput>(coalesce(tSR_rRow_flt));
        Tensor tSR_rRow_compute_frg = recast<FrgCompute>(filter(tSR_rRow));
        ConvertInput convert_input{};

        tSR_rRow_compute_frg(_0{}) = convert_input(tSR_rRow_input_frg(_0{}));
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_row;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_row[i] = tSR_rRow(epi_v * FragmentSize + i);
      }

      return frg_row;
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    using ThreadCount = decltype(size(args.tiled_copy));

    auto layout_N = [&] () {
      auto shape_N = get<1>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_N = repeat_like(shape_N, int(0));
        if (get<1>(params.dRow) == bool(1)) {
          stride_N = transform_leaf(compact_major<LayoutLeft>(shape_N),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_N, stride_N);
      }
      else {
        return make_layout(shape_N);
      }
    }();

    auto layout_M = make_layout(M, repeat_like(M, _0{}));
    auto layout_L = make_layout(L, get<2>(params.dRow));
    Tensor mRow = make_tensor(make_gmem_ptr(params.ptr_row_array[l]), make_layout(layout_M,layout_N,layout_L));
    Tensor gRow = local_tile(mRow(_,_,l), take<0,2>(args.tile_shape_mnk), make_coord(m, n));          // (CTA_M, CTA_N)
    Tensor sRow = make_tensor(make_smem_ptr(smem),
        make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})), make_shape(_0{}, _1{}));  // (CTA_M, CTA_N)
    //// G2S: Gmem to Smem
    auto tiled_g2s = make_tiled_copy(Copy_Atom<DefaultCopy, ElementInput>{},
                                     Layout< Shape<_1, ThreadCount>,
                                            Stride<_0,          _1>>{},
                                     Layout<_1>{});
    auto thr_g2s = tiled_g2s.get_slice(args.thread_idx);
    Tensor tGS_gRow = thr_g2s.partition_S(gRow);
    Tensor tGS_sRow = thr_g2s.partition_D(sRow);

    //// G2S: Coord
    Tensor tGS_cRow = thr_g2s.partition_S(args.cD);

    //// S2R: Smem to Reg
    Tensor tSR_sRow = sm90_partition_for_epilogue<ReferenceSrc>(sRow, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tSR_rRow = make_tensor_like<ElementCompute>(take<0,3>(tSR_sRow));                        // (CPY,CPY_M,CPY_N)

    return ConsumerStoreCallbacks(
      tGS_gRow,
      tGS_sRow,
      tGS_cRow, tiled_g2s,
      tSR_sRow,
      tSR_rRow,
      args.residue_cD,
      ThreadCount{},
      params);
  }
};


// Column vector broadcast with support for grouping.
template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput,
  class ElementCompute = ElementInput,
  class StrideMNL_ = Stride<_1,_0,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementInput>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct Sm90ColBroadcastPtrArray {
  using StrideMNL = StrideMNL_;
  static_assert(Stages == 0, "Column broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<0>(StrideMNL{}))>, bool>; // Column vector or scalar broadcast
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_1,_0>{} || IsDynamicBroadcast);

  // Accumulator distributes col elements evenly amongst threads so we can just directly load from gmem
  struct SharedStorage { };

  struct Arguments {
    ElementInput const* const* ptr_col_array = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dCol = {};
  };

  struct Params {
    ElementInput const* const* ptr_col_array = nullptr;
    ElementCompute null_default = ElementCompute(0);
    StrideMNL dCol = {};
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {args.ptr_col_array, ElementCompute(args.null_default), args.dCol};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  CUTLASS_HOST_DEVICE
  Sm90ColBroadcastPtrArray() { }

  CUTLASS_HOST_DEVICE
  Sm90ColBroadcastPtrArray(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false) {
    auto const& [stride_M, stride_N, stride_L] = params.dCol;
    // Nullptr default
    if (EnableNullptr && params.ptr_col_array == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    // Dynamic non-batched scalar broadcast
    else if (IsDynamicBroadcast && stride_M == bool(0) && stride_L == repeat_like(stride_L, 0)) {
      is_zero_ = params.ptr_col_array[0][0] == ElementInput(0);
    }
  }

  Params params;
  bool is_zero_;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class GTensor, class RTensor, class CTensor, class ThrResidue>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(GTensor tCgCol_, RTensor tCrCol_, CTensor tCcCol_, ThrResidue residue_tCcCol_, Params const& params_)
      : tCgCol(tCgCol_),
        tCrCol(tCrCol_),
        tCcCol(tCcCol_),
        residue_tCcCol(residue_tCcCol_),
        params(params_) {
      if (EnableNullptr && params.ptr_col_array == nullptr) {
        fill(tCrCol, params.null_default);
      }
    }

    GTensor tCgCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    RTensor tCrCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    CTensor tCcCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    ThrResidue residue_tCcCol;
    Params const& params;

    CUTLASS_DEVICE void
    begin() {
      if (EnableNullptr && params.ptr_col_array == nullptr) {
        return;
      }

      // Filter so we don't issue redundant copies over stride-0 modes
      // (only works if 0-strides are in same location, which is by construction)
      Tensor tCgCol_flt = filter_zeros(tCgCol);
      Tensor tCrCol_flt = make_tensor_like<ElementInput>(filter_zeros(tCrCol));
      Tensor tCcCol_flt = filter_zeros(tCcCol, tCgCol.stride());

      constexpr auto MCL = decltype(max_common_layout(tCgCol_flt, tCrCol_flt)){};
      constexpr int V = cute::min(Alignment, size(MCL));
      if constexpr (V > 1) {
        using VecType = uint_bit_t<V * sizeof_bits_v<ElementInput>>;
        Tensor tCgCol_vec = recast<VecType>(coalesce(tCgCol_flt));
        Tensor tCrCol_vec = recast<VecType>(coalesce(tCrCol_flt));
        Tensor tCcCol_vec = tensor<1>(zipped_divide(tCcCol_flt, MCL.compose(Int<V>{})));
        auto pred_fn = [&] (auto const&... coords) { return elem_less(tCcCol_vec(coords...), residue_tCcCol); };
        copy_if(pred_fn, tCgCol_vec, tCrCol_vec);
      }
      else {
        auto pred_fn = [&] (auto const&... coords) { return elem_less(tCcCol_flt(coords...), residue_tCcCol); };
        copy_if(pred_fn, tCgCol_flt, tCrCol_flt);
      }

      constexpr int FrgSize = size(tCrCol_flt);
      using FrgInput = Array<ElementInput, FrgSize>;
      using FrgCompute = Array<ElementCompute, FrgSize>;
      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FrgSize>;

      Tensor tCrCol_input_frg = recast<FrgInput>(coalesce(tCrCol_flt));
      Tensor tCrCol_compute_frg = recast<FrgCompute>(filter(tCrCol));
      ConvertInput convert_input{};

      tCrCol_compute_frg(_0{}) = convert_input(tCrCol_input_frg(_0{}));
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_col;
      Tensor tCrCol_mn = tCrCol(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_col[i] = tCrCol_mn(epi_v * FragmentSize + i);
      }

      return frg_col;
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto layout_M = [&] () {
      auto shape_M = get<0>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_M = repeat_like(shape_M, int(0));
        if (get<0>(params.dCol) == bool(1)) {
          stride_M = transform_leaf(compact_major<LayoutLeft>(shape_M),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_M, stride_M);
      }
      else {
        return make_layout(shape_M);
      }
    }();

    auto layout_N = make_layout(N, repeat_like(N, _0{}));
    auto layout_L = make_layout(L, get<2>(params.dCol));
    Tensor mCol = make_tensor(make_gmem_ptr(params.ptr_col_array[l]), make_layout(layout_M,layout_N,layout_L));
    Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      mCol, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);

    Tensor mCol_static = make_tensor(make_gmem_ptr(params.ptr_col_array[l]), make_layout(make_layout(M),layout_N,layout_L));
    Tensor tCgCol_static = sm90_partition_for_epilogue<ReferenceSrc>(                  // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      mCol_static, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrCol = make_tensor_like<ElementCompute>(tCgCol_static);                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    return ConsumerStoreCallbacks(tCgCol, tCrCol, args.tCcD, args.residue_tCcD, params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Do outer product from the column and row loaded
//
template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementScalar,
  class StrideColMNL_ = Stride<_1,_0,int64_t>, /// NOTE: Batched scaling untested for now
  class StrideRowMNL_ = Stride<_0,_1,int64_t>,
  int Alignment = 128 / sizeof_bits_v<ElementScalar>,
  bool EnableNullptr = false // Fallback scalar broadcast for nullptr params
>
struct Sm90OuterProduct {
  using StrideColMNL = StrideColMNL_;
  using StrideRowMNL = StrideRowMNL_;
  static_assert(Stages == 0, "OuterProduct doesn't support smem usage");
  static_assert(Alignment * sizeof_bits_v<ElementScalar> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(!EnableNullptr, "Nullptr fallback not implemented");
  static_assert(is_static_v<decltype(take<0,2>(StrideColMNL{}))> &&
                is_static_v<decltype(take<0,2>(StrideRowMNL{}))>, "Only batch stride can be dynamic");
  static_assert(take<0,2>(StrideColMNL{}) == Stride<_1,_0>{} &&
                take<0,2>(StrideRowMNL{}) == Stride<_0,_1>{}, "Row and column incorrectly formatted");

  // Accumulator distributes col/row elements evenly amongst threads so we can just directly load from gmem
  struct SharedStorage { };

  struct Arguments {
    ElementScalar const* ptr_col = nullptr;
    ElementScalar const* ptr_row = nullptr;
    StrideColMNL dCol = {};
    StrideRowMNL dRow = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90OuterProduct() { }

  CUTLASS_HOST_DEVICE
  Sm90OuterProduct(Params const& params, SharedStorage const& shared_storage)
  : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<
    class GTensorCol, class RTensorCol,
    class GTensorRow, class RTensorRow
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(GTensorCol&& tCgCol, RTensorCol&& tCrCol,
                           GTensorRow&& tCgRow, RTensorRow&& tCrRow,
                           Params const& params)
      : tCgCol(cute::forward<GTensorCol>(tCgCol))
      , tCrCol(cute::forward<RTensorCol>(tCrCol))
      , tCgRow(cute::forward<GTensorRow>(tCgRow))
      , tCrRow(cute::forward<RTensorRow>(tCrRow))
      , params(params) {}

    GTensorCol tCgCol;                                                                 // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    RTensorCol tCrCol;                                                                 // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    GTensorRow tCgRow;                                                                 // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    RTensorRow tCrRow;                                                                 // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    Params const& params;

    CUTLASS_DEVICE void
    begin() {

      // Filter so we don't issue redundant copies over stride-0 modes
      copy(filter(tCgCol), filter(tCrCol));
      copy(filter(tCgRow), filter(tCrRow));
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementScalar, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementScalar, FragmentSize> frg_colrow;
      Tensor tCrCol_mn = tCrCol(_,_,_,epi_m,epi_n);
      Tensor tCrRow_mn = tCrRow(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_colrow[i] = static_cast<ElementScalar>(tCrCol_mn(epi_v * FragmentSize + i) * tCrRow_mn(epi_v * FragmentSize + i));
      }
      return frg_colrow;
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    Tensor mCol = make_tensor(make_gmem_ptr(params.ptr_col), make_shape(M,N,L), params.dCol);
    Tensor mRow = make_tensor(make_gmem_ptr(params.ptr_row), make_shape(M,N,L), params.dRow);
    Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      mCol, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCgRow = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      mRow, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrCol = make_tensor_like(tCgCol);                                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    Tensor tCrRow = make_tensor_like(tCgRow);                                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    return ConsumerStoreCallbacks<
      decltype(tCgCol), decltype(tCrCol),
      decltype(tCgRow), decltype(tCrRow)
    >(
      cute::move(tCgCol), cute::move(tCrCol),
      cute::move(tCgRow), cute::move(tCrRow),
      params
    );
  }

};



}
