#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

// Two warninngs in Cutlass included header files
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wset-but-not-used")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-parameter")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-variable")

// Determine if the architecture supports rowwise scaled mm
// Currently failing on windows with:
// https://github.com/NVIDIA/cutlass/issues/1571
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && \
    CUDA_VERSION >= 12000

#define BUILD_ROWWISE_FP8_KERNEL
#endif

#if defined(BUILD_ROWWISE_FP8_KERNEL)

#include <ATen/ops/empty.h>
#include <ATen/native/cuda/GroupMMCommon.cuh>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/version.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

// Added for SM100 support
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/tensor_ref.h>

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12080) && \
    (defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) ||      \
     defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED))
#define BUILD_SM100_KERNEL
#endif

C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

namespace at::cuda::detail {

GroupCountInfo get_group_count(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs) {
  int M = mat_a.size(-2);
  int K = mat_a.size(-1);
  int N = mat_b.size(-1);
  int group_count = 0;
  GroupMMInputMatrixType type{};

  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    // if both inputs are ragged, K is dynamic, M and N come from inputs
    group_count = offs->size(0);
    type = GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_2D_MatrixB_2D;

    // stack on the K dimension
    K = K / group_count;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    type = GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_2D_MatrixB_3D;
    // stack on the M dimension
    M = M / group_count;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    type = GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_3D_MatrixB_2D;
    // stack on the N dimension
    N = N / group_count;
  } else {
    // regular bmm
    group_count = mat_a.size(0);
    type = GroupMMInputMatrixType::GroupMMInputMatrixType_MatrixA_3D_MatrixB_3D;
  }

  return GroupCountInfo{M, N, K, group_count, type};
}

} // namespace at::cuda::detail

namespace {

using Strides = at::cuda::detail::Strides;

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

Strides make_strides(at::IntArrayRef strides) {
  Strides out;
  std::copy(strides.begin(), strides.end(), out.begin());
  return out;
};

namespace sm90_detail {

using DtypeScale = float;
using DtypeAccum = float;
using DtypeEpilogue = float;
using DtypeOutput = cutlass::bfloat16_t;

using Multiply = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies,
    DtypeEpilogue,
    DtypeEpilogue,
    cutlass::FloatRoundStyle::round_to_nearest>;

using Add = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus,
    DtypeEpilogue,
    DtypeEpilogue,
    cutlass::FloatRoundStyle::round_to_nearest>;

using Cast = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::epilogue::thread::Identity,
    DtypeOutput,
    DtypeEpilogue,
    cutlass::FloatRoundStyle::round_to_nearest>;

using ProblemShape = cutlass::gemm::GroupProblemShape<
    cute::Shape<int32_t, int32_t, int32_t>>; // <M,N,K> per
                                             // group

template <
    bool FastAccum,
    bool PONG,
    typename TB_M,
    typename TB_N,
    typename TB_K>
struct Schedule {
  using FastCooperativeSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
  using CooperativeSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using FastPongSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using PongSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using PongEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using KernelSchedule = cute::conditional_t<
      PONG,
      cute::conditional_t<FastAccum, FastPongSchedule, PongSchedule>,
      cute::conditional_t<
          FastAccum,
          FastCooperativeSchedule,
          CooperativeSchedule>>;
  using EpilogueSchedule = cute::
      conditional_t<PONG, PongEpilogueSchedule, CooperativeEpilogueSchedule>;
  using TileShape = cute::Shape<TB_M, TB_N, TB_K>;
  using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;
};

template <
    typename FastAccum,
    typename BiasType,
    typename Pong,
    typename TB_M,
    typename TB_N,
    typename TB_K>
void f8f8bf16_grouped_gemm_impl_sm90(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
  using DtypeA = cutlass::float_e4m3_t;
  using DtypeB = cutlass::float_e4m3_t;
  using DtypeOutput = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 16 / sizeof(DtypeA);
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 16 / sizeof(DtypeB);
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  // Tag indicating the minimum SM that supports the intended feature
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using TileShape =
      typename Schedule<FastAccum::value, Pong::value, TB_M, TB_N, TB_K>::
          TileShape;
  using ClusterShape =
      typename Schedule<FastAccum::value, Pong::value, TB_M, TB_N, TB_K>::
          ClusterShape;
  using KernelSchedule =
      typename Schedule<FastAccum::value, Pong::value, TB_M, TB_N, TB_K>::
          KernelSchedule;
  using EpilogueSchedule =
      typename Schedule<FastAccum::value, Pong::value, TB_M, TB_N, TB_K>::
          EpilogueSchedule;
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      DtypeScale*,
      DtypeScale,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      DtypeScale*,
      DtypeScale,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using AccumScale = cutlass::epilogue::fusion::Sm90EVT<
      Multiply,
      ScaleB,
      cutlass::epilogue::fusion::Sm90EVT<Multiply, ScaleA, Accum>>;

  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<Cast, AccumScale>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          DtypeAccum,
          DtypeAccum,
          DtypeOutput,
          LayoutOutput*,
          AlignmentOutput,
          DtypeOutput,
          LayoutOutput*,
          AlignmentOutput,
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          DtypeA,
          LayoutA*,
          AlignmentA,
          DtypeB,
          LayoutB*,
          AlignmentB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;

  int32_t group_count;
  auto group_count_info = at::cuda::detail::get_group_count(mat_a, mat_b, offs);
  group_count = group_count_info.group_count;

  TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");

  const int64_t problem_shape_size =
      group_count * ((int64_t)sizeof(ProblemShape::UnderlyingProblemShape));

  const int64_t stride_size = 3 * group_count * ((int64_t)sizeof(StrideA));

  // dummy tmas are created based on these pointer-to-pointers
  // the actual values are never used, they are replaced
  // by real addresses, but for dummy tma creation to succeed
  // due to bug in cuda < 12.4 the pointers have to be aligned to 128 bits
  const int group_alignment = 16 / sizeof(void*);
  const int aligned_group_count =
      round_up_to_nearest_multiple(group_count, group_alignment);
  int64_t input_args_size = aligned_group_count * 5 * sizeof(void*) +
      problem_shape_size + stride_size;

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto input_buf = allocator.allocate(input_args_size);
  void* buf_ptr = input_buf.get();
  DtypeA** inputA_ptrs = reinterpret_cast<DtypeA**>(buf_ptr);
  DtypeB** inputB_ptrs =
      reinterpret_cast<DtypeB**>(inputA_ptrs + aligned_group_count);
  DtypeOutput** output_ptrs =
      reinterpret_cast<DtypeOutput**>(inputB_ptrs + aligned_group_count);
  DtypeScale** inputA_scale_ptrs =
      reinterpret_cast<DtypeScale**>(output_ptrs + aligned_group_count);
  DtypeScale** inputB_scale_ptrs =
      reinterpret_cast<DtypeScale**>(inputA_scale_ptrs + aligned_group_count);
  static_assert(
      sizeof(StrideA) == 8, "expected StrideA to be 8 bytes for alignment");
  StrideA* stride_A =
      reinterpret_cast<StrideA*>(inputB_scale_ptrs + aligned_group_count);
  StrideB* stride_B = reinterpret_cast<StrideB*>(stride_A + group_count);
  StrideOutput* stride_output =
      reinterpret_cast<StrideOutput*>(stride_B + group_count);
  ProblemShape::UnderlyingProblemShape* problem_sizes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          stride_output + group_count);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  Strides tensor_StrideA = make_strides(mat_a.strides());
  Strides tensor_StrideB = make_strides(mat_b.strides());
  Strides tensor_StrideOutput = make_strides(out.strides());
  // scale stride will be used inside the kernel only if needed,
  // so for 1d scales the "1" assigned here won't be used
  int64_t a_scale_stride = scale_a.stride(0);
  int64_t b_scale_stride = scale_b.stride(0);

  at::cuda::detail::prepare_grouped_gemm_data<<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeA*>(mat_a.data_ptr()),
      reinterpret_cast<DtypeB*>(mat_b.data_ptr()),
      reinterpret_cast<DtypeOutput*>(out.data_ptr()),
      scale_a.data_ptr<DtypeScale>(),
      scale_b.data_ptr<DtypeScale>(),
      inputA_ptrs,
      inputB_ptrs,
      output_ptrs,
      inputA_scale_ptrs,
      inputB_scale_ptrs,
      problem_sizes,
      stride_A,
      stride_B,
      stride_output,
      offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
      group_count_info,
      tensor_StrideA,
      tensor_StrideB,
      tensor_StrideOutput,
      a_scale_stride,
      b_scale_stride);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes, nullptr},
      {(const DtypeA**)inputA_ptrs,
       stride_A,
       (const DtypeB**)inputB_ptrs,
       stride_B},
      {{{{inputB_scale_ptrs}, {{inputA_scale_ptrs}, {}, {}}, {}}, {}},
       (const DtypeOutput**)output_ptrs,
       stride_output,
       output_ptrs,
       stride_output}};

  int sm_count =
      at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }
  arguments.hw_info.sm_count = sm_count;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = allocator.allocate(workspace_size);
  Gemm gemm;
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "cutlass cannot implement");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess,
      "cutlass cannot initialize");
  auto status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "cutlass cannot run, error ",
      int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FastAccum, typename BiasType>
void dispatch_fp8_grouped_gemm_on_tile_size(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {

  auto [M, N, K, group_count, type] = at::cuda::detail::get_group_count(mat_a, mat_b, offs);

  bool large =
      ((M >= 2048 && K >= 2048) || (M >= 2048 && N >= 2048) ||
       (K >= 2048 && N >= 2048));
  bool small = (M <= 128 || N <= 128);
  if (small) {
    f8f8bf16_grouped_gemm_impl_sm90<
        FastAccum,
        BiasType,
        /*Pong*/ std::true_type,
        cute::_64,
        cute::_128,
        cute::_128>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  } else if (large && FastAccum::value) {
    f8f8bf16_grouped_gemm_impl_sm90<
        FastAccum,
        BiasType,
        /*Pong*/ std::false_type,
        cute::_256,
        cute::_128,
        cute::_128>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  } else if (large) { // use smaller tile for slow accum to avoid spilling
    f8f8bf16_grouped_gemm_impl_sm90<
        FastAccum,
        BiasType,
        /*Pong*/ std::false_type,
        cute::_128,
        cute::_128,
        cute::_128>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);

  } else
    f8f8bf16_grouped_gemm_impl_sm90<
        FastAccum,
        BiasType,
        /*Pong*/ std::false_type,
        cute::_128,
        cute::_256,
        cute::_64>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
}

template <typename BiasType>
void dispatch_fp8_grouped_gemm_on_fast_accum(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
  if (use_fast_accum) {
    dispatch_fp8_grouped_gemm_on_tile_size<std::true_type, BiasType>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  } else {
    dispatch_fp8_grouped_gemm_on_tile_size<std::false_type, BiasType>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  }
}

void dispatch_fp8_grouped_gemm_on_bias_dtype(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
  if (bias.has_value() && bias->dtype() == at::kBFloat16) {
    dispatch_fp8_grouped_gemm_on_fast_accum<cutlass::bfloat16_t>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  } else {
    dispatch_fp8_grouped_gemm_on_fast_accum<float>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  }
}

} // namespace sm90_detail

#if defined(BUILD_SM100_KERNEL)

namespace sm100_detail {

struct Sm100ConfigSmall {
  using MmaTileShape = cute::Shape<cute::_64, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      /*SFVecSizeM*/ 1,
      /*SFVecSizeN*/ 128,
      /*SFVecSizeK*/ 128,
      /*majorSFA   */ cute::UMMA::Major::K,
      /*majorSFB   */ cute::UMMA::Major::MN>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

struct Sm100ConfigLargeM {
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      /*SFVecSizeM*/ 1,
      /*SFVecSizeN*/ 128,
      /*SFVecSizeK*/ 128,
      /*majorSFA   */ cute::UMMA::Major::K,
      /*majorSFB   */ cute::UMMA::Major::MN>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

using DtypeScale = float;
using DtypeAccum = float;
using DtypeEpilogue = float;
using DtypeOutput = cutlass::bfloat16_t;

template <typename Config, typename LayoutOutput>
void launch_gemm_sm100(at::Tensor mat_a,   // FP8
                       at::Tensor mat_b,   // FP8
                       at::Tensor scale_a, // FP32
                       at::Tensor scale_b, // FP32
                       std::optional<at::Tensor> offs, at::Tensor &out,
                       bool transpose = false) {
  using DtypeA = cutlass::float_e4m3_t;
  using DtypeB = cutlass::float_e4m3_t;
  using DtypeOutput = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<DtypeA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<DtypeB>::value;
  static constexpr int AlignmentOutput =
      128 / cutlass::sizeof_bits<DtypeOutput>::value;

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, typename Config::MmaTileShape,
          typename Config::ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, DtypeAccum,
          DtypeAccum, DtypeOutput, LayoutOutput *, AlignmentOutput, DtypeOutput,
          LayoutOutput *, AlignmentOutput,
          typename Config::EpilogueSchedule>::CollectiveOp;

  // Collective Mainloop
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, DtypeA,
          cute::tuple<cutlass::layout::RowMajor *,
                      typename Config::LayoutSFA *>,
          AlignmentA, DtypeB,
          cute::tuple<cutlass::layout::ColumnMajor *,
                      typename Config::LayoutSFB *>,
          AlignmentB, DtypeAccum, typename Config::MmaTileShape,
          typename Config::ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Config::KernelSchedule>::CollectiveOp;

  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;

  using DtypeProblemShape = cutlass::gemm::GroupProblemShape<
      cute::Shape<int, int, int>>::UnderlyingProblemShape;

  auto group_count_info = at::cuda::detail::get_group_count(mat_a, mat_b, offs);
  auto [M, N, K, group_count, type] = group_count_info;

  int aligned_group_count =
      round_up_to_nearest_multiple(group_count, 16 / int(sizeof(void *)));

  auto ptr_opts = at::TensorOptions().device(mat_a.device()).dtype(at::kLong);
  auto stride_opts =
      at::TensorOptions().device(mat_a.device()).dtype(at::kLong);
  auto shape_opts = at::TensorOptions().device(mat_a.device()).dtype(at::kInt);

  at::Tensor inputA_ptrs = at::empty({aligned_group_count}, ptr_opts);
  at::Tensor inputB_ptrs = at::empty({aligned_group_count}, ptr_opts);
  at::Tensor output_ptrs = at::empty({aligned_group_count}, ptr_opts);
  at::Tensor inputA_scale_ptrs = at::empty({aligned_group_count}, ptr_opts);
  at::Tensor inputB_scale_ptrs = at::empty({aligned_group_count}, ptr_opts);

  at::Tensor stride_A = at::empty({group_count}, stride_opts);
  at::Tensor stride_B = at::empty({group_count}, stride_opts);
  at::Tensor stride_output = at::empty({group_count}, stride_opts);
  at::Tensor problem_sizes = at::empty({group_count, 3}, shape_opts);

  int layout_bytes_as_int = static_cast<int>(sizeof(typename Config::LayoutSFA) / sizeof(int));

  auto layout_sfa = at::empty({group_count, layout_bytes_as_int}, shape_opts);
  auto layout_sfb = at::empty({group_count, layout_bytes_as_int}, shape_opts);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  Strides tensor_StrideA = make_strides(mat_a.strides());
  Strides tensor_StrideB = make_strides(mat_b.strides());
  Strides tensor_StrideOutput = make_strides(out.strides());
  Strides tensor_StrideSFA = make_strides(scale_a.strides());
  Strides tensor_StrideSFB = make_strides(scale_b.strides());

  at::cuda::detail::prepare_grouped_gemm_data_sm100<
      DtypeA,
      DtypeB,
      DtypeOutput,
      DtypeScale,
      DtypeProblemShape,
      StrideA,
      StrideB,
      StrideOutput,
      typename Config::LayoutSFA,
      typename Config::LayoutSFB,
      typename Config::ScaleConfig>
      <<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeA *>(mat_a.data_ptr()),
      reinterpret_cast<DtypeB *>(mat_b.data_ptr()),
      reinterpret_cast<DtypeOutput *>(out.data_ptr()),
      scale_a.data_ptr<DtypeScale>(),
      scale_b.data_ptr<DtypeScale>(),
      reinterpret_cast<DtypeA **>(inputA_ptrs.data_ptr()),
      reinterpret_cast<DtypeB **>(inputB_ptrs.data_ptr()),
      reinterpret_cast<DtypeOutput **>(output_ptrs.data_ptr()),
      reinterpret_cast<DtypeScale **>(inputA_scale_ptrs.data_ptr()),
      reinterpret_cast<DtypeScale **>(inputB_scale_ptrs.data_ptr()),
      reinterpret_cast<DtypeProblemShape *>(problem_sizes.data_ptr()),
      // Strides for cutlass, cute::Stride
      reinterpret_cast<StrideA *>(stride_A.data_ptr()),
      reinterpret_cast<StrideB *>(stride_B.data_ptr()),
      reinterpret_cast<StrideOutput *>(stride_output.data_ptr()),
      offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
      group_count_info,
      // Original strides of the input tensors
      tensor_StrideA,
      tensor_StrideB,
      tensor_StrideOutput,
      tensor_StrideSFA,
      tensor_StrideSFB,
      reinterpret_cast<typename Config::LayoutSFA *>(layout_sfa.data_ptr()),
      reinterpret_cast<typename Config::LayoutSFB *>(layout_sfb.data_ptr()),
      transpose);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const DtypeA**>(
          inputA_ptrs.data_ptr()), // ArrayElementA const* ptr_A{nullptr};
      static_cast<StrideA*>(stride_A.data_ptr()), // StrideA dA{};
      static_cast<const DtypeB**>(
          inputB_ptrs.data_ptr()), // ArrayElementB const* ptr_B{nullptr};
      static_cast<StrideB*>(stride_B.data_ptr()), //  StrideB dB{};
      static_cast<const DtypeScale**>(
          inputA_scale_ptrs
              .data_ptr()), // ElementAccumulator const* ptr_SFA{nullptr};
      reinterpret_cast<typename Config::LayoutSFA*>(
          layout_sfa.data_ptr()), // LayoutSFA layout_SFA{};
      static_cast<const DtypeScale**>(
          inputB_scale_ptrs
              .data_ptr()), // ElementAccumulator const* ptr_SFB{nullptr};
      reinterpret_cast<typename Config::LayoutSFB*>(
          layout_sfb.data_ptr()), // LayoutSFB layout_SFB{};
      {}, // RuntimeDataTypeA runtime_data_type_a{};
      {}}; // RuntimeDataTypeB runtime_data_type_b{};

  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, // typename FusionCallbacks::Arguments thread{}
      {}, // ElementC const** ptr_C = nullptr;
      {}, // StrideC dC{};
      static_cast<DtypeOutput**>(
          output_ptrs.data_ptr()), // ElementD** ptr_D = nullptr;
      static_cast<StrideOutput*>(stride_output.data_ptr())}; // StrideD dD{};

  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count,
       reinterpret_cast<DtypeProblemShape *>(problem_sizes.data_ptr()),
       nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  int sm_count =
      at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }
  arguments.hw_info.sm_count = sm_count;

  auto &allocator = *c10::cuda::CUDACachingAllocator::get();
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = allocator.allocate(workspace_size);
  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "cutlass cannot implement");
  TORCH_CHECK(gemm.initialize(arguments, workspace.get()) ==
                  cutlass::Status::kSuccess,
              "cutlass cannot initialize");
  auto status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass cannot run, error ",
              int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_fp8_grouped_gemm_size_sm100(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    std::optional<at::Tensor> offs,
    at::Tensor& out) {
  auto group_count_info = at::cuda::detail::get_group_count(mat_a, mat_b, offs);

  if (group_count_info.M > 2048 && group_count_info.K >= 2048) {
    launch_gemm_sm100<Sm100ConfigLargeM, cutlass::layout::RowMajor>(
        mat_a, mat_b, scale_a, scale_b, offs, out, false);
  } else {
    launch_gemm_sm100<Sm100ConfigSmall, cutlass::layout::RowMajor>(
        mat_a, mat_b, scale_a, scale_b, offs, out, false);
  }
}

inline void dispatch_fp8_grouped_gemm_on_bias_dtype(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias,
    bool use_fast_accum, // This is ignored for SM100
    at::Tensor& out) {
  if (bias.has_value()) {
    TORCH_CHECK(
        false,
        "Bias add is not yet supported for SM100 grouped GEMM (requested dtype: " +
            std::string(bias->dtype().name()));
  }
  dispatch_fp8_grouped_gemm_size_sm100(
      mat_a, mat_b, scale_a, scale_b, offs, out);
}

} // namespace sm100_detail

#endif // SM100 support guard

} // namespace

#endif // BUILD_ROWWISE_FP8_KERNEL guard

namespace at::cuda::detail {
void f8f8bf16_grouped_mm(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
#if defined(BUILD_ROWWISE_FP8_KERNEL)
  auto dprops = at::cuda::getCurrentDeviceProperties();

  if (dprops->major >= 10) {
#if defined(BUILD_SM100_KERNEL)
    sm100_detail::dispatch_fp8_grouped_gemm_on_bias_dtype(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
#else
    TORCH_CHECK(
        false,
        "Grouped MM for SM100+ requires supported device. Your build does not meet these requirements.");
#endif
  } else if (dprops->major >= 9) {
    sm90_detail::dispatch_fp8_grouped_gemm_on_bias_dtype(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
  } else {
    TORCH_CHECK(
        false,
        "Grouped MM is only supported on SM90 and SM100+ architectures.");
  }
#else
  TORCH_CHECK(false, "grouped mm is not supported on your system");
#endif
}

} // namespace at::cuda::detail
