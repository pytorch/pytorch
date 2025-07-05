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

C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

namespace {

using Strides = at::cuda::detail::Strides;

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

  int32_t M, N, K, group_count;

  M = mat_a.size(-2);
  K = mat_a.size(-1);
  N = mat_b.size(-1);

  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    // if both inputs are ragged, K is dynamic, M and N come from inputs
    group_count = offs->size(0);
    K = -1;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    M = -1;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    N = -1;
  } else {
    // regular bmm
    group_count = mat_a.size(0);
  }

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
      M,
      N,
      K,
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
  int32_t M, N, K, group_count;

  M = mat_a.size(-2);
  K = mat_a.size(-1);
  N = mat_b.size(-1);

  // below we assume that gemms are approx same size
  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    // if both inputs are ragged, K is dynamic, M and N come from inputs
    group_count = offs->size(0);
    K = K / group_count;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    M = M / group_count;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    N = N / group_count;
  }
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

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12080) && \
    (defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) ||      \
     defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED))

// The following section is adapted from fp8_blockwise_moe_kernel.cu to add
// SM100+ support. Note: Bias is not yet supported in this path.

using ProblemShapeSm100 =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
template <typename OutType, typename ScheduleConfig, typename LayoutD>
void launch_f8f8bf16_grouped_gemm_sm100(
    at::Tensor& out_ptrs,
    const at::Tensor& a_ptrs,
    const at::Tensor& b_ptrs,
    const at::Tensor& a_scales_ptrs,
    const at::Tensor& b_scales_ptrs,
    const at::Tensor& stride_a,
    const at::Tensor& stride_b,
    const at::Tensor& stride_c,
    const at::Tensor& layout_sfa,
    const at::Tensor& layout_sfb,
    const at::Tensor& problem_sizes,
    int group_count) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          typename ScheduleConfig::MmaTileShape,
          typename ScheduleConfig::ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          void, // ElementCompute
          LayoutC*,
          AlignmentC,
          ElementD,
          LayoutC*,
          AlignmentC,
          typename ScheduleConfig::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
          AlignmentA,
          ElementB,
          cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
          AlignmentB,
          ElementAccumulator,
          typename ScheduleConfig::MmaTileShape,
          typename ScheduleConfig::ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShapeSm100,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape =
      typename ProblemShapeSm100::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(
          layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(
          layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = at::cuda::getDeviceProperties(a_ptrs.device().index())
                         ->multiProcessorCount;

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr, // bias ptr
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  size_t workspace_size = Gemm::get_workspace_size(args);
  auto workspace = allocator.allocate(workspace_size);

  auto status = gemm_op.initialize(args, workspace.get(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOut,
    typename DtypeScale,
    typename ProblemShape,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
__global__ void prepare_grouped_gemm_data_sm100_kernel(
    DtypeA* mat_a_ptr,
    DtypeB* mat_b_ptr,
    DtypeOut* out_ptr,
    DtypeScale* scale_a_ptr,
    DtypeScale* scale_b_ptr,
    DtypeA** inputA_ptrs,
    DtypeB** inputB_ptrs,
    DtypeOut** output_ptrs,
    DtypeScale** inputA_scale_ptrs,
    DtypeScale** inputB_scale_ptrs,
    ProblemShape* problem_sizes,
    int64_t* stride_a,
    int64_t* stride_b,
    int64_t* stride_c,
    LayoutSFA* layout_sfa_ptr,
    LayoutSFB* layout_sfb_ptr,
    const int32_t* offs,
    int32_t M,
    int32_t N,
    int32_t K,
    Strides tensor_StrideA,
    Strides tensor_StrideB,
    Strides tensor_StrideOut,
    int64_t a_scale_stride,
    int64_t b_scale_stride) {
  int group_idx = blockIdx.x;

  int64_t stride_a_batch = tensor_StrideA[0];
  int64_t stride_b_batch = tensor_StrideB[0];
  int64_t stride_out_batch = tensor_StrideOut[0];

  int32_t M_i = M;
  int32_t N_i = N;
  int32_t K_i = K;

  if (offs) {
    if (K < 0) { // Both A and B are ragged
      const int32_t start_row = (group_idx == 0) ? 0 : offs[group_idx - 1];
      const int32_t end_row = offs[group_idx];
      M_i = end_row - start_row;
      K_i = tensor_StrideA[0] / M_i;
      inputA_ptrs[group_idx] = mat_a_ptr + start_row;
      inputB_ptrs[group_idx] = mat_b_ptr + (group_idx * N * K_i); // N is fixed
      inputA_scale_ptrs[group_idx] = scale_a_ptr + start_row;
    } else if (M < 0) { // Only A is ragged
      const int32_t start_row = (group_idx == 0) ? 0 : offs[group_idx - 1];
      const int32_t end_row = offs[group_idx];
      M_i = end_row - start_row;
      inputA_ptrs[group_idx] = mat_a_ptr + start_row * K_i;
      inputB_ptrs[group_idx] = mat_b_ptr + group_idx * stride_b_batch;
      inputA_scale_ptrs[group_idx] = scale_a_ptr + start_row;
    } else { // Only B is ragged
      const int32_t start_col = (group_idx == 0) ? 0 : offs[group_idx - 1];
      const int32_t end_col = offs[group_idx];
      N_i = end_col - start_col;
      inputA_ptrs[group_idx] = mat_a_ptr + group_idx * stride_a_batch;
      inputB_ptrs[group_idx] = mat_b_ptr + start_col * K_i;
      inputB_scale_ptrs[group_idx] = scale_b_ptr + start_col;
    }
    output_ptrs[group_idx] = out_ptr + (group_idx * N_i);
  } else { // Neither is ragged (standard BMM)
    inputA_ptrs[group_idx] = mat_a_ptr + group_idx * stride_a_batch;
    inputB_ptrs[group_idx] = mat_b_ptr + group_idx * stride_b_batch;
    output_ptrs[group_idx] = out_ptr + group_idx * stride_out_batch;
    inputA_scale_ptrs[group_idx] = scale_a_ptr + group_idx * a_scale_stride;
    inputB_scale_ptrs[group_idx] = scale_b_ptr + group_idx * b_scale_stride;
  }

  problem_sizes[group_idx] = {M_i, N_i, K_i};

  stride_a[group_idx] = tensor_StrideA[0];
  stride_b[group_idx] = tensor_StrideB[0];
  stride_c[group_idx] = tensor_StrideOut[0];

  layout_sfa_ptr[group_idx] =
      ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M_i, N_i, K_i, 1));
  layout_sfb_ptr[group_idx] =
      ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M_i, N_i, K_i, 1));
}

template <typename OutType>
void f8f8bf16_grouped_gemm_impl_sm100(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
  // SM100 path does not support bias yet
  TORCH_CHECK(
      !bias.has_value(), "Bias is not supported for SM100 grouped GEMM yet.");

  using DtypeA = cutlass::float_e4m3_t;
  using DtypeB = cutlass::float_e4m3_t;
  using DtypeOut = OutType;
  using DtypeScale = float;

  // Dispatch logic based on matrix sizes from fp8_blockwise_moe_kernel.cu
  struct MmaConfig1 {
    using MmaTileShape = cute::Shape<cute::_256, cute::_32, cute::_128>;
    using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
        128,
        1,
        128,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MmaConfig2 {
    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
        1,
        128,
        128,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MmaConfig3 {
    using MmaTileShape = cute::Shape<cute::_64, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
        1,
        128,
        128,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };

  int32_t M_val, N_val, K_val, group_count;
  M_val = mat_a.size(-2);
  K_val = mat_a.size(-1);
  N_val = mat_b.size(-1);
  const bool ragged_a = mat_a.dim() == 2;
  const bool ragged_b = mat_b.dim() == 2;
  const bool transpose_inputs = (M_val <= 2048 && K_val >= 2048);

  // Use transposed inputs for certain shapes for performance
  at::Tensor mat_a_final = transpose_inputs ? mat_a.t() : mat_a;
  at::Tensor mat_b_final = transpose_inputs ? mat_b.transpose(1, 2) : mat_b;
  at::Tensor out_final = transpose_inputs ? out.t() : out;
  at::Tensor scale_a_final = transpose_inputs ? scale_b : scale_a;
  at::Tensor scale_b_final = transpose_inputs ? scale_a : scale_b;

  if (ragged_a && ragged_b) {
    group_count = offs->size(0);
    K_val = -1;
    M_val = -1;
    N_val = -1;
  } else if (ragged_a) {
    group_count = mat_b_final.size(0);
    M_val = -1;
  } else if (ragged_b) {
    group_count = mat_a_final.size(0);
    N_val = -1;
  } else {
    group_count = mat_a_final.size(0);
  }

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  const int group_alignment = 16 / sizeof(void*);
  const int aligned_group_count =
      round_up_to_nearest_multiple(group_count, group_alignment);
  at::TensorOptions ptr_options =
      at::TensorOptions().device(mat_a.device()).dtype(at::kLong);
  at::TensorOptions int32_options =
      at::TensorOptions().device(mat_a.device()).dtype(at::kInt);
  at::TensorOptions int64_options =
      at::TensorOptions().device(mat_a.device()).dtype(at::kLong);

  at::Tensor a_ptrs = at::empty({aligned_group_count}, ptr_options);
  at::Tensor b_ptrs = at::empty({aligned_group_count}, ptr_options);
  at::Tensor out_ptrs = at::empty({aligned_group_count}, ptr_options);
  at::Tensor a_scales_ptrs = at::empty({aligned_group_count}, ptr_options);
  at::Tensor b_scales_ptrs = at::empty({aligned_group_count}, ptr_options);

  at::Tensor stride_a = at::empty({group_count}, int64_options);
  at::Tensor stride_b = at::empty({group_count}, int64_options);
  at::Tensor stride_c = at::empty({group_count}, int64_options);
  at::Tensor problem_sizes = at::empty({group_count, 3}, int32_options);

  at::Tensor layout_sfa = at::empty({group_count}, int32_options);
  at::Tensor layout_sfb = at::empty({group_count}, int32_options);

  // Prepare data on GPU
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto launch_prep_kernel = [&](auto scale_config) {
    using ScaleConfig = decltype(scale_config);
    using LayoutSFA = typename ScaleConfig::LayoutSFA;
    using LayoutSFB = typename ScaleConfig::LayoutSFB;
    prepare_grouped_gemm_data_sm100_kernel<
        DtypeA,
        DtypeB,
        DtypeOut,
        DtypeScale,
        typename ProblemShapeSm100::UnderlyingProblemShape,
        LayoutSFA,
        LayoutSFB,
        ScaleConfig><<<group_count, 1, 0, stream>>>(
        (DtypeA*)mat_a_final.data_ptr(),
        (DtypeB*)mat_b_final.data_ptr(),
        (DtypeOut*)out_final.data_ptr(),
        scale_a_final.data_ptr<DtypeScale>(),
        scale_b_final.data_ptr<DtypeScale>(),
        (DtypeA**)a_ptrs.data_ptr(),
        (DtypeB**)b_ptrs.data_ptr(),
        (DtypeOut**)out_ptrs.data_ptr(),
        (DtypeScale**)a_scales_ptrs.data_ptr(),
        (DtypeScale**)b_scales_ptrs.data_ptr(),
        (typename ProblemShapeSm100::UnderlyingProblemShape*)
            problem_sizes.data_ptr(),
        (int64_t*)stride_a.data_ptr(),
        (int64_t*)stride_b.data_ptr(),
        (int64_t*)stride_c.data_ptr(),
        reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
        reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),
        offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
        M_val,
        N_val,
        K_val,
        make_strides(mat_a_final.strides()),
        make_strides(mat_b_final.strides()),
        make_strides(out_final.strides()),
        scale_a_final.stride(0),
        scale_b_final.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };

  // Pick config and launch
  if (transpose_inputs) {
    launch_prep_kernel(typename MmaConfig1::ScaleConfig{});
    launch_f8f8bf16_grouped_gemm_sm100<
        OutType,
        MmaConfig1,
        cutlass::layout::ColumnMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        group_count);
    out = out_final.t();
  } else if (M_val > 2048 && K_val >= 2048) {
    launch_prep_kernel(typename MmaConfig2::ScaleConfig{});
    launch_f8f8bf16_grouped_gemm_sm100<
        OutType,
        MmaConfig2,
        cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        group_count);
  } else {
    launch_prep_kernel(typename MmaConfig3::ScaleConfig{});
    launch_f8f8bf16_grouped_gemm_sm100<
        OutType,
        MmaConfig3,
        cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        group_count);
  }
}
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
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12080) && \
    (defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) ||      \
     defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED))
    f8f8bf16_grouped_gemm_impl_sm100<cutlass::bfloat16_t>(
        mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
#else
    TORCH_CHECK(
        false,
        "Grouped MM for SM100+ requires supported device. Your build does not meet these requirements.");
#endif
  } else if (dprops->major >= 9) {
    dispatch_fp8_grouped_gemm_on_bias_dtype(
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
