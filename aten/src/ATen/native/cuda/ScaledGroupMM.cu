#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

// Two warnings in Cutlass included header files
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

#include <ATen/native/cuda/GroupMMCommon.cuh>

#include <cute/tensor.hpp>
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
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

C10_DIAGNOSTIC_POP()
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

  auto make_strides = [](at::IntArrayRef strides) -> Strides {
    Strides out;
    std::copy(strides.begin(), strides.end(), out.begin());
    return out;
  };

  Strides tensor_StrideA = make_strides(mat_a.strides());
  Strides tensor_StrideB = make_strides(mat_b.strides());
  Strides tensor_StrideOutput = make_strides(out.strides());
  Strides tensor_ShapeA = make_strides(mat_a.sizes());
  Strides tensor_ShapeB = make_strides(mat_b.sizes());

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
      tensor_ShapeA,
      tensor_ShapeB,
      a_scale_stride,
      b_scale_stride);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  //   auto buf_cpu = mat_a.new_empty(
  //       input_args_size,
  //       at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  //   AT_CUDA_CHECK(cudaMemcpy(
  //       (char*)buf_cpu.data_ptr(),
  //       buf_ptr,
  //       input_args_size,
  //       cudaMemcpyDeviceToHost));
  //   char* buf_ptr_cpu = (char*)buf_cpu.data_ptr();
  //   DtypeA** inputA_ptrs_h = reinterpret_cast<DtypeA**>(buf_ptr_cpu);
  //   DtypeB** inputB_ptrs_h =
  //       reinterpret_cast<DtypeB**>(inputA_ptrs_h + aligned_group_count);
  //   DtypeOutput** output_ptrs_h =
  //       reinterpret_cast<DtypeOutput**>(inputB_ptrs_h + aligned_group_count);
  //   DtypeScale** inputA_scale_ptrs_h =
  //       reinterpret_cast<DtypeScale**>(output_ptrs_h + aligned_group_count);
  //   DtypeScale** inputB_scale_ptrs_h =
  //       reinterpret_cast<DtypeScale**>(inputA_scale_ptrs_h +
  //       aligned_group_count);
  //   StrideA* stride_A_h =
  //       reinterpret_cast<StrideA*>(inputB_scale_ptrs_h +
  //       aligned_group_count);
  //   StrideB* stride_B_h = reinterpret_cast<StrideB*>(stride_A_h +
  //   group_count); StrideOutput* stride_output_h =
  //       reinterpret_cast<StrideOutput*>(stride_B_h + group_count);
  //   ProblemShape::UnderlyingProblemShape* problem_sizes_h =
  //       reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
  //           stride_output_h + group_count);

  //   std::cout << "PTRS " << mat_a.data_ptr() << " " << mat_b.data_ptr() << "
  //   "
  //             << out.data_ptr() << " " << scale_a.data_ptr() << " "
  //             << scale_b.data_ptr() << "\n";
  //   for (int i = 0; i < group_count; i++) {
  //     std::cout << "A " << (void*)inputA_ptrs_h[i] << "\n";
  //     std::cout << "B " << (void*)inputB_ptrs_h[i] << "\n";
  //     std::cout << "O " << (void*)output_ptrs_h[i] << "\n";
  //     std::cout << "A_scale " << (void*)inputA_scale_ptrs_h[i] << "\n";
  //     std::cout << "B_scale " << (void*)inputB_scale_ptrs_h[i] << "\n";
  //     std::cout << "sizes " << problem_sizes_h[i] << "\n";
  //     std::cout << "strideA" << stride_A_h[i] << "\n";
  //     std::cout << "strideB" << stride_B_h[i] << "\n";
  //     std::cout << "stride_output" << stride_output_h[i] << "\n";
  //   }
  //   int device_id = 0;
  //   cutlass::KernelHardwareInfo kernel_hw_info =
  //   cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

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

} // namespace

#endif

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
  dispatch_fp8_grouped_gemm_on_bias_dtype(
      mat_a, mat_b, scale_a, scale_b, offs, bias, use_fast_accum, out);
#else
  TORCH_CHECK(false, "grouped mm is not supported on your system");
#endif
}

} // namespace at::cuda::detail
