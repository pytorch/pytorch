#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>


// Three warnings in Cutlass included header files
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wset-but-not-used")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-parameter")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-variable")

// Determine if the architecture supports rowwise scaled mm
// Currently failing on windows with:
// https://github.com/NVIDIA/cutlass/issues/1571
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && \
    CUDA_VERSION >= 12000

#define BUILD_GG_KERNEL
#endif

#if defined(BUILD_GG_KERNEL)

#include <cute/tensor.hpp>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/version.h>
#include <ATen/native/cuda/GroupMMCommon.cuh>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include <ATen/native/cuda/cutlass_common.cuh>

namespace {
using Strides = at::cuda::detail::Strides; // std::array<int64_t, 3>;

template <typename ArchTag, bool PONGOr2SM, typename TB_M, typename TB_N, typename TB_K>
struct Schedule {
  // SM90
  using CooperativeSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using PongEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  // SM100
  using MMA1SMKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using MMA1SMEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using MMA2SMKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using MMA2SMEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;

  using KernelSchedule = cute::conditional_t<std::is_same_v<ArchTag, cutlass::arch::Sm100>,
    cute::conditional_t<PONGOr2SM, MMA2SMKernelSchedule, MMA1SMKernelSchedule>,
    cute::conditional_t<PONGOr2SM, PongSchedule, CooperativeSchedule>>;
  using EpilogueSchedule = cute::conditional_t<std::is_same_v<ArchTag, cutlass::arch::Sm100>,
    cute::conditional_t<PONGOr2SM, MMA2SMEpilogueSchedule, MMA1SMEpilogueSchedule>,
    cute::conditional_t<PONGOr2SM, PongEpilogueSchedule, CooperativeEpilogueSchedule>>;

};

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

template <
    typename ArchTag,
    bool a_row_major,
    bool b_row_major,
    bool PONGOr2SM,
    typename TB_M,
    typename TB_N,
    typename TB_K>
void bf16bf16_grouped_gemm_impl_sm90_sm100(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out) {
  using DtypeA = cutlass::bfloat16_t;
  using DtypeB = cutlass::bfloat16_t;
  using DtypeOutput = cutlass::bfloat16_t;
  using DtypeAccum = float;
  using LayoutA = cute::conditional_t<
      a_row_major,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor>;
  constexpr int AlignmentA = 16 / sizeof(DtypeA);

  using LayoutB = cute::conditional_t<
      b_row_major,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor>;
  constexpr int AlignmentB = 16 / sizeof(DtypeB);
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<TB_M, TB_N, TB_K>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using KernelSchedule =
      typename Schedule<ArchTag, PONGOr2SM, TB_M, TB_N, TB_K>::KernelSchedule;
  using EpilogueSchedule =
      typename Schedule<ArchTag, PONGOr2SM, TB_M, TB_N, TB_K>::EpilogueSchedule;
  using ProblemShape = cutlass::gemm::GroupProblemShape<
      cute::Shape<int32_t, int32_t, int32_t>>; // <M,N,K> per
                                               // group

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
          cutlass::epilogue::fusion::
              LinearCombination<DtypeOutput, DtypeAccum>>::CollectiveOp;

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

  using GemmKernelBase = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using GemmKernel = std::conditional_t<
      std::is_same_v<ArchTag, cutlass::arch::Sm100>,
      at::cuda::detail::enable_3x_kernel_for_sm10<GemmKernelBase>,
      at::cuda::detail::enable_3x_kernel_for_sm9x<GemmKernelBase>>;

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
  int64_t input_args_size = aligned_group_count * 3 * sizeof(void*) +
      problem_shape_size + stride_size;

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto input_buf = allocator.allocate(input_args_size);
  void* buf_ptr = input_buf.get();
  DtypeA** inputA_ptrs = reinterpret_cast<DtypeA**>(buf_ptr);
  DtypeB** inputB_ptrs =
      reinterpret_cast<DtypeB**>(inputA_ptrs + aligned_group_count);
  DtypeOutput** output_ptrs =
      reinterpret_cast<DtypeOutput**>(inputB_ptrs + aligned_group_count);
  static_assert(
      sizeof(StrideA) == 8, "expected StrideA to be 8 bytes for alignment");
  StrideA* stride_A =
      reinterpret_cast<StrideA*>(output_ptrs + aligned_group_count);
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

  at::cuda::detail::prepare_grouped_gemm_data<<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeA*>(mat_a.data_ptr()),
      reinterpret_cast<DtypeB*>(mat_b.data_ptr()),
      reinterpret_cast<DtypeOutput*>(out.data_ptr()),
      static_cast<float*>(nullptr), // type for template inference
      static_cast<float*>(nullptr), // type for template inference
      inputA_ptrs,
      inputB_ptrs,
      output_ptrs,
      static_cast<float**>(nullptr), // type for template inference
      static_cast<float**>(nullptr), // type for template inference
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
      0,
      0,
      a_row_major,
      b_row_major);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes, nullptr},
      {(const DtypeA**)inputA_ptrs,
       stride_A,
       (const DtypeB**)inputB_ptrs,
       stride_B},
      {{},
       (const DtypeOutput**)output_ptrs,
       stride_output,
       output_ptrs,
       stride_output}};

  arguments.epilogue.thread.alpha = 1.0;
  arguments.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};

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

template <bool a_row_major, bool b_row_major>
void dispatch_bf16_grouped_kernel_on_tile_size(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
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
  //   bool large =
  //       ((M >= 2048 && K >= 2048) || (M >= 2048 && N >= 2048) ||
  //        (K >= 2048 && N >= 2048));
  bool small = (M <= 128 || N <= 128);
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool sm10x = properties != nullptr && properties->major == 10;

  if (sm10x) {
    if (small){
      bf16bf16_grouped_gemm_impl_sm90_sm100<
        cutlass::arch::Sm100,
        a_row_major,
        b_row_major,
        /*PONGOr2SM*/ false,
        cute::_128,
        cute::_256,
        cute::_64>(mat_a, mat_b, offs, bias, out); // Tile shape taken from CUTLASS examples, 64 = 128/sizeof(bfloat16)
    } else {
      bf16bf16_grouped_gemm_impl_sm90_sm100<
        cutlass::arch::Sm100,
        a_row_major,
        b_row_major,
        /*PONGOr2SM*/ true,
        cute::_256,
        cute::_256,
        cute::_64>(mat_a, mat_b, offs, bias, out); // Same as above ^
    }
  } else {
    if(small) {
      bf16bf16_grouped_gemm_impl_sm90_sm100<
        cutlass::arch::Sm90,
        a_row_major,
        b_row_major,
        /*PONGOr2SM*/ true,
        cute::_64,
        cute::_128,
        cute::_128>(mat_a, mat_b, offs, bias, out);
    } else {
      bf16bf16_grouped_gemm_impl_sm90_sm100<
        cutlass::arch::Sm90,
        a_row_major,
        b_row_major,
        /*PONGOr2SM*/ false,
        cute::_128,
        cute::_256,
        cute::_64>(mat_a, mat_b, offs, bias, out);
    }
  }
}

void dispatch_bf16_grouped_kernel_on_ab_transpose(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out) {
  // we already checked that one of the strides is 1
  bool a_row_major = mat_a.stride(-1) == 1;
  bool b_row_major = mat_b.stride(-1) == 1;
  if (a_row_major && b_row_major) {
    dispatch_bf16_grouped_kernel_on_tile_size<true, true>(
        mat_a, mat_b, offs, bias, out);
  } else if (a_row_major && !b_row_major) {
    dispatch_bf16_grouped_kernel_on_tile_size<true, false>(
        mat_a, mat_b, offs, bias, out);
  } else if (!a_row_major && b_row_major) {
    dispatch_bf16_grouped_kernel_on_tile_size<false, true>(
        mat_a, mat_b, offs, bias, out);
  } else {
    dispatch_bf16_grouped_kernel_on_tile_size<false, false>(
        mat_a, mat_b, offs, bias, out);
  }
}

} // namespace
#endif

namespace at::cuda::detail {

void bf16bf16_grouped_mm(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out) {
#if defined(BUILD_GG_KERNEL)
  dispatch_bf16_grouped_kernel_on_ab_transpose(mat_a, mat_b, offs, bias, out);
#else
  TORCH_CHECK(false, "grouped mm is not supported on your system");
#endif
}

} // namespace at::cuda::detail
