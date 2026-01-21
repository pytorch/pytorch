#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/macros/Macros.h>

// Two warnings in Cutlass included header files
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wset-but-not-used")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-parameter")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wmissing-field-initializers")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-variable")

// Determine if the architecture supports rowwise scaled mm
// Currently failing on windows with:
// https://github.com/NVIDIA/cutlass/issues/1571
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION)

#define BUILD_ROWWISE_FP8_KERNEL
#endif

#if defined(BUILD_ROWWISE_FP8_KERNEL)

#include <cute/tensor.hpp>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/version.h>

#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <ATen/native/cuda/cutlass_common.cuh>

C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

namespace {

using DtypeScale = float;
using DtypeAccum = float;
using DtypeEpilogue = float;

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

template <bool LargeTile, bool FastAccum>
struct Schedule;

template <>
struct Schedule</*LargeTile=*/false, /*FastAccum=*/false> {
  using type = cutlass::gemm::KernelTmaWarpSpecialized;
  using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
};

template <>
struct Schedule</*LargeTile=*/true, /*FastAccum=*/false> {
  // For a 128x128x128 tile with fastAccum = false, using
  // pingpong schedule will lead to spilling, and WarpSpecialized w/o pingpong
  // is slow
  using type = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using epilogue_type = cutlass::epilogue::TmaWarpSpecializedCooperative;
};

template <>
struct Schedule</*LargeTile=*/false, /*FastAccum=*/true> {
  using type = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
};

template <>
struct Schedule</*LargeTile=*/true, /*FastAccum=*/true> {
  using type = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using epilogue_type = cutlass::epilogue::TmaWarpSpecialized;
};

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

// Cutlass rowwise kernel for sm90
template <
    typename TileShape,
    typename ClusterShape,
    typename Transposed,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias,
    typename DtypeOutput>
void f8f8bf16_rowwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out,
    const int swizzle) {
  int M = XQ.size(0);
  int N = WQ.size(1);
  int K = XQ.size(1);

  // Workaround for https://github.com/pytorch/pytorch/issues/133334.
  if (M % 256 > 0) {
    int padded_M = ((M - 1) / 256 + 1) * 256;
    at::Tensor padded_x_scale = x_scale.new_empty({padded_M, 1});
    padded_x_scale.slice(/*dim=*/0, /*start=*/0, /*end=*/M)
        .copy_(std::move(x_scale));
    x_scale = std::move(padded_x_scale);
  }

  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(DtypeA);

  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(DtypeB);

  using LayoutOutput = std::conditional_t<
      Transposed::value,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor>;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  // Tag indicating the minimum SM that supports the intended feature
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  // Implement rowwise scaling epilogue.
  constexpr int ColBroadcastStages = 0;
  constexpr int RowBroadcastStages = 0;

  using XScale = cutlass::epilogue::fusion::
      Sm90ColBroadcast<ColBroadcastStages, TileShape, DtypeScale>;

  using WScale = cutlass::epilogue::fusion::
      Sm90RowBroadcast<RowBroadcastStages, TileShape, DtypeScale>;

  using Bias = std::conditional_t<
      Transposed::value,
      cutlass::epilogue::fusion::
          Sm90ColBroadcast<ColBroadcastStages, TileShape, DtypeBias>,
      cutlass::epilogue::fusion::
          Sm90RowBroadcast<RowBroadcastStages, TileShape, DtypeBias>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using AccumScale = cutlass::epilogue::fusion::Sm90EVT<
      Multiply,
      WScale,
      cutlass::epilogue::fusion::Sm90EVT<Multiply, XScale, Accum>>;

  using Cast = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::epilogue::thread::Identity,
      DtypeOutput,
      DtypeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
      Cast,
      cutlass::epilogue::fusion::Sm90EVT<
          Add,
          Bias,
          AccumScale>>;

  constexpr bool large_tile = std::is_same_v<TileShape, cute::Shape<cute::_128, cute::_128, cute::_128>>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          DtypeAccum,
          DtypeEpilogue,
          DtypeOutput,
          LayoutOutput,
          AlignmentOutput,
          DtypeOutput,
          LayoutOutput,
          AlignmentOutput,
          typename Schedule<large_tile, FastAccum::value>::epilogue_type,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          DtypeA,
          LayoutInputA,
          AlignmentInputA,
          DtypeB,
          LayoutInputB,
          AlignmentInputB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Schedule<large_tile, FastAccum::value>::type>::
          CollectiveOp;

  using GemmKernel = at::cuda::detail::enable_3x_kernel_for_sm9x<
      cutlass::gemm::kernel::GemmUniversal<
          cute::Shape<int, int, int>,
          CollectiveMainloop,
          CollectiveEpilogue>>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, static_cast<int>(XQ.stride(0)), 1));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, static_cast<int>(WQ.stride(1)), 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(M, static_cast<int>(out.stride(0)), 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {reinterpret_cast<DtypeA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<DtypeB*>(WQ.data_ptr()),
       stride_b},
      {{{{bias.has_value() ? reinterpret_cast<DtypeBias*>(bias->data_ptr())
                           : nullptr},
         {{reinterpret_cast<DtypeScale*>(w_scale.data_ptr())},
          {{reinterpret_cast<DtypeScale*>(x_scale.data_ptr())}}}}},
       reinterpret_cast<DtypeOutput*>(out.data_ptr()),
       stride_output,
       reinterpret_cast<DtypeOutput*>(out.data_ptr()),
       stride_output}};

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Ensure persistent kernels leave enough free SMs for NCCL background ops.
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    arguments.hw_info.sm_count =
        at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount -
        at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }

  // Set the swizzle size
  arguments.scheduler.max_swizzle_size = swizzle;

  // Allocate workspace memory
  auto workspace = XQ.new_empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// Cutlass rowwise kernel for SM100/SM120
template <
    typename ArchTag,
    typename TileShape,
    typename ClusterShape,
    typename Transposed,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias,
    typename DtypeOutput>
void f8f8bf16_rowwise_impl_sm100_sm120(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out,
    const int swizzle) {
  int M = XQ.size(0);
  int N = WQ.size(1);
  int K = XQ.size(1);

  // Workaround for https://github.com/pytorch/pytorch/issues/133334.
  if (M % 256 > 0) {
    int padded_M = ((M - 1) / 256 + 1) * 256;
    at::Tensor padded_x_scale = x_scale.new_empty({padded_M, 1});
    padded_x_scale.slice(/*dim=*/0, /*start=*/0, /*end=*/M)
        .copy_(std::move(x_scale));
    x_scale = std::move(padded_x_scale);
  }

  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(DtypeA);

  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(DtypeB);

  using LayoutOutput = std::conditional_t<
      Transposed::value,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor>;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  using OperatorClass = cutlass::arch::OpClassTensorOp;

  // Implement rowwise scaling epilogue.
  constexpr int ColBroadcastStages = 0;
  constexpr int RowBroadcastStages = 0;

  using XScale = cutlass::epilogue::fusion::
      Sm90ColBroadcast<ColBroadcastStages, TileShape, DtypeScale>;

  using WScale = cutlass::epilogue::fusion::
      Sm90RowBroadcast<RowBroadcastStages, TileShape, DtypeScale>;

  using Bias = std::conditional_t<
      Transposed::value,
      cutlass::epilogue::fusion::
          Sm90ColBroadcast<ColBroadcastStages, TileShape, DtypeBias>,
      cutlass::epilogue::fusion::
          Sm90RowBroadcast<RowBroadcastStages, TileShape, DtypeBias>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using AccumScale = cutlass::epilogue::fusion::Sm90EVT<
      Multiply,
      WScale,
      cutlass::epilogue::fusion::Sm90EVT<Multiply, XScale, Accum>>;

  using Cast = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::epilogue::thread::Identity,
      DtypeOutput,
      DtypeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
      Cast,
      cutlass::epilogue::fusion::Sm90EVT<
          Add,
          Bias,
          AccumScale>>;

  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      DtypeAccum, DtypeEpilogue,
      DtypeOutput, LayoutOutput, AlignmentOutput,
      DtypeOutput, LayoutOutput, AlignmentOutput,
      EpilogueScheduleType,
      EpilogueEVT>::CollectiveOp;

  // as of CUTLASS 3.9.2, on sm120, KernelScheduleAuto resolves to
  // KernelTmaWarpSpecializedCooperativeSm120<2>>,
  // which does not support TileShape.M < 128
  using MainloopScheduleType = std::conditional_t<
      std::is_same_v<ArchTag, cutlass::arch::Sm120> && cute::size<0>(TileShape{}) < 128,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong,
      cutlass::gemm::collective::KernelScheduleAuto>;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          DtypeA,
          LayoutInputA,
          AlignmentInputA,
          DtypeB,
          LayoutInputB,
          AlignmentInputB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduleType>::CollectiveOp;

  using GemmKernel = at::cuda::detail::enable_3x_kernel_for_sm10_or_later<
      cutlass::gemm::kernel::GemmUniversal<
          cute::Shape<int, int, int>,
          CollectiveMainloop,
          CollectiveEpilogue>>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, static_cast<int>(XQ.stride(0)), 1));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, static_cast<int>(WQ.stride(1)), 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(M, static_cast<int>(out.stride(0)), 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {reinterpret_cast<DtypeA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<DtypeB*>(WQ.data_ptr()),
       stride_b},
      {{{{bias.has_value() ? reinterpret_cast<DtypeBias*>(bias->data_ptr())
                           : nullptr},
         {{reinterpret_cast<DtypeScale*>(w_scale.data_ptr())},
          {{reinterpret_cast<DtypeScale*>(x_scale.data_ptr())}}}}},
       reinterpret_cast<DtypeOutput*>(out.data_ptr()),
       stride_output,
       reinterpret_cast<DtypeOutput*>(out.data_ptr()),
       stride_output}};

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Ensure persistent kernels leave enough free SMs for NCCL background ops.
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    arguments.hw_info.sm_count =
        at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount -
        at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }

  // Set the swizzle size
  arguments.scheduler.max_swizzle_size = swizzle;

  // Allocate workspace memory
  auto workspace = XQ.new_empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Cutlass rowwise kernel for SM89
template <
    typename ThreadblockShape,
    typename WarpShape,
    int NumStages,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias,
    typename DtypeOutput>
void f8f8bf16_rowwise_impl_sm89(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out) {
  int M = XQ.size(0);
  int N = WQ.size(1);
  int K = XQ.size(1);

  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(DtypeA);

  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(DtypeB);

  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  // Tag indicating the minimum SM that supports the intended feature
  using ArchTag = cutlass::arch::Sm89;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using Operator = std::conditional_t<
      FastAccum::value,
      cutlass::arch::OpMultiplyAddFastAccum,
      cutlass::arch::OpMultiplyAdd>;
  constexpr auto NumEVTEpilogueStages = 1;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          DtypeOutput,
          AlignmentOutput,
          NumEVTEpilogueStages>;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, DtypeScale,
      cute::Stride<cute::_1, cute::_0, int64_t>>;
  using XScaleArguments = typename XScale::Arguments;

  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, DtypeScale,
      cute::Stride<cute::_0, cute::_1, int64_t>>;
  using WScaleArguments = typename WScale::Arguments;

  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, DtypeBias,
      cute::Stride<cute::_0, cute::_1, int64_t>>;
  using BiasArguments = typename Bias::Arguments;

  using ApplyXScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, DtypeEpilogue, DtypeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyXScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyXScale,
      Accum,
      XScale>;

  using ApplyWScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, DtypeEpilogue, DtypeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyWScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyWScale,
      EVTApplyXScale,
      WScale>;

  using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, DtypeEpilogue, DtypeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyBias = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyBias,
      EVTApplyWScale,
      Bias>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, DtypeOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
      Output,
      EVTApplyBias>;

  using EVTKernel = at::cuda::detail::enable_2x_kernel_for_sm89<
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
          DtypeA, LayoutInputA, cutlass::ComplexTransform::kNone, AlignmentInputA,
          DtypeB, LayoutInputB, cutlass::ComplexTransform::kNone, AlignmentInputB,
          DtypeOutput, LayoutOutput, AlignmentOutput,
          DtypeAccum,
          DtypeEpilogue,
          OperatorClass,
          ArchTag,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EVTOutput,
          ThreadblockSwizzle,
          NumStages,
          Operator,
          NumEVTEpilogueStages>::GemmKernel>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;

  cutlass::gemm::GemmCoord problem_size(M, N, K);
  constexpr auto SplitKFactor = 1;

  XScaleArguments x_scale_arguments{
      (DtypeScale*)x_scale.data_ptr(),
      DtypeScale(1),
      {cute::_1{}, cute::_0{}, problem_size.m()}
  };
  WScaleArguments w_scale_arguments{
      (DtypeScale*)w_scale.data_ptr(),
      DtypeScale(1),
      {cute::_0{}, cute::_1{}, problem_size.n()}
  };
  BiasArguments bias_arguments{
      bias.has_value() ? reinterpret_cast<DtypeBias*>(bias->data_ptr()) : nullptr,
      DtypeBias(0),
      {cute::_0{}, cute::_1{}, problem_size.n()}
  };
  typename Output::Arguments output_arguments{
    (DtypeOutput*)out.data_ptr(),
    {problem_size.n(), cute::_1{}, problem_size.mn().product()}
  };
  typename EVTOutput::Arguments callback_arguments{
    {
      {
        {
          {},                 // Accum
          x_scale_arguments,  // XScale
          {}                  // ApplyXScale
        },                    // EVTApplyXScale
        w_scale_arguments,    // WScale
        {}                    // ApplyWScale
      },                      // EVTApplyWScale
      bias_arguments,         // Bias
      {}                      // ApplyBias
    },                        // EVTApplyBias
    output_arguments          // Output
  };                          // EVTOutput

  typename Gemm::Arguments arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    SplitKFactor,
    callback_arguments,           // arguments of EVT callbacks
    (DtypeA*)XQ.data_ptr(),
    (DtypeB*)WQ.data_ptr(),
    nullptr,                      // ptr C (unused)
    nullptr,                      // ptr D (unused)
    problem_size.mk().product(),  // batch stride A
    problem_size.nk().product(),  // batch stride B
    0,                            // batch stride C (unused)
    0,                            // batch stride D (unused)
    problem_size.k(),             // stride A
    problem_size.k(),             // stride B
    0,                            // stride C (unused)
    0);                           // stride D (unused)

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  auto workspace = XQ.new_empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ClusterShape, typename ArchTag, typename... Types>
void dispatch_fp8_rowwise_kernel_on_tile_size(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out,
    const int swizzle) {
  int M = XQ.size(0);
  int N = WQ.size(1);

  int smTarget = at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    smTarget -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }

  // We prefer to use smaller tiles (less wasted compute in case of padding),
  // but if this causes us to have more CUDA blocks than there are SMs on the
  // GPU then we'll hit wave quantization, hence we'll switch to larger tiles.
  const bool use_smaller_tiles = ceildiv(M, 64 * cute::get<0>(ClusterShape{})) *
          ceildiv(N, 128 * cute::get<1>(ClusterShape{})) <=
      smTarget / cute::size(ClusterShape{});

  if (use_smaller_tiles) {
    if constexpr (std::is_same_v<ArchTag, cutlass::arch::Sm90>) {
      return f8f8bf16_rowwise_impl<
          /*TileShape=*/cute::Shape<cute::_64, cute::_128, cute::_128>,
          ClusterShape,
          Types...>(XQ, WQ, x_scale, w_scale, bias, out, swizzle);
    } else {
      return f8f8bf16_rowwise_impl_sm100_sm120<
        ArchTag,
        /*TileShape=*/cute::Shape<cute::_64, cute::_128, cute::_128>,
        ClusterShape,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out, swizzle);
    }
  } else {
    if constexpr (std::is_same_v<ArchTag, cutlass::arch::Sm90>) {
      return f8f8bf16_rowwise_impl<
        /*TileShape=*/cute::Shape<cute::_128, cute::_128, cute::_128>,
        ClusterShape,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out, swizzle);
    } else {
      return f8f8bf16_rowwise_impl_sm100_sm120<
        ArchTag,
        /*TileShape=*/cute::Shape<cute::_128, cute::_128, cute::_128>,
        ClusterShape,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out, swizzle);
    }
  }
}

template <
    typename ClusterShape,
    typename Transposed,
    typename ArchTag,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias,
    typename DtypeOutput>
void handle_transposition(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out,
    const int swizzle=1) {
  if constexpr (!Transposed::value) {
    dispatch_fp8_rowwise_kernel_on_tile_size<
        ClusterShape,
        ArchTag,
        Transposed,
        FastAccum,
        DtypeA,
        DtypeB,
        DtypeBias,
        DtypeOutput>(XQ, WQ, x_scale, w_scale, bias, out, swizzle);
  } else {
    dispatch_fp8_rowwise_kernel_on_tile_size<
        ClusterShape,
        ArchTag,
        Transposed,
        FastAccum,
        DtypeB,
        DtypeA,
        DtypeBias,
        DtypeOutput>(WQ.t(), XQ.t(), w_scale.t(), x_scale.t(), bias, out.t(), swizzle);
  }
}

template <typename... Types>
void dispatch_fp8_rowwise_kernel_on_cluster_size_and_transpose(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out) {
  int M = XQ.size(0);
  int N = WQ.size(1);

  // All the tiles we use have sizes which are multiples of 64, hence any
  // non-multiple of 64 will get padded anyways. Let's round up to simplify.
  M = round_up_to_nearest_multiple(M, 64);
  N = round_up_to_nearest_multiple(N, 64);

  // Small/skinny shapes with odd multiples of 64.
  if (M == 64 && N >= 3072) {
    return handle_transposition<
        /*ClusterShape=*/cute::Shape<cute::_1, cute::_2, cute::_1>,
        /*Transposed=*/std::false_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  }
  if (N == 64 && M >= 3072) {
    return handle_transposition<
        /*ClusterShape=*/cute::Shape<cute::_1, cute::_2, cute::_1>,
        /*Transposed=*/std::true_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  }
  if (M == 192 && N >= 4096) {
    return handle_transposition<
        /*ClusterShape=*/cute::Shape<cute::_1, cute::_2, cute::_1>,
        /*Transposed=*/std::true_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  }
  if (N == 192 && M >= 4096) {
    return handle_transposition<
        /*ClusterShape=*/cute::Shape<cute::_1, cute::_2, cute::_1>,
        /*Transposed=*/std::false_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  }

  // Now to odd multiples of 128 (but only if not too large).
  if (M * N <= 4096 * 4096) {
    if (M % 256 > 0 && N % 256 == 0) {
      return handle_transposition<
          /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
          /*Transposed=*/std::true_type,
          Types...>(XQ, WQ, x_scale, w_scale, bias, out);
    }
    if (N % 256 > 0 && M % 256 == 0) {
      return handle_transposition<
          /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
          /*Transposed=*/std::false_type,
          Types...>(XQ, WQ, x_scale, w_scale, bias, out);
    }
  }
  if (M % 256 > 0 && N % 256 > 0) {
    if ((M <= N) ^ (M * N <= 1024 * 1024)) {
      return handle_transposition<
          /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
          /*Transposed=*/std::true_type,
          Types...>(XQ, WQ, x_scale, w_scale, bias, out);
    } else {
      return handle_transposition<
          /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
          /*Transposed=*/std::false_type,
          Types...>(XQ, WQ, x_scale, w_scale, bias, out);
    }
  }

  // General case for large tensors.

  // Large M, N, k
  if (M >= 4096 && N >= 4096) {
    if (M >= N){
          return handle_transposition<
          /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
          /*Transposed=*/std::false_type,
          Types...>(XQ, WQ, x_scale, w_scale, bias, out, 8);
    }
    return handle_transposition<
        /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
        /*Transposed=*/std::true_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out, 8);
  }
  if ((M <= N) ^ (M >= 2048 && N >= 2048)) {
    return handle_transposition<
        /*ClusterShape=*/cute::Shape<cute::_1, cute::_2, cute::_1>,
        /*Transposed=*/std::true_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    return handle_transposition<
        /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
        /*Transposed=*/std::true_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  }
}

template <typename... Types>
void dispatch_fp8_rowwise_kernel_sm89(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out) {
  int M = XQ.size(0);

  if (M <= 16) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<16, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<16, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else if (M <= 32) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<32, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<16, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else if (M <= 64) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<64, 64, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<32, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else if (M <= 256) {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<64, 128, 128>,
        /*WarpShape=*/cutlass::gemm::GemmShape<64, 64, 64>,
        /*NumStages=*/3,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    return f8f8bf16_rowwise_impl_sm89<
        /*ThreadblockShape=*/cutlass::gemm::GemmShape<128, 128, 64>,
        /*WarpShape=*/cutlass::gemm::GemmShape<64, 64, 64>,
        /*NumStages=*/5,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  }
}

template <typename... Types>
void dispatch_fp8_rowwise_kernel_on_sm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out) {
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool sm89 = properties != nullptr && properties->major == 8 && properties->minor == 9;
  const bool sm9x = properties != nullptr && properties->major == 9;
  const bool sm10x = properties != nullptr && properties->major == 10;
  const bool sm11x = properties != nullptr && properties->major == 11;
  const bool sm12x = properties != nullptr && properties->major == 12;
  if (!(sm89 || sm9x || sm10x || sm11x || sm12x)) {
    TORCH_CHECK(
        false, "Rowwise scaling is not currently supported on your device");
  }

  if (sm9x) {
    dispatch_fp8_rowwise_kernel_on_cluster_size_and_transpose<
      /*ArchTag=*/cutlass::arch::Sm90,
      Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else if (sm10x || sm11x) {
    dispatch_fp8_rowwise_kernel_on_cluster_size_and_transpose<
      /*ArchTag=*/cutlass::arch::Sm100,
      Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else if (sm12x) {
    // sm12x doesn't have multicast feature
    handle_transposition<
      /*ClusterShape=*/cute::Shape<cute::_1, cute::_1, cute::_1>,
      /*Transposed=*/std::false_type,
      /*ArchTag=*/cutlass::arch::Sm120,
      Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    dispatch_fp8_rowwise_kernel_sm89<Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  }
}

template <typename... Types>
void dispatch_fp8_rowwise_kernel_on_fast_accum(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    at::Tensor out) {
  if (use_fast_accum) {
    dispatch_fp8_rowwise_kernel_on_sm<
        std::true_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    dispatch_fp8_rowwise_kernel_on_sm<
        std::false_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  }
}

template <typename... Types>
void dispatch_fp8_rowwise_kernel_on_input_dtypes(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    at::Tensor out) {
  if (XQ.dtype() == at::kFloat8_e5m2) {
    dispatch_fp8_rowwise_kernel_on_fast_accum<
        cutlass::float_e5m2_t,
        cutlass::float_e4m3_t,
        Types...>(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  } else {
    dispatch_fp8_rowwise_kernel_on_fast_accum<
        cutlass::float_e4m3_t,
        cutlass::float_e4m3_t,
        Types...>(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  }
}

void dispatch_fp8_rowwise_kernel_on_bias_dtype(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    at::Tensor out) {
  if (bias.has_value() && bias->dtype() == at::kBFloat16) {
    dispatch_fp8_rowwise_kernel_on_input_dtypes<
        cutlass::bfloat16_t,
        cutlass::bfloat16_t>
        (XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  } else if (bias.has_value() && bias->dtype() == at::kHalf){
    TORCH_CHECK(out.dtype() == at::kHalf, "Output should be Float16 when bias is Float16");
    dispatch_fp8_rowwise_kernel_on_input_dtypes<
        cutlass::half_t,
        cutlass::half_t>
        (XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  } else {
    dispatch_fp8_rowwise_kernel_on_input_dtypes<
        float,
        cutlass::bfloat16_t>
        //Types...>
        (XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  }
}

void check_inputs(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& out) {
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.device() == b.device());
  TORCH_CHECK(scale_a.device() == a.device());
  TORCH_CHECK(scale_b.device() == b.device());

  TORCH_CHECK(a.dtype() == at::kFloat8_e4m3fn || a.dtype() == at::kFloat8_e5m2);
  TORCH_CHECK(b.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(scale_a.dtype() == at::kFloat);
  TORCH_CHECK(scale_b.dtype() == at::kFloat);

  TORCH_CHECK(a.dim() == 2);
  TORCH_CHECK(b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(0));
  TORCH_CHECK(scale_a.dim() == 2);
  TORCH_CHECK(scale_b.dim() == 2);
  TORCH_CHECK(scale_a.size(0) == a.size(0));
  TORCH_CHECK(scale_a.size(1) == 1);
  TORCH_CHECK(scale_b.size(0) == 1);
  TORCH_CHECK(scale_b.size(1) == b.size(1));

  TORCH_CHECK(a.stride(1) == 1);
  TORCH_CHECK(a.stride(0) >= a.size(1));
  TORCH_CHECK(b.stride(0) == 1);
  TORCH_CHECK(b.stride(1) >= b.size(0));
  TORCH_CHECK(scale_a.stride(0) == 1);
  TORCH_CHECK(scale_b.stride(1) == 1);

  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == b.device());
    TORCH_CHECK(bias->dtype() == at::kFloat || bias->dtype() == at::kBFloat16 || bias->dtype() == at::kHalf);
    TORCH_CHECK(bias->dim() == 1);
    TORCH_CHECK(bias->size(0) == b.size(1));
    TORCH_CHECK(bias->stride(0) == 1);
  }

  TORCH_CHECK(out.device() == a.device());
  TORCH_CHECK(out.dtype() == at::kBFloat16 || out.dtype() == at::kHalf);
  TORCH_CHECK(out.dim() == 2);
  TORCH_CHECK(out.size(0) == a.size(0));
  TORCH_CHECK(out.size(1) == b.size(1));
  TORCH_CHECK(out.stride(1) == 1);
  TORCH_CHECK(out.stride(0) >= out.size(1));
}

} // namespace

#endif // !defined(USE_ROCM)

namespace at::cuda::detail {
void f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
#if defined(BUILD_ROWWISE_FP8_KERNEL)
  check_inputs(XQ, WQ, x_scale, w_scale, bias, out);

  dispatch_fp8_rowwise_kernel_on_bias_dtype(
      XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
#else // BUILD_ROWWISE_FP8_KERNEL
  TORCH_CHECK(
      false, "Rowwise scaling is not currently supported on your device");
#endif
}

} // namespace at::cuda::detail
