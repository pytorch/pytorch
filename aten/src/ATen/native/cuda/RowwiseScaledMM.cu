#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

// Determine if the architecture supports rowwise scaled mm
// Currenlty failing on windows with: https://github.com/NVIDIA/cutlass/issues/1571
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000

#define BUILD_ROWWISE_FP8_KERNEL
#endif

#if defined(BUILD_ROWWISE_FP8_KERNEL)

// We are going to override the cuTensorMapEncodeTiled driver api with our lazy loader
static CUresult CUDAAPI nvrtc_cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap,
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,
    void* globalAddress,
    const cuuint64_t* globalDim,
    const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim,
    const cuuint32_t* elementStrides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  return at::globalContext().getNVRTC().cuTensorMapEncodeTiled(
      tensorMap,
      tensorDataType,
      tensorRank,
      globalAddress,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      interleave,
      swizzle,
      l2Promotion,
      oobFill);
}


#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/util/host_tensor.h>

// Rename the global function symbol
#define cuTensorMapEncodeTiled nvrtc_cuTensorMapEncodeTiled
#include <cute/tensor.hpp>
#undef cuTensorMapEncodeTiled
// Set everything back to normal

#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>


namespace {
// Cutlass rowwise kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    bool FAST_ACCUM,
    bool USE_BIAS,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
void f8f8bf16_rowwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out) {
  int M = XQ.size(0);
  int N = WQ.size(1);
  int K = XQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(
      WQ.is_cuda() && WQ.ndimension() == 2 && WQ.stride(1) == WQ.size(0) &&
      WQ.stride(0) == 1);

  // auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = INPUT_DTYPE;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(ElementInputA);

  using ElementInputB = cutlass::float_e4m3_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(ElementInputB);

  using ElementBias = BIAS_DTYPE;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(ElementOutput);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using KernelSchedule = cutlass::gemm::collective::
      KernelScheduleAuto; // Kernel to launch based on the default setting in
                          // the Collective Builder

  // Implement rowwise scaling epilogue.
  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      PONG ? 2 : 1,
      TileShape,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      PONG ? 2 : 1,
      TileShape,
      ElementBias,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementComputeEpilogue, // First stage output type.
      ElementComputeEpilogue, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      cute::conditional_t< // Second stage output type.
          USE_BIAS,
          ElementBias,
          ElementOutput>,
      ElementComputeEpilogue, // Second stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  using ComputeBias = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus,
      ElementOutput, // Final (optional) stage output type.
      ElementBias, // Final stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeBias =
      cutlass::epilogue::fusion::Sm90EVT<ComputeBias, Bias, EVTCompute1>;

  using EpilogueEVT =
      cute::conditional_t<USE_BIAS, EVTComputeBias, EVTCompute1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          cutlass::epilogue::TmaWarpSpecialized,
          EpilogueEVT>::CollectiveOp;

  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using FastDefaultSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using FastPongSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using SlowAccum = cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;
  using FastAccum =
      cute::conditional_t<PONG, FastPongSchedule, FastDefaultSchedule>;
  using MainLoopSchedule =
      cute::conditional_t<FAST_ACCUM, FastAccum, SlowAccum>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementInputB,
          LayoutInputB,
          AlignmentInputB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, 1));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {reinterpret_cast<ElementInputA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b},
      {{}, // Epilogue thread we populate below.
       (ElementOutput*)out.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)out.data_ptr<at::BFloat16>(),
       stride_output}};

  if constexpr (USE_BIAS) {
    arguments.epilogue.thread = {
        {reinterpret_cast<ElementBias*>(bias.value().data_ptr())}, // bias
        // compute_1
        {
            {reinterpret_cast<ElementComputeEpilogue*>(
                x_scale.data_ptr())}, // x_scale
            // compute_0
            {
                {reinterpret_cast<ElementComputeEpilogue*>(
                    w_scale.data_ptr())}, // w_scale
                {}, // Accumulator
                {} // Multiplies
            },
            {}, // Multiplies
        },
        {}, // Plus
    };
  } else {
    arguments.epilogue.thread = {
        {reinterpret_cast<ElementComputeEpilogue*>(
            x_scale.data_ptr())}, // x_scale
        // compute_0
        {
            {reinterpret_cast<ElementComputeEpilogue*>(
                w_scale.data_ptr())}, // w_scale
            {}, // Accumulator
            {} // Multiplies
        },
        {}, // Multiplies
    };
  }

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.get());
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

// FP8 Rowwise Cutlass kernel dispatch.
enum class KernelMode { Small, Large, Default };

KernelMode get_kernel_mode(at::Tensor XQ, at::Tensor WQ) {
  auto M = XQ.size(0);
  auto K = XQ.size(1);
  auto N = WQ.size(0);
  // Use a large kernel if at least two shapes are large....
  bool use_large_kernel =
      ((M >= 2048 && K >= 2048) || (M >= 2048 && N >= 2048) ||
       (K >= 2048 && N >= 2048));
  if (M <= 128 || N <= 128) {
    return KernelMode::Small;
  } else if (use_large_kernel) {
    return KernelMode::Large;
  } else {
    return KernelMode::Default;
  }
}

template <typename InputDType, bool FastAccum, bool UseBias, typename BiasDType>
void dispatch_fp8_rowwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_rowwise_impl<
        64,
        128,
        128,
        2,
        1,
        1,
        false,
        FastAccum,
        UseBias,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, out);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_rowwise_impl<
        128,
        128,
        128,
        2,
        1,
        1,
        true,
        FastAccum,
        UseBias,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    return f8f8bf16_rowwise_impl<
        128,
        128,
        128,
        1,
        2,
        1,
        false,
        FastAccum,
        UseBias,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, out);
  }
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
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
      "Scale tensors must be float32.");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias.value().dtype() == at::kFloat ||
            bias.value().dtype() == at::kBFloat16,
        "Bias type must be bfloat16 or float32 if provided.");
  }
  // Extract problem size.
  int M = XQ.size(0);
  int N = WQ.size(1);
  int K = XQ.size(1);

  bool use_bias = bias.has_value();
  bool bf16_bias = use_bias && bias.value().dtype() == at::kBFloat16;

  // Templatize based on input dtype.
  bool use_e5m2 = XQ.dtype() == at::kFloat8_e5m2;
  TORCH_CHECK(WQ.dtype() == at::kFloat8_e4m3fn, "For RowWise scaling the second input is required to be a float8_e4m3fn dtype.");

  if (use_bias) {
    if (bf16_bias) {
      if (use_fast_accum) {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e5m2_t,
              true,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, out);
        } else {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e4m3_t,
              true,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, out);
        }
      } else {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e5m2_t,
              false,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, out);
        } else {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e4m3_t,
              false,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, out);
        }
      }
    } else {
      if (use_fast_accum) {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e5m2_t,
              true,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, out);
        } else {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e4m3_t,
              true,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, out);
        }
      } else {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e5m2_t,
              false,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, out);
        } else {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e4m3_t,
              false,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, out);
        }
      }
    }
  } else {
    if (use_fast_accum) {
      if (use_e5m2) {
        return dispatch_fp8_rowwise_kernel<
            cutlass::float_e5m2_t,
            true,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, out);
      } else {
        return dispatch_fp8_rowwise_kernel<
            cutlass::float_e4m3_t,
            true,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, out);
      }
    } else {
      if (use_e5m2) {
        return dispatch_fp8_rowwise_kernel<
            cutlass::float_e5m2_t,
            false,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, out);
      } else {
        return dispatch_fp8_rowwise_kernel<
            cutlass::float_e4m3_t,
            false,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, out);
      }
    }
  }
#else // BUILD_ROWWISE_FP8_KERNEL
  TORCH_CHECK(false, "Rowwise scaling is not currenlty supported on your device");
#endif
}

} // namespace at::cuda::detail
