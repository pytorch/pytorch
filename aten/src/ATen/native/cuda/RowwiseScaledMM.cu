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

template <bool PingPong, bool FastAccum>
struct Schedule;

template <>
struct Schedule</*PingPong=*/false, /*FastAccum=*/false> {
  using type = cutlass::gemm::KernelTmaWarpSpecialized;
};

template <>
struct Schedule</*PingPong=*/true, /*FastAccum=*/false> {
  using type = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
};

template <>
struct Schedule</*PingPong=*/false, /*FastAccum=*/true> {
  using type = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
};

template <>
struct Schedule</*PingPong=*/true, /*FastAccum=*/true> {
  using type = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
};

// Cutlass rowwise kernel
template <
    typename TileShape,
    typename ClusterShape,
    typename PingPong,
    typename FastAccum,
    typename DtypeA,
    typename DtypeB,
    typename DtypeBias>
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

  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);

  // Tag indicating the minimum SM that supports the intended feature
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  // Implement rowwise scaling epilogue.
  constexpr int ColBroadcastStages = 0;
  constexpr int RowBroadcastStages = PingPong::value ? 2 : 1;

  using XScale = cutlass::epilogue::fusion::
      Sm90ColBroadcast<ColBroadcastStages, TileShape, DtypeScale>;

  using WScale = cutlass::epilogue::fusion::
      Sm90RowBroadcast<RowBroadcastStages, TileShape, DtypeScale>;

  using Bias = cutlass::epilogue::fusion::
      Sm90RowBroadcast<RowBroadcastStages, TileShape, DtypeBias>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
      Cast,
      cutlass::epilogue::fusion::Sm90EVT<
          Add,
          Bias,
          cutlass::epilogue::fusion::Sm90EVT<
              Multiply,
              XScale,
              cutlass::epilogue::fusion::Sm90EVT<Multiply, WScale, Accum>>>>;

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
          cutlass::epilogue::TmaWarpSpecialized,
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
          typename Schedule<PingPong::value, FastAccum::value>::type>::
          CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

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
         {{reinterpret_cast<DtypeScale*>(x_scale.data_ptr())},
          {{reinterpret_cast<DtypeScale*>(w_scale.data_ptr())}}}}},
       reinterpret_cast<DtypeOutput*>(out.data_ptr()),
       stride_output,
       reinterpret_cast<DtypeOutput*>(out.data_ptr()),
       stride_output}};

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

template <typename... Types>
void dispatch_fp8_rowwise_kernel_on_tile_size(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    at::Tensor out) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_rowwise_impl<
        /*TileShape=*/cute::Shape<cute::_64, cute::_128, cute::_128>,
        /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
        /*PingPong=*/std::false_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_rowwise_impl<
        /*TileShape=*/cute::Shape<cute::_128, cute::_128, cute::_128>,
        /*ClusterShape=*/cute::Shape<cute::_2, cute::_1, cute::_1>,
        /*PingPong=*/std::true_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    return f8f8bf16_rowwise_impl<
        /*TileShape=*/cute::Shape<cute::_128, cute::_128, cute::_128>,
        /*ClusterShape=*/cute::Shape<cute::_1, cute::_2, cute::_1>,
        /*PingPong=*/std::false_type,
        Types...>(XQ, WQ, x_scale, w_scale, bias, out);
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
    dispatch_fp8_rowwise_kernel_on_tile_size<std::true_type, Types...>(
        XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    dispatch_fp8_rowwise_kernel_on_tile_size<std::false_type, Types...>(
        XQ, WQ, x_scale, w_scale, bias, out);
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
    dispatch_fp8_rowwise_kernel_on_input_dtypes<cutlass::bfloat16_t>(
        XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  } else {
    dispatch_fp8_rowwise_kernel_on_input_dtypes<float>(
        XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
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
    TORCH_CHECK(bias->dtype() == at::kFloat || bias->dtype() == at::kBFloat16);
    TORCH_CHECK(bias->dim() == 1);
    TORCH_CHECK(bias->size(0) == b.size(1));
    TORCH_CHECK(bias->stride(0) == 1);
  }

  TORCH_CHECK(out.device() == a.device());
  TORCH_CHECK(out.dtype() == at::kBFloat16);
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
      false, "Rowwise scaling is not currenlty supported on your device");
#endif
}

} // namespace at::cuda::detail
