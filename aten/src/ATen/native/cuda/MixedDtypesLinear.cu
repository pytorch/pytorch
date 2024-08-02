#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
// Doesn't work on ROCm or Windows yet
// TODO: Add compiler warning? Add PyTorch config flag?
#else
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>

#include <cutlass/gemm/device/gemm_universal_base.h>
#include <cutlass/gemm/kernel/default_gemm.h>

#include <ATen/native/cuda/cutlass_extensions/epilogue_helpers.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/kernel/fpA_intB_gemm.h>
#include <ATen/native/cuda/cutlass_extensions/gemm/threadblock/default_mma.h>
#endif

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
// Doesn't work on ROCm or Windows yet
#else
#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }
#endif

namespace at {
namespace native {

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
// Doesn't work on ROCm or Windows yet or old compiler
#else
template<typename ElementInputA, typename ElementInputB, typename EpilogueTag>
Tensor
mixed_dtypes_linear_cutlass(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias) {
  const int length_m = input.size(0);
  const int length_k = weight.size(0);
  const int length_n = scale.size(0);

  using ElementOutput = ElementInputA;

  using SmArch = cutlass::arch::Sm80;
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using Operator = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;

  constexpr auto ThreadblockK = 64;
  constexpr auto ElementsPerCacheLine = 128 * 8 / cutlass::sizeof_bits<ElementInputB>::value;
  constexpr auto ColumnsInterleaved   = ElementsPerCacheLine / ThreadblockK;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
  using LayoutOutput = LayoutInputA;

  constexpr auto ElementsPerAccessA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr auto ElementsPerAccessB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  constexpr auto ElementsPerAccessC = ElementsPerAccessA;
  constexpr auto Stages = 4;
  constexpr auto SplitKFactor = 1; // Wrong outputs if !=1, even if
                                   // GemmFpAIntB instantiated with
                                   // SplitKSerial set to false.

  // Check for current CUTLASS limitations w.r.t. weight sizes.
  TORCH_CHECK(length_k % 64 == 0 && length_n % 64 == 0,
              "mixed_dtypes_linear_dispatch_dtype: Number of rows/columns of "
              "the weight matrix must be divisible by ", 64);

  using ElementAccumulator = float;

  using EpilogueOp = typename fastertransformer::Epilogue<
      ElementOutput,
      ElementsPerAccessC,
      ElementAccumulator,
      EpilogueTag>::Op;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInputA,
      LayoutInputA,
      ElementsPerAccessA,
      ElementInputB,
      LayoutInputB,
      ElementsPerAccessB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      ThreadblockSwizzle,
      Stages,
      true,
      Operator>::GemmKernel;
  using GemmKernel = cutlass::gemm::kernel::GemmFpAIntB<
      typename DefaultGemmKernel::Mma,
      typename DefaultGemmKernel::Epilogue,
      typename DefaultGemmKernel::ThreadblockSwizzle,
      SmArch,
      DefaultGemmKernel::kSplitKSerial>;

  using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

  auto output = input.new_empty({length_m, length_n});

  const auto ldb = length_k * GemmKernel::kInterleave;

  typename Gemm::Arguments arguments(
      {length_m, length_n, length_k},
      {(ElementInputA*)input.data_ptr(), length_k},
      {(ElementInputB*)weight.data_ptr(), ldb},
      {(ElementInputA*)scale.data_ptr(), 0},
      {(ElementInputA*)(bias.numel() == 0 ? nullptr : bias.data_ptr()), 0},
      {(ElementOutput*)output.data_ptr(), length_n},
      SplitKFactor,
      {ElementAccumulator(1.f), ElementAccumulator(0.f)});

  Gemm gemm_op;

  cutlass::Status status;

  // Verify that GEMM operation with given arguments can be performed
  // by CUTLASS.
  status = gemm_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status);

  // Allocate workspace for CUTLASS mixed datatypes GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = input.new_empty({(int64_t)workspace_size},
                                  at::TensorOptions().dtype(at::kByte));

  // Initialize CUTLASS mixed datatypes GEMM object.
  status = gemm_op.initialize(arguments, workspace.data_ptr(),
                              at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  // Perform mixed datatypes GEMM operation.
  status = gemm_op.run(at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

template<typename ElementInputA, typename ElementInputB>
Tensor
mixed_dtypes_linear_dispatch_bias_activation(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias, const c10::string_view& activation) {
    if (bias.numel() == 0) {
      if (activation == "none") {
        return mixed_dtypes_linear_cutlass<
          ElementInputA,
          ElementInputB,
          fastertransformer::EpilogueOpNoBias>(input, weight, scale, bias);
      }
      AT_ERROR("mixed_dtypes_linear_dispatch_bias_activation: Activation \"",
               activation, "\" is not supported");
      return Tensor{};
    }
    else {
      if (activation == "none") {
        return mixed_dtypes_linear_cutlass<
            ElementInputA,
            ElementInputB,
            fastertransformer::EpilogueOpBias>(input, weight, scale, bias);
      } else if (activation == "relu") {
        return mixed_dtypes_linear_cutlass<
            ElementInputA,
            ElementInputB,
            fastertransformer::EpilogueOpBiasReLU>(input, weight, scale, bias);
      } else if (activation == "silu") {
        return mixed_dtypes_linear_cutlass<
            ElementInputA,
            ElementInputB,
            fastertransformer::EpilogueOpBiasSilu>(input, weight, scale, bias);
      }
      AT_ERROR("mixed_dtypes_linear_dispatch_bias_activation: Activation \"",
               activation, "\" is not supported");
      return Tensor{};
    }
}
#endif

Tensor
_mixed_dtypes_linear(const Tensor& input, const Tensor& weight,
                     const Tensor& scale,
                     const std::optional<Tensor>& bias_opt,
                     const std::optional<c10::string_view> activation_opt) {
#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
  AT_ERROR("_mixed_dtypes_linear: not compiled for this platform");
  return Tensor{};
#else
  const auto bias = bias_opt.has_value() ? *bias_opt : Tensor{};
  const auto activation = activation_opt.has_value() ? *activation_opt : "none";

  // For now, only CC 8.x devices are supported.
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm8x = dprops->major == 8;
  TORCH_CHECK(is_sm8x,
              "_mixed_dtypes_linear: Supported only on GPUs with compute "
              "capability 8.x");

  // Validate datatypes of input tensors.
  TORCH_CHECK(input.dtype() == at::kHalf ||
              input.dtype() == at::kBFloat16,
              "_mixed_dtypes_linear: The input datatype ", input.dtype(),
              " is not supported");
  TORCH_CHECK(weight.dtype() == at::kByte,
              "_mixed_dtypes_linear: The weight datatype ", weight.dtype(),
              " is not supported");
  TORCH_CHECK(scale.dtype() == input.dtype(),
              "_mixed_dtypes_linear: Expected scale datatype ", input.dtype(),
              " but got", scale.dtype());
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.dtype() == input.dtype(),
                "_mixed_dtypes_linear: Expected bias datatype ", input.dtype(),
                " but got", bias.dtype());
  }

  // Squash the batch dimensions of the input tensor with its
  // next-to-last dimensions.
  const auto input_sizes = input.sizes().vec();
  const auto input_2d = input.reshape({-1, input_sizes.back()});

  // Validate layouts of input tensors.
  TORCH_CHECK(input_2d.layout() == Layout::Strided,
              "_mixed_dtypes_linear: Expected input argument to be strided, "
              "but got layout ", input_2d.layout());
  TORCH_CHECK(input_2d.dim() == 2,
              "_mixed_dtypes_linear: Expected input argument to be 2D tensor, "
              "got ", input_2d.dim(), " dims");
  const auto strides_input = input_2d.strides();
  TORCH_CHECK(strides_input[0] > 1 && strides_input[1] == 1,
              "_mixed_dtypes_linear: Invalid strides for input argument: row "
              "stride = ", strides_input[0], ", column stride = ",
              strides_input[1]);
  TORCH_CHECK(weight.layout() == Layout::Strided,
              "_mixed_dtypes_linear: Expected input argument to be strided, "
              "but got layout ", weight.layout());
  TORCH_CHECK(weight.dim() == 2,
              "_mixed_dtypes_linear: Expected weight argument to be 2D tensor, "
              "got ", weight.dim(), " dims");
  const auto strides_weight = weight.strides();
  TORCH_CHECK(strides_weight[0] > 1 && strides_weight[1] == 1,
              "_mixed_dtypes_linear: Invalid strides for weight argument: row "
              "stride = ", strides_weight[0], ", column stride = ",
              strides_weight[1]);
  TORCH_CHECK(scale.dim() == 1,
              "_mixed_dtypes_linear: Expected scale argument to be 1D tensor, "
              "got ", scale.dim(), " dims");
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.dim() == 1,
                "_mixed_dtypes_linear: Expected bias argument to be 1D ",
                "tensor, got ", bias.dim(), " dims");
  }

  // Validate sizes of input tensors.
  TORCH_CHECK(input_2d.size(1) == weight.size(0),
              "_mixed_dtypes_linear: Expected input argument to have ",
              weight.size(0), " columns, but got ", input_2d.size(1));
  TORCH_CHECK(weight.size(1) == scale.size(0)  ||
              2 * weight.size(1) == scale.size(0),
              "_mixed_dtypes_linear: Expected weight argument to have either ",
              scale.size(0), " or ", scale.size(0) / 2.f, " columns, but got ",
              weight.size(1));
  if (bias.numel() != 0) {
      TORCH_CHECK(bias.size(0) == scale.size(0),
                  "_mixed_dtypes_linear: Expected bias argument to have ",
                  scale.size(0), " elements, but got ", bias.size(0));
  }

  Tensor output;
  auto scalar_type_quant = weight.scalar_type();
  if (weight.size(1) != scale.size(0)) {
    scalar_type_quant = at::ScalarType::QUInt4x2;
  }
  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "_mixed_dtypes_linear",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_linear",
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_linear_dispatch_bias_activation<
                              cutlass::half_t,
                              uint8_t>(input_2d, weight, scale, bias,
                                       activation);
                      return;
                    })
                AT_DISPATCH_CASE(
                    at::ScalarType::QUInt4x2,
                    [&]() {
                      output =
                          mixed_dtypes_linear_dispatch_bias_activation<
                              cutlass::half_t,
                              cutlass::uint4b_t>(input_2d, weight, scale, bias,
                                                 activation);
                      return;
                    }));
          })
      AT_DISPATCH_CASE(
          at::ScalarType::BFloat16,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_linear",
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_linear_dispatch_bias_activation<
                              cutlass::bfloat16_t,
                              uint8_t>(input_2d, weight, scale, bias,
                                       activation);
                      return;
                    })
                AT_DISPATCH_CASE(
                    at::ScalarType::QUInt4x2,
                    [&]() {
                      output =
                          mixed_dtypes_linear_dispatch_bias_activation<
                              cutlass::bfloat16_t,
                              cutlass::uint4b_t>(input_2d, weight, scale, bias,
                                                 activation);
                      return;
                    }));
          }));

  auto output_sizes = input_sizes;
  output_sizes.back() = scale.size(0);
  return output.reshape(output_sizes);
#endif
}

}  // namespace native
}  // namespace at
