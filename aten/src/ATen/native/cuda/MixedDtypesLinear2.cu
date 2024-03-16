#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/Dispatch.h>

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>

#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }

namespace {
  enum class Activation{NONE, RELU, SILU};
}
#endif

namespace at {
namespace native {

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
template<typename ElementInputA, typename ElementInputB, bool use_scale,
        bool use_bias, Activation activation>
Tensor mixed_dtypes_linear_cutlass(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias) {
  // Weight matrix is transposed implicitly, by considering that its
  // elements are given in column-major order in this method (the code
  // below still takes into accoun that it's not explicitly transposed
  // when inquring about its shape and strides.

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  const int length_m = input.size(0);
  const int length_k = input.size(1);
  const int length_n = weight.size(0);

  // Check for current CUTLASS limitations w.r.t. weight sizes.
  TORCH_CHECK(length_m % 16 == 0 && length_k % 16 == 0 && length_n % 16 == 0,
              "mixed_dtypes_linear_cutlass: Number of rows/columns of the "
              "operands must be divisible by ", 16);

  using ElementC = ElementInputA;
  using ElementScale = ElementInputA;
  using ElementBias = ElementInputA;
  using ElementAccumulator = float;
  using ElementEpilogue = float;
  using ElementOutput = ElementInputA;

  constexpr int AlignmentInputA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentInputB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentScale = 128 / cutlass::sizeof_bits<ElementScale>::value;
  constexpr int AlignmentBias = 128 / cutlass::sizeof_bits<ElementBias>::value;
  constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using SmArch = cutlass::arch::Sm80;
  using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape           = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 16>;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  constexpr auto NumStages = 3;

  constexpr auto NumEVTEpilogueStages = 1;

  using ScaleTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementScale,
      AlignmentScale,
      NumEVTEpilogueStages
  >;
  using BiasTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementBias,
      AlignmentBias,
      NumEVTEpilogueStages
  >;
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementOutput,
      AlignmentOutput,
      NumEVTEpilogueStages
  >;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using ScaleScalar =
      cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementScale>;
  using ScaleTensor =
      cutlass::epilogue::threadblock::VisitorAuxLoad<
          ScaleTileThreadMap,
          ElementScale,
          cute::Stride<int64_t, cute::_1, int64_t>>;
  using Scale = std::conditional_t<use_scale, ScaleTensor, ScaleScalar>;
  using ScaleArguments = typename Scale::Arguments;

  using ApplyScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyScale,
      Accum,
      Scale>;

  using BiasScalar =
      cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementBias>;
  using BiasTensor =
      cutlass::epilogue::threadblock::VisitorAuxLoad<
          BiasTileThreadMap,
          ElementBias,
          cute::Stride<int64_t, cute::_1, int64_t>>;
  using Bias = std::conditional_t<use_bias, BiasTensor, BiasScalar>;
  using BiasArguments = typename Bias::Arguments;

  using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyBias = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyBias,
      EVTApplyScale,
      Bias>;

  using ApplyActivationNone = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::epilogue::thread::Identity, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using ApplyActivationReLu = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::epilogue::thread::ReLu, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using ApplyActivationSiLu = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::epilogue::thread::SiLu, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using ApplyActivation =
      std::conditional_t<
          activation == Activation::NONE,
          ApplyActivationNone,
          std::conditional_t<
              activation == Activation::RELU,
              ApplyActivationReLu,
              ApplyActivationSiLu>>;
  using EVTApplyActivation = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyActivation,
      EVTApplyBias>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
      Output,
      EVTApplyActivation>;

  using EVTKernel =
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementInputA, LayoutInputA, cutlass::ComplexTransform::kNone, AlignmentInputA,
      ElementInputB, LayoutInputB, cutlass::ComplexTransform::kNone, AlignmentInputB,
      ElementC, LayoutC, AlignmentC,
      ElementAccumulator,
      ElementEpilogue,
      cutlass::arch::OpClassTensorOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EVTOutput,
      ThreadblockSwizzle,
      NumStages,
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      NumEVTEpilogueStages
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalBase<EVTKernel>;

  auto output = input.new_empty({length_m, length_n});

  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  constexpr auto SplitKFactor = 1;

  ScaleArguments scale_arguments{
    [&]() -> ScaleArguments {
      if constexpr (use_scale) {
        return {(ElementScale*)scale.data_ptr(),
                ElementScale(1),
                {cute::_0{}, cute::_1{}, problem_size.n()}};
      } else {
        return {ElementScale(1)};
      }
    }()
  };
  BiasArguments bias_arguments{
    [&]() -> BiasArguments {
      if constexpr (use_bias) {
        return {(ElementBias*)bias.data_ptr(),
                ElementBias(0),
                {cute::_0{}, cute::_1{}, problem_size.n()}};
      } else {
        return {ElementBias(0)};
      }
    }()
  };
  typename Output::Arguments output_arguments{
    (ElementOutput*)output.data_ptr(),
    {problem_size.n(), cute::_1{}, problem_size.mn().product()}
  };
  typename EVTOutput::Arguments callback_arguments{
    {
      {
        {
          {},                // Accum
          scale_arguments,   // Scale
          {}                 // ApplyScale
        },                   // EVTApplyScale
        bias_arguments,      // Bias
        {}                   // ApplyBias
      },                     // EVTApplyBias
      {}                     // ApplyActivation
    },                       // EVTApplyActivation
    output_arguments,        // Output
  };                         // EVTOutput
  constexpr auto AvailSms = -1;
  typename Gemm::Arguments arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    SplitKFactor,
    callback_arguments,                       // arguments of EVT callbacks
    (ElementInputA*)input.data_ptr(),
    (ElementInputB*)weight.data_ptr(),
    nullptr,                                  // ptr C (unused)
    nullptr,                                  // ptr D (unused)
    problem_size.mk().product(),              // batch stride A
    problem_size.nk().product(),              // batch stride B
    0,                                        // batch stride C (unused)
    0,                                        // batch stride D (unused)
    input.strides()[0],                       // stride A
    weight.strides()[0],                      // stride B
    0,                                        // stride C (unused)
    0,                                        // stride D (unused)
    AvailSms);

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

template<typename ElementInputA, typename ElementInputB, bool use_scale,
        bool use_bias>
Tensor mixed_dtypes_linear_cutlass_dispatch_activation(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias, const c10::string_view& activation) {
  if (activation == "none") {
    return mixed_dtypes_linear_cutlass<
        ElementInputA, ElementInputB, use_scale, use_bias, Activation::NONE>(
            input, weight, scale, bias);
  } else if (activation == "relu") {
    return mixed_dtypes_linear_cutlass<
        ElementInputA, ElementInputB, use_scale, use_bias, Activation::RELU>(
            input, weight, scale, bias);
  } else if (activation == "silu") {
    return mixed_dtypes_linear_cutlass<
        ElementInputA, ElementInputB, use_scale, use_bias, Activation::SILU>(
            input, weight, scale, bias);
  }

  AT_ERROR("mixed_dtypes_linear_cutlass_dispatch_activation: Activation \"",
           activation, "\" is not supported");
  return Tensor{};
}

template<typename ElementInputA, typename ElementInputB>
Tensor mixed_dtypes_linear_cutlass_dispatch_scale_bias(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias, const c10::string_view& activation) {
    if (scale.numel() > 0) {
        if (bias.numel() > 0) {
            return mixed_dtypes_linear_cutlass_dispatch_activation<
                       ElementInputA,
                       ElementInputB,
                       true,
                       true>(input, weight, scale, bias, activation);
        }
        else {
            return mixed_dtypes_linear_cutlass_dispatch_activation<
                       ElementInputA,
                       ElementInputB,
                       true,
                       false>(input, weight, scale, bias, activation);
        }
    }
    else {
        if (bias.numel() > 0) {
            return mixed_dtypes_linear_cutlass_dispatch_activation<
                       ElementInputA,
                       ElementInputB,
                       false,
                       true>(input, weight, scale, bias, activation);
        }
        else {
            return mixed_dtypes_linear_cutlass_dispatch_activation<
                       ElementInputA,
                       ElementInputB,
                       false,
                       false>(input, weight, scale, bias, activation);
        }
    }
}
#endif

Tensor
_mixed_dtypes_linear2(const Tensor& input, const Tensor& weight,
                      const c10::optional<Tensor>& scale_opt,
                      const c10::optional<Tensor>& bias_opt,
                      const c10::optional<c10::string_view> activation_opt) {
#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
  AT_ERROR("_mixed_dtypes_linear2: ROCm doesn't support CUTLASS");
  return Tensor{};
#else
  const auto scale = scale_opt.has_value() ? *scale_opt : Tensor{};
  const auto bias = bias_opt.has_value() ? *bias_opt : Tensor{};
  const auto activation = activation_opt.has_value() ? *activation_opt : "none";

  // For now, only CC 8.x devices are supported.
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm8x = dprops->major == 8;
  TORCH_CHECK(is_sm8x,
              "_mixed_dtypes_linear2: Supported only on GPUs with compute "
              "capability 8.x");

  // Validate datatypes of input tensors.
  TORCH_CHECK(input.dtype() == at::kHalf ||
              input.dtype() == at::kBFloat16,
              "_mixed_dtypes_linear2: The input datatype ", input.dtype(),
              " is not supported");
  TORCH_CHECK(weight.dtype() == at::kChar ||
              weight.dtype() == at::kByte,
              "_mixed_dtypes_linear2: The weight datatype ", weight.dtype(),
              " is not supported");
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.dtype() == input.dtype(),
                "_mixed_dtypes_linear2: Expected bias datatype ", input.dtype(),
                " but got", bias.dtype());
  }
  if (scale.numel() != 0) {
    TORCH_CHECK(scale.dtype() == input.dtype(),
                "_mixed_dtypes_linear2: Expected scale datatype ",
                input.dtype(), " but got", scale.dtype());
  }

  // Squash the batch dimensions of the input tensor with its
  // next-to-last dimensions.
  const auto input_sizes = input.sizes().vec();
  const auto input_2d = input.reshape({-1, input_sizes.back()});

  // Validate layouts of input tensors.
  TORCH_CHECK(input_2d.layout() == Layout::Strided,
              "_mixed_dtypes_linear2: Expected input argument to be strided, "
              "but got layout ", input_2d.layout());
  TORCH_CHECK(input_2d.dim() == 2,
              "_mixed_dtypes_linear2: Expected input argument to be 2D tensor, "
              "got ", input_2d.dim(), " dims");
  const auto input_strides = input_2d.strides();
  TORCH_CHECK(input_strides[0] > 1 && input_strides[1] == 1,
              "_mixed_dtypes_linear2: Invalid strides for input argument: row "
              "stride = ", input_strides[0], ", column stride = ",
              input_strides[1]);
  TORCH_CHECK(weight.layout() == Layout::Strided,
              "_mixed_dtypes_linear2: Expected input argument to be strided, "
              "but got layout ", weight.layout());
  TORCH_CHECK(weight.dim() == 2,
              "_mixed_dtypes_linear2: Expected weight argument to be 2D "
              " tensor, got ", weight.dim(), " dims");
  const auto weight_strides = weight.strides();
  TORCH_CHECK(weight_strides[0] > 1 && weight_strides[1] == 1,
              "_mixed_dtypes_linear2: Invalid strides for weight argument: row "
              "stride = ", weight_strides[0], ", column stride = ",
              weight_strides[1]);
  if (scale.numel() != 0) {
    TORCH_CHECK(scale.layout() == Layout::Strided,
              "_mixed_dtypes_linear2: Expected scale argument to be strided, "
              "but got layout ", scale.layout());
    TORCH_CHECK(scale.dim() == 1,
                "_mixed_dtypes_linear: Expected scale argument to be 1D ",
                "tensor, got ", scale.dim(), " dims");
    const auto scale_strides = scale.strides();
    TORCH_CHECK(scale_strides[0] == 1,
              "_mixed_dtypes_linear2: Invalid strides for scale argument: "
              "element stride = ", scale_strides[0]);
  }
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.layout() == Layout::Strided,
              "_mixed_dtypes_linear2: Expected bias argument to be strided, "
              "but got layout ", bias.layout());
    TORCH_CHECK(bias.dim() == 1,
                "_mixed_dtypes_linear: Expected bias argument to be 1D ",
                "tensor, got ", bias.dim(), " dims");
    const auto bias_strides = bias.strides();
    TORCH_CHECK(bias_strides[0] == 1,
              "_mixed_dtypes_linear2: Invalid strides for bias argument: "
              "element stride = ", bias_strides[0]);
  }

  // Validate sizes of input tensors.
  TORCH_CHECK(input_2d.size(1) == weight.size(1),
              "_mixed_dtypes_linear2: Expected input argument to have ",
              weight.size(1), " columns, but got ", input_2d.size(1));
  if (scale.numel() != 0) {
    TORCH_CHECK(scale.dim() == 1,
                "_mixed_dtypes_linear: Expected scale argument to have ",
                weight.size(0), " elements, got ", scale.numel(), " elements");
  }
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.dim() == 1,
                "_mixed_dtypes_linear: Expected bias argument to have ",
                weight.size(0), " elements, got ", bias.numel(), " elements");
  }

  Tensor output;
  auto scalar_type_quant = weight.scalar_type();
  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "_mixed_dtypes_linear2",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_linear2",
                AT_DISPATCH_CASE(
                    at::ScalarType::Char,
                    [&]() {
                      output =
                          mixed_dtypes_linear_cutlass_dispatch_scale_bias<
                              cutlass::half_t,
                              int8_t>(input_2d, weight, scale, bias,
                                      activation);
                      return;
                    })
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_linear_cutlass_dispatch_scale_bias<
                              cutlass::half_t,
                              uint8_t>(input_2d, weight, scale, bias,
                                       activation);
                      return;
                    }));
          })
      AT_DISPATCH_CASE(
          at::ScalarType::BFloat16,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_linear2",
                AT_DISPATCH_CASE(
                    at::ScalarType::Char,
                    [&]() {
                      output =
                          mixed_dtypes_linear_cutlass_dispatch_scale_bias<
                              cutlass::bfloat16_t,
                              int8_t>(input_2d, weight, scale, bias,
                                      activation);
                      return;
                    })
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_linear_cutlass_dispatch_scale_bias<
                              cutlass::bfloat16_t,
                              uint8_t>(input_2d, weight, scale, bias,
                                       activation);
                      return;
                    }));
          }));

  auto output_sizes = input_sizes;
  output_sizes.back() = weight.size(0);
  return output.reshape(output_sizes);
#endif
}

}  // namespace native
}  // namespace at
