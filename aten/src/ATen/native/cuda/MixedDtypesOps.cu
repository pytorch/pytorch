#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/Dispatch.h>

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>

#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }
#endif

namespace at {
namespace native {

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
template<
  typename ElementInputA,
  typename ElementInputB,
  bool use_tensor_b_scale,
  bool use_tensor_c>
void mdtgemm_cutlass(
    const Tensor& tensor_a, const Tensor& tensor_b,
    const Tensor& tensor_b_scale, const Tensor& tensor_c, Tensor& tensor_d) {
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  const int length_m = tensor_a.size(0);
  const int length_k = tensor_a.size(1);
  const int length_n = tensor_b.size(1);

  // Check for current CUTLASS limitations w.r.t. problem sizes.
  TORCH_CHECK(length_m % 16 == 0 && length_k % 16 == 0 && length_n % 16 == 0,
              "mdtgemm_cutlass: Number of rows/columns of the operands must be "
              "divisible by ", 16);

  using ElementC = ElementInputA;
  using ElementTensorBScale = ElementInputA;
  using ElementTensorC = ElementInputA;
  using ElementAccumulator = float;
  using ElementEpilogue = float;
  using ElementOutput = ElementInputA;

  constexpr int AlignmentInputA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentInputB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentTensorBScale = 128 / cutlass::sizeof_bits<ElementTensorBScale>::value;
  constexpr int AlignmentTensorC = 128 / cutlass::sizeof_bits<ElementTensorC>::value;
  constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using SmArch = cutlass::arch::Sm80;
  using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape           = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 16>;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  constexpr auto NumStages = 3;

  constexpr auto NumEVTEpilogueStages = 1;

  using TensorBScaleTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementTensorBScale,
      AlignmentTensorBScale,
      NumEVTEpilogueStages
  >;
  using TensorCTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementTensorC,
      AlignmentTensorC,
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

  using TensorBScaleScalar =
      cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementTensorBScale>;
  using TensorBScaleTensor =
      cutlass::epilogue::threadblock::VisitorAuxLoad<
          TensorBScaleTileThreadMap,
          ElementTensorBScale,
          cute::Stride<int64_t, cute::_1, int64_t>>;
  using TensorBScale = std::conditional_t<use_tensor_b_scale, TensorBScaleTensor, TensorBScaleScalar>;
  using TensorBScaleArguments = typename TensorBScale::Arguments;

  using ApplyTensorBScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyTensorBScale = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyTensorBScale,
      Accum,
      TensorBScale>;

  using TensorCScalar =
      cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementTensorC>;
  using TensorCTensor =
      cutlass::epilogue::threadblock::VisitorAuxLoad<
          TensorCTileThreadMap,
          ElementTensorC,
          cute::Stride<int64_t, cute::_1, int64_t>>;
  using TensorC = std::conditional_t<use_tensor_c, TensorCTensor, TensorCScalar>;
  using TensorCArguments = typename TensorC::Arguments;

  using ApplyTensorC = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementEpilogue, ElementEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using EVTApplyTensorC = cutlass::epilogue::threadblock::Sm80EVT<
      ApplyTensorC,
      EVTApplyTensorBScale,
      TensorC>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
      Output,
      EVTApplyTensorC>;

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

  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  constexpr auto SplitKFactor = 1;

  TensorBScaleArguments tensor_b_scale_arguments{
    [&]() -> TensorBScaleArguments {
      if constexpr (use_tensor_b_scale) {
        return {(ElementTensorBScale*)tensor_b_scale.data_ptr(),
                ElementTensorBScale(1),
                {cute::_0{}, cute::_1{}, problem_size.n()}};
      } else {
        return {ElementTensorBScale(1)};
      }
    }()
  };
  TensorCArguments tensor_c_arguments{
    [&]() -> TensorCArguments {
      if constexpr (use_tensor_c) {
        return {(ElementTensorC*)tensor_c.data_ptr(),
                ElementTensorC(0),
                {cute::_0{}, cute::_1{}, problem_size.n()}};
      } else {
        return {ElementTensorC(0)};
      }
    }()
  };
  typename Output::Arguments output_arguments{
    (ElementOutput*)tensor_d.data_ptr(),
    {problem_size.n(), cute::_1{}, problem_size.mn().product()}
  };
  typename EVTOutput::Arguments callback_arguments{
    {
      {
        {},                        // Accum
        tensor_b_scale_arguments,  // TensorBScale
        {}                         // ApplyTensorBScale
      },                           // EVTApplyTensorBScale
      tensor_c_arguments,          // TensorC
      {}                           // ApplyTensorC
    },                             // EVTApplyTensorC
    output_arguments,              // Output
  };                               // EVTOutput
  constexpr auto AvailSms = -1;
  typename Gemm::Arguments arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    SplitKFactor,
    callback_arguments,                       // arguments of EVT callbacks
    (ElementInputA*)tensor_a.data_ptr(),
    (ElementInputB*)tensor_b.data_ptr(),
    nullptr,                                  // ptr C (unused)
    nullptr,                                  // ptr D (unused)
    problem_size.mk().product(),              // batch stride A
    problem_size.nk().product(),              // batch stride B
    0,                                        // batch stride C (unused)
    0,                                        // batch stride D (unused)
    tensor_a.strides()[0],                    // stride A
    tensor_b.strides()[1],                    // stride B
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
  auto workspace = tensor_a.new_empty({(int64_t)workspace_size},
                                      at::TensorOptions().dtype(at::kByte));

  // Initialize CUTLASS mixed datatypes GEMM object.
  status = gemm_op.initialize(arguments, workspace.data_ptr(),
                              at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  // Perform mixed datatypes GEMM operation.
  status = gemm_op.run(at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename ElementInputA, typename ElementInputB>
void mdtgemm_cutlass_dispatch_tensor_b_scale_tensor_c(
    const Tensor& tensor_a, const Tensor& tensor_b,
    const Tensor& tensor_b_scale, const Tensor& tensor_c, Tensor& tensor_d) {
    if (tensor_b_scale.numel() > 0) {
        if (tensor_c.numel() > 0) {
            mdtgemm_cutlass<ElementInputA, ElementInputB, true, true>(
                  tensor_a, tensor_b, tensor_b_scale, tensor_c, tensor_d);
        }
        else {
            mdtgemm_cutlass<ElementInputA, ElementInputB, true, false>(
                  tensor_a, tensor_b, tensor_b_scale, tensor_c, tensor_d);
        }
    }
    else {
        if (tensor_c.numel() > 0) {
            mdtgemm_cutlass<ElementInputA, ElementInputB, false, true>(
                  tensor_a, tensor_b, tensor_b_scale, tensor_c, tensor_d);
        }
        else {
            mdtgemm_cutlass<ElementInputA, ElementInputB, false, false>(
                  tensor_a, tensor_b, tensor_b_scale, tensor_c, tensor_d);
        }
    }
}
#endif

Tensor
_mixed_dtypes_mad_op(const Tensor& mat1, const Tensor& mat2,
                     const Tensor& mat2_scale, const Tensor& input,
                     const Scalar& alpha, const Scalar& beta) {
#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
  AT_ERROR("_mixed_dtypes_mad_op: ROCm doesn't support CUTLASS");
  return Tensor{};
#else

  // For now, only CC 8.x devices are supported.
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm8x = dprops->major == 8;
  TORCH_CHECK(is_sm8x,
              "_mixed_dtypes_mad_op: Supported only on GPUs with compute "
              "capability 8.x");

  // Validate datatypes of input tensors.
  TORCH_CHECK(mat1.dtype() == at::kHalf ||
              mat1.dtype() == at::kBFloat16,
              "_mixed_dtypes_mad_op: The mat1 datatype ", mat1.dtype(),
              " is not supported");
  TORCH_CHECK(mat2.dtype() == at::kChar ||
              mat2.dtype() == at::kByte,
              "_mixed_dtypes_mad_op: The mat2 datatype ", mat2.dtype(),
              " is not supported");
  if (input.numel() != 0) {
    TORCH_CHECK(input.dtype() == mat1.dtype(),
                "_mixed_dtypes_mad_op: Expected input datatype ", mat1.dtype(),
                " but got", input.dtype());
  }
  if (mat2_scale.numel() != 0) {
    TORCH_CHECK(mat2_scale.dtype() == mat1.dtype(),
                "_mixed_dtypes_mad_op: Expected mat2_scale datatype ",
                mat1.dtype(), " but got", mat2_scale.dtype());
  }

  // Squash the batch dimensions of the mat1 tensor with its
  // next-to-last dimensions.
  const auto mat1_sizes = mat1.sizes().vec();
  const auto mat1_2d = mat1.reshape({-1, mat1_sizes.back()});

  // Validate layouts of input tensors.
  TORCH_CHECK(mat1_2d.layout() == Layout::Strided,
              "_mixed_dtypes_mad_op: Expected mat1 argument to be strided, but "
              "got layout ", mat1_2d.layout());
  TORCH_CHECK(mat1_2d.dim() == 2,
              "_mixed_dtypes_mad_op: Expected mat1 argument to be 2D tensor, "
              "got ", mat1_2d.dim(), " dims");
  const auto mat1_strides = mat1_2d.strides();
  TORCH_CHECK(mat1_strides[0] >= 1 && mat1_strides[1] == 1,
              "_mixed_dtypes_mad_op: Invalid strides for mat1 argument: row "
              "stride = ", mat1_strides[0], ", column stride = ",
              mat1_strides[1]);
  TORCH_CHECK(mat2.layout() == Layout::Strided,
              "_mixed_dtypes_mad_op: Expected mat1 argument to be strided, but "
              "got layout ", mat2.layout());
  TORCH_CHECK(mat2.dim() == 2,
              "_mixed_dtypes_mad_op: Expected mat2 argument to be 2D tensor, "
              " got ", mat2.dim(), " dims");
  const auto mat2_strides = mat2.strides();
  TORCH_CHECK(mat2_strides[0] == 1 && mat2_strides[1] >= 1,
              "_mixed_dtypes_mad_op: Invalid strides for mat2 argument: row "
              "stride = ", mat2_strides[0], ", column stride = ",
              mat2_strides[1]);
  if (mat2_scale.numel() != 0) {
    TORCH_CHECK(mat2_scale.layout() == Layout::Strided,
              "_mixed_dtypes_mad_op: Expected mat2_scale argument to be "
              "strided, but got layout ", mat2_scale.layout());
    TORCH_CHECK(mat2_scale.dim() == 1,
                "_mixed_dtypes_mad_op: Expected mat2_scale argument to be 1D ",
                "tensor, got ", mat2_scale.dim(), " dims");
    const auto mat2_scale_strides = mat2_scale.strides();
    TORCH_CHECK(mat2_scale_strides[0] == 1,
              "_mixed_dtypes_mad_op: Invalid strides for mat2_scale argument: "
              "element stride = ", mat2_scale_strides[0]);
  }
  if (input.numel() != 0) {
    TORCH_CHECK(input.layout() == Layout::Strided,
              "_mixed_dtypes_mad_op: Expected input argument to be strided, "
              "but got layout ", input.layout());
    TORCH_CHECK(input.dim() == 1,
                "_mixed_dtypes_mad_op: Expected input argument to be 1D "
                "tensor, got ", input.dim(), " dims");
    const auto input_strides = input.strides();
    TORCH_CHECK(input_strides[0] == 1,
              "_mixed_dtypes_mad_op: Invalid strides for input argument: "
              "element stride = ", input_strides[0]);
  }

  // Validate sizes of input tensors.
  TORCH_CHECK(mat1_2d.size(1) == mat2.size(0),
              "_mixed_dtypes_mad_op: Expected mat1 argument to have ",
              mat2.size(0), " columns, but got ", mat1_2d.size(1));
  if (mat2_scale.numel() != 0) {
    TORCH_CHECK(mat2_scale.numel() == mat2.size(1),
                "_mixed_dtypes_mad_op: Expected mat2_scale argument to have ",
                mat2.size(1), " elements, got ", mat2_scale.numel(),
                " elements");
  }
  if (input.numel() != 0) {
    TORCH_CHECK(input.numel() == mat2.size(1),
                "_mixed_dtypes_mad_op: Expected input argument to have ",
                mat2.size(1), " elements, got ", input.numel(), " elements");
  }

  // Introduce alias names for arguments, according to the CUTLASS
  // naming conventions.
  const auto& tensor_a = mat1_2d;
  const auto& tensor_b = mat2;
  const auto& tensor_b_scale = mat2_scale;
  const auto& tensor_c = input;

  // Create output tensor.
  Tensor tensor_d = tensor_a.new_empty({tensor_a.size(0), tensor_b.size(1)});

  auto scalar_type_quant = mat2.scalar_type();
  AT_DISPATCH_SWITCH(
      mat1.scalar_type(),
      "_mixed_dtypes_mad_op",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_mad_op",
                AT_DISPATCH_CASE(
                    at::ScalarType::Char,
                    [&]() {
                      mdtgemm_cutlass_dispatch_tensor_b_scale_tensor_c<
                          cutlass::half_t,
                          int8_t>(tensor_a, tensor_b, tensor_b_scale, tensor_c,
                                  tensor_d);
                    })
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      mdtgemm_cutlass_dispatch_tensor_b_scale_tensor_c<
                          cutlass::half_t,
                          uint8_t>(tensor_a, tensor_b, tensor_b_scale,
                                   tensor_c, tensor_d);
                    }));
          })
      AT_DISPATCH_CASE(
          at::ScalarType::BFloat16,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_mad_op",
                AT_DISPATCH_CASE(
                    at::ScalarType::Char,
                    [&]() {
                      mdtgemm_cutlass_dispatch_tensor_b_scale_tensor_c<
                          cutlass::bfloat16_t,
                          int8_t>(tensor_a, tensor_b, tensor_b_scale, tensor_c,
                                  tensor_d);
                    })
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      mdtgemm_cutlass_dispatch_tensor_b_scale_tensor_c<
                          cutlass::bfloat16_t,
                          uint8_t>(tensor_a, tensor_b, tensor_b_scale,
                                   tensor_c, tensor_d);
                    }));
          }));

  auto tensor_d_sizes = mat1_sizes;
  tensor_d_sizes.back() = mat2.size(1);
  return tensor_d.reshape(tensor_d_sizes);
#endif
}

// Implementation of aten._mixed_dtypes_mm operator.
Tensor _mixed_dtypes_mm(
      const Tensor& mat1, const Tensor& mat2, const Tensor& mat2_scale) {
    return _mixed_dtypes_mad_op(mat1, mat2, mat2_scale, Tensor{}, 1, 0);
}

// Implementation of aten._mixed_dtypes_addmm operator.
Tensor _mixed_dtypes_addmm(
      const Tensor& input, const Tensor& mat1, const Tensor& mat2,
      const Tensor& mat2_scale, const Scalar& alpha, const Scalar& beta) {
    return _mixed_dtypes_mad_op(mat1, mat2, mat2_scale, input, alpha, beta);
}

}  // namespace native
}  // namespace at
