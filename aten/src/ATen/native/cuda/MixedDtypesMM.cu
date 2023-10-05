#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/Dispatch.h>

#ifndef USE_ROCM
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm_universal.h>
#endif

#ifndef USE_ROCM
#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }
#endif

namespace at {
namespace native {

#ifndef USE_ROCM
template<typename ElementInputA, typename ElementInputB>
Tensor
mixed_dtypes_mm_cutlass(
    const Tensor& input, const Tensor& weight) {
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  const int length_m = input.size(0);
  const int length_k = input.size(1);
  const int length_n = weight.size(1);

  // Check for current CUTLASS limitations w.r.t. weight sizes.
  TORCH_CHECK(length_k % 16 == 0 && length_n % 16 == 0,
              "mixed_dtypes_mm_dispatch_dtype: Number of rows/columns of "
              "the weight matrix must be divisible by ", 16);

  using ElementOutput = ElementInputA;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;

  using SmArch = cutlass::arch::Sm80;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  constexpr auto ElementsPerAccessA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr auto ElementsPerAccessB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  constexpr auto ElementsPerAccessOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  constexpr auto NumStages = 4;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      ElementsPerAccessOutput,
      ElementAccumulator,
      ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
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
      NumStages,
      ElementsPerAccessA,
      ElementsPerAccessB,
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      cutlass::ComplexTransform::kNone,
      cutlass::ComplexTransform::kNone>;

  auto output = input.new_empty({length_m, length_n});

  const auto input_strides = input.strides();
  const auto weight_strides = weight.strides();

  const auto mode = cutlass::gemm::GemmUniversalMode::kGemm;
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  const auto batch_count = 1;
  const auto alpha = ElementComputeEpilogue(1);
  const auto beta = ElementComputeEpilogue(0);
  LayoutInputA input_layout(input_strides[0]);
  LayoutInputB weight_layout(weight_strides[1]);
  LayoutOutput output_layout(output.stride(0));
  auto input_device_ref =
      cutlass::TensorRef<ElementInputA, LayoutInputA>(
          (ElementInputA*)input.data_ptr(), input_layout);
  auto weight_device_ref =
      cutlass::TensorRef<ElementInputB, LayoutInputB>(
          (ElementInputB*)weight.data_ptr(), weight_layout);
  auto output_device_ref =
      cutlass::TensorRef<ElementOutput, LayoutOutput>(
          (ElementOutput*)output.data_ptr(), output_layout);

  typename Gemm::Arguments arguments{
    mode,
    problem_size,
    batch_count,
    {alpha, beta},
    input_device_ref.data(),
    weight_device_ref.data(),
    output_device_ref.data(),
    output_device_ref.data(),
    input_layout.capacity(problem_size.mk()),
    weight_layout.capacity(problem_size.kn()),
    output_layout.capacity(problem_size.mn()),
    output_layout.capacity(problem_size.mn()),
    input_layout.stride(),
    weight_layout.stride(),
    output_layout.stride(),
    output_layout.stride()
  };

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
#endif

Tensor
_mixed_dtypes_mm(const Tensor& input, const Tensor& weight) {
#ifndef USE_ROCM
  // For now, only CC 8.x devices are supported.
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm8x = dprops->major == 8;
  TORCH_CHECK(is_sm8x,
              "_mixed_dtypes_mm: Supported only on GPUs with compute "
              "capability 8.x");

  // Validate datatypes of input tensors.
  TORCH_CHECK(input.dtype() == at::kHalf,
              "_mixed_dtypes_mm: The input datatype ", input.dtype(),
              " is not supported");
  TORCH_CHECK(weight.dtype() == at::kChar ||
              weight.dtype() == at::kByte,
              "_mixed_dtypes_mm: The weight datatype ", weight.dtype(),
              " is not supported");

  // Squash the batch dimensions of the input tensor with its
  // next-to-last dimensions.
  const auto input_sizes = input.sizes().vec();
  const auto input_2d = input.reshape({-1, input_sizes.back()});

  // Validate layouts of input tensors.
  TORCH_CHECK(input_2d.layout() == Layout::Strided,
              "_mixed_dtypes_mm: Expected input argument to be strided, "
              "but got layout ", input_2d.layout());
  TORCH_CHECK(input_2d.dim() == 2,
              "_mixed_dtypes_mm: Expected input argument to be 2D tensor, "
              "got ", input_2d.dim(), " dims");
  const auto input_strides = input_2d.strides();
  TORCH_CHECK(input_strides[0] > 1 && input_strides[1] == 1,
              "_mixed_dtypes_mm: Invalid strides for input argument: row "
              "stride = ", input_strides[0], ", column stride = ",
              input_strides[1]);
  TORCH_CHECK(weight.layout() == Layout::Strided,
              "_mixed_dtypes_mm: Expected input argument to be strided, but "
              "got layout ", weight.layout());
  TORCH_CHECK(weight.dim() == 2,
              "_mixed_dtypes_mm: Expected weight argument to be 2D tensor, "
              "got ", weight.dim(), " dims");
  const auto weight_strides = weight.strides();
  TORCH_CHECK(weight_strides[0] == 1 && weight_strides[1] > 1,
              "_mixed_dtypes_mm: Invalid strides for weight argument: row "
              "stride = ", weight_strides[0], ", column stride = ",
              weight_strides[1]);

  // Validate sizes of input tensors.
  TORCH_CHECK(input_2d.size(1) == weight.size(0),
              "_mixed_dtypes_mm: Expected input argument to have ",
              weight.size(0), " columns, but got ", input_2d.size(1));

  Tensor output;
  auto scalar_type_quant = weight.scalar_type();
  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "_mixed_dtypes_mm",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_mm",
                AT_DISPATCH_CASE(
                    at::ScalarType::Char,
                    [&]() {
                      output =
                          mixed_dtypes_mm_cutlass<
                              cutlass::half_t,
                              int8_t>(input_2d, weight);
                      return;
                    })
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_mm_cutlass<
                              cutlass::half_t,
                              uint8_t>(input_2d, weight);
                      return;
                    }));
          }));

  auto output_sizes = input_sizes;
  output_sizes.back() = weight.size(1);
  return output.reshape(output_sizes);
#else
  AT_ERROR("_mixed_dtypes_mm: ROCm doesn't support CUTLASS");
  return Tensor{};
#endif
}

}  // namespace native
}  // namespace at
