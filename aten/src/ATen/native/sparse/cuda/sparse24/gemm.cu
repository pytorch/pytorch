// BEGIN COPY-PASTE FROM PyTorch
// https://github.com/pytorch/pytorch/blob/b937510a3f254fe0223b9b29235e0eb6e6da912a/aten/src/ATen/native/sparse/cuda/StructuredSparseLinearCUTLASS.cu
// Some very small modifications, like we don't need to support uint8, and we
// always have the meta-reordered available
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/library.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>

#include <tuple>
#include <type_traits>

namespace {
#define CUTLASS_STATUS_CHECK(status)         \
  {                                          \
    TORCH_CHECK(                             \
        status == cutlass::Status::kSuccess, \
        "Got CUTLASS error: ",               \
        cutlassGetStatusString(status));     \
  }

using namespace at;

// Wrapper function for CUTLASS sparse GEMM implementation, used
// solely to simplify dispatching from _structured_sparse_linear()
// function below.
template <
    bool kIsMeta,
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ElementComputeEpilogue,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOp,
    typename LayoutInputA,
    typename LayoutInputB>
Tensor two_four_sgemm_cutlass(
    const Tensor& tensor_a,
    const at::IntArrayRef::value_type& tensor_a_stride,
    const Tensor& tensor_b,
    const at::IntArrayRef::value_type& tensor_b_stride,
    const Tensor& meta_reordered) {
  // Fix CUTLASS sparse GEMM template arguments that are not
  // provided as template argument of this function, and create an
  // alias for particular instantiation of this template.
  using LayoutOutput =
      cutlass::layout::RowMajor; // Result of the operation will be provided in
                                 // row-major format.
  using MMAOp = cutlass::arch::OpClassTensorOp; // Tensor cores are to be used
                                                // for maximum performance.
  using SmArch =
      cutlass::arch::Sm80; // Only CC 8.x devices are suported at the moment.
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<
          3>; // This choice provides good performance
              // across wide range of operand sizes.
  constexpr int NumStages = 4; // This choice provides good performance across
                               // wide range of operand sizes.
  using Gemm = cutlass::gemm::device::SparseGemm<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages>;

  // Datatype and layout of metadata matrix are inferred from sparse
  // GEMM template.
  using ElementInputE = typename Gemm::ElementE;
  using ReorderedLayoutInputE = typename Gemm::LayoutE;

  constexpr auto kSparse = Gemm::kSparse;
  constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;

  // Operand sizes.
  const int length_m = tensor_a.size(0);
  const int length_k = tensor_b.size(0);
  const int length_n = tensor_b.size(1);
  const auto meta_ncols = length_k / kSparse / kElementsPerElementE;

  // Check for current CUTLASS limitations w.r.t. input sizes.
  constexpr auto input_a_is_half =
      std::is_same<ElementInputA, cutlass::half_t>::value ||
      std::is_same<ElementInputA, cutlass::bfloat16_t>::value;
  TORCH_CHECK(
      length_m % 32 == 0,
      "torch._structured_sparse_linear: Number of rows of sparse matrix must "
      "be divisible by 32");
  TORCH_CHECK(
      length_k % (input_a_is_half ? 64 : 128) == 0,
      "torch._structured_sparse_linear: Number of rows of dense matrix must "
      "be divisible by ",
      (input_a_is_half ? 64 : 128));
  TORCH_CHECK(
      length_n % (input_a_is_half ? 8 : 16) == 0,
      "torch._structured_sparse_linear: Number of columns of dense matrix "
      "must be divisible by ",
      (input_a_is_half ? 8 : 16));

  // Determine PyTorch datatype for the output matrix.
  auto tensor_d_dtype = at::kChar;
  if (std::is_same<ElementOutput, int32_t>::value) {
    tensor_d_dtype = at::kInt;
  } else if (std::is_same<ElementOutput, cutlass::half_t>::value) {
    tensor_d_dtype = at::kHalf;
  } else if (std::is_same<ElementOutput, cutlass::bfloat16_t>::value) {
    tensor_d_dtype = at::kBFloat16;
  } else {
    AT_ERROR(
        "torch._structured_sparse_linear: invalid sparse GEMM output "
        "datatype encountered");
  }

  // Create output matrix.
  auto tensor_d = tensor_a.new_empty(
      {length_m, length_n}, at::TensorOptions().dtype(tensor_d_dtype));
  if (kIsMeta) {
    return tensor_d;
  }

  // Prepare arguments for CUTLASS sparse GEMM kernel.
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  LayoutInputA layout_a(tensor_a_stride);
  LayoutInputB layout_b(tensor_b_stride);
  LayoutOutput layout_d(tensor_d.stride(0));
  auto tensor_a_device_ref = cutlass::TensorRef<ElementInputA, LayoutInputA>(
      (ElementInputA*)tensor_a.data_ptr(), layout_a);
  auto tensor_b_device_ref = cutlass::TensorRef<ElementInputB, LayoutInputB>(
      (ElementInputB*)tensor_b.data_ptr(), layout_b);
  auto tensor_d_device_ref = cutlass::TensorRef<ElementOutput, LayoutOutput>(
      (ElementOutput*)tensor_d.data_ptr(), layout_d);
  auto tensor_e_reordered_device_ref =
      cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>(
          (ElementInputE*)meta_reordered.data_ptr(),
          ReorderedLayoutInputE::packed({length_m, meta_ncols}));
  ElementComputeEpilogue alpha(1);
  ElementComputeEpilogue beta(0);
  constexpr int split_k_slices = 1;

  // Create a tuple of CUTLASS sparse GEMM kernel arguments.
  typename Gemm::Arguments arguments{
      problem_size,
      tensor_a_device_ref,
      tensor_b_device_ref,
      tensor_d_device_ref,
      tensor_d_device_ref,
      tensor_e_reordered_device_ref,
      {alpha, beta},
      split_k_slices};

  cutlass::Status status;

  // Create CUTLASS sparse GEMM kernel object.
  Gemm gemm_op;

  // Verify that sparse GEMM operation with given arguments can be
  // performed by CUTLASS.
  status = gemm_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status);

  // Allocate workspace for CUTLASS sparse GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = tensor_a.new_empty(
      {(int64_t)workspace_size}, at::TensorOptions().dtype(at::kByte));

  // Initialize CUTLASS sparse GEMM object.
  status = gemm_op.initialize(
      arguments, workspace.data_ptr(), at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  // Perform sparse GEMM operation.
  status = gemm_op.run(at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return tensor_d;
}

template <bool kIsMeta>
Tensor _sparse24_gemm(
    const Tensor& tensor_a,
    const Tensor& tensor_b,
    const Tensor& mask_or_meta) {
  // No need to check that all tensors are on CUDA device, as this
  // is provided by dispatch.

  // For now, only CC 8.x devices are supported.
  if (!kIsMeta) {
    const auto dprops = at::cuda::getCurrentDeviceProperties();
    const auto is_sm8x = dprops->major == 8;
    TORCH_CHECK(
        is_sm8x,
        "torch._structured_sparse_linear: Supported only on GPUs with "
        "compute capability 8.x");
  }

  // Validate layouts of input tensors.
  TORCH_CHECK(
      tensor_a.layout() == Layout::Strided,
      "torch._structured_sparse_linear: Expected tensor_a argument "
      "to be strided, but got layout ",
      tensor_a.layout());
  TORCH_CHECK(
      tensor_a.dim() == 2,
      "torch._structured_sparse_linear: Expected tensor_a argument "
      "to be 2D tensor, got ",
      tensor_a.dim(),
      " dims");
  const auto strides_a = tensor_a.strides();
  TORCH_CHECK(
      (strides_a[0] == 1 || strides_a[1] == 1) && strides_a[0] != strides_a[1],
      "torch._structured_sparse_linear: Invalid strides for tensor_a "
      "argument: row stride = ",
      strides_a[0],
      ", column stride = ",
      strides_a[1]);
  TORCH_CHECK(
      tensor_b.layout() == Layout::Strided,
      "torch._structured_sparse_linear: Expected tensor_b argument "
      "to be strided, but got layout ",
      tensor_b.layout());
  TORCH_CHECK(
      tensor_b.dim() == 2,
      "torch._structured_sparse_linear: Expected tensor_b argument "
      "to be 2D tensor, got ",
      tensor_b.dim(),
      " dims");
  const auto strides_b = tensor_b.strides();
  TORCH_CHECK(
      (strides_b[0] == 1 || strides_b[1] == 1) && strides_b[0] != strides_b[1],
      "torch._structured_sparse_linear: Invalid strides for tensor_b "
      "argument: row stride = ",
      strides_b[0],
      ", column stride = ",
      strides_b[1]);

  // Determine layout (row-major or column-major) of input tensors.
  auto tensor_a_row_major = strides_a[1] == 1;
  auto tensor_a_stride = tensor_a_row_major ? strides_a[0] : strides_a[1];
  auto tensor_b_row_major = strides_b[1] == 1;
  auto tensor_b_stride = tensor_b_row_major ? strides_b[0] : strides_b[1];

  // Call wrapper function for CUTLASS sparse GEMM, dispatching on
  // the input datatype, and then on input tensors layouts.
  // According to the input tensors datatypes and layouts,
  // correspnding template arguments are supplied for instantiating
  // the wrapper function.  The tile sizes template arguments are
  // selected according to the CUTLASS profiler results, for number
  // of runs.
  Tensor result;
  auto runGemm = [&](auto dtype) {
    using ElementInputA = decltype(dtype);
    using ElementInputB = decltype(dtype);
    using ElementOutput = decltype(dtype);

    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;
    using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;
    if (tensor_a_row_major && tensor_b_row_major) {
      result = two_four_sgemm_cutlass<
          kIsMeta,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ElementComputeEpilogue,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueOp,
          cutlass::layout::RowMajor,
          cutlass::layout::RowMajor>(
          tensor_a, tensor_a_stride, tensor_b, tensor_b_stride, mask_or_meta);
    } else if (tensor_a_row_major && !tensor_b_row_major) {
      result = two_four_sgemm_cutlass<
          kIsMeta,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ElementComputeEpilogue,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueOp,
          cutlass::layout::RowMajor,
          cutlass::layout::ColumnMajor>(
          tensor_a, tensor_a_stride, tensor_b, tensor_b_stride, mask_or_meta);
    } else if (!tensor_a_row_major && tensor_b_row_major) {
      result = two_four_sgemm_cutlass<
          kIsMeta,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ElementComputeEpilogue,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueOp,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>(
          tensor_a, tensor_a_stride, tensor_b, tensor_b_stride, mask_or_meta);
    } else if (!tensor_a_row_major && !tensor_b_row_major) {
      result = two_four_sgemm_cutlass<
          kIsMeta,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ElementComputeEpilogue,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueOp,
          cutlass::layout::ColumnMajor,
          cutlass::layout::ColumnMajor>(
          tensor_a, tensor_a_stride, tensor_b, tensor_b_stride, mask_or_meta);
    }
  };
  if (tensor_a.scalar_type() == at::ScalarType::Half) {
    runGemm(cutlass::half_t());
  } else if (tensor_a.scalar_type() == at::ScalarType::BFloat16) {
    runGemm(cutlass::bfloat16_t());
  } else {
    TORCH_CHECK(false, "Unsupported Sparse24 GEMM")
  }
  return result;
}
// END PyTorch copy-pasted code
} // namespace

TORCH_LIBRARY_IMPL(sparse, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::_sparse24_gemm"),
      TORCH_FN(_sparse24_gemm<false>));
}

TORCH_LIBRARY_IMPL(sparse, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::_sparse24_gemm"),
      TORCH_FN(_sparse24_gemm<true>));
}
