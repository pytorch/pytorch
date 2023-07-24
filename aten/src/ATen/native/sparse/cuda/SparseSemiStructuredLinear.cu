#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/Dispatch.h>

#ifndef USE_ROCM
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <ATen/native/sparse/cuda/cutlass/gemm_sparse_row_broadcast.h>
#endif

#include <type_traits>
#include <tuple>

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
// Wrapper function for CUTLASS sparse GEMM implementation, used
// solely to simplify dispatching from
// _sparse_semi_structured_linear() function below.
template <
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
    const Tensor& tensor_c, const Tensor& meta) {
    // Fix CUTLASS sparse GEMM template arguments that are not
    // provided as template argument of this function, and create an
    // alias for particular instantiation of this template.
    using LayoutOutput = cutlass::layout::RowMajor; // Result of the operation will be provided in row-major format.
    using MMAOp = cutlass::arch::OpClassTensorOp; // Tensor cores are to be used for maximum performance.
    using SmArch = cutlass::arch::Sm80; // Only CC 8.x devices are suported at the moment.
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // This choice provides good performance across wide range of operand sizes.
    constexpr int NumStages = 3; // This choice provides good performance across wide range of operand sizes.
    using Gemm = cutlass::gemm::device::SparseGemmRowBroadcast<
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
    using LayoutInputE = cutlass::layout::RowMajor;
    using ReorderedLayoutInputE = typename Gemm::LayoutE;
    static_assert(
        std::is_same<ReorderedLayoutInputE,
                     cutlass::layout::ColumnMajorInterleaved<2>>::value,
        "Matrix layout used by CUTLASS for reordered metadata for sparse GEMM "
        "change, thus code doing conversions from/to dense matrix has to be "
        "updated.");

    constexpr auto kSparse = Gemm::kSparse;
    constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;

    // Operand sizes.
    const int length_m = tensor_a.size(0);
    const int length_k = tensor_b.size(0);
    const int length_n = tensor_b.size(1);
    const auto meta_ncols = length_k / kSparse / kElementsPerElementE;

    // Check for current CUTLASS limitations w.r.t. input sizes.
    constexpr auto input_a_16bit = sizeof(ElementInputA) == 2;
    TORCH_CHECK(length_m % 32 == 0,
        "two_four_sgemm_cutlass: Number of rows of sparse matrix must be "
        "divisible by 32");
    TORCH_CHECK(length_k % (input_a_16bit ? 64 : 128) == 0,
        "two_four_sgemm_cutlass: Number of rows of dense matrix must be "
        "divisible by ", (input_a_16bit ? 64 : 128));
    TORCH_CHECK(length_n % (input_a_16bit ? 8 : 16) == 0,
        "two_four_sgemm_cutlass: Number of columns of dense matrix must be "
        "divisible by ", (input_a_16bit ? 8 : 16));

    // Determine PyTorch datatype for the metadata matrix.
    auto meta_dtype = at::kChar;
    switch (sizeof(ElementInputE)) {
    case 1:
        break;
    case 2:
        meta_dtype = at::kShort;
        break;
    case 4:
        meta_dtype = at::kInt;
        break;
    default:
        AT_ERROR("two_four_sgemm_cutlass: invalid size of meta tensor datatype "
                 "encountered");
    }
    TORCH_CHECK(meta.dtype() == meta_dtype,
                "two_four_sgemm_cutlass: Expected meta datatype ", meta_dtype,
                ", but got ", meta.dtype());

    // Determine PyTorch datatype for the output matrix.
    auto tensor_d_dtype = at::kChar;
    if constexpr (std::is_same_v<ElementOutput, int32_t>) {
        tensor_d_dtype = at::kInt;
    }
    else if constexpr (std::is_same_v<ElementOutput, cutlass::half_t>) {
        tensor_d_dtype = at::kHalf;
    } else if constexpr (std::is_same_v<ElementOutput, cutlass::bfloat16_t>) {
        tensor_d_dtype = at::kBFloat16;
    }
    else {
        AT_ERROR("two_four_sgemm_cutlass: invalid datatype for sparse GEMM ",
                 " output encountered");
    }
    if (tensor_c.numel() != 0) {
        TORCH_CHECK(tensor_c.dtype() == tensor_d_dtype,
                    "two_four_sgemm_cutlass: Expected spars GTEMM bias "
                    "datatype ", tensor_d_dtype, ", but got ",
                    tensor_c.dtype());
    }

    // Create output matrix.
    Tensor tensor_d;
    if (tensor_c.numel() != 0) {
        tensor_d = tensor_c.new_empty({length_m, length_n});
    } else {
        tensor_d =
            tensor_a.new_empty({length_m, length_n},
                               at::TensorOptions().dtype(tensor_d_dtype));
    }

    // Prepare arguments for CUTLASS sparse GEMM kernel.
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    LayoutInputA layout_a(tensor_a_stride);
    LayoutInputB layout_b(tensor_b_stride);
    LayoutOutput layout_c(tensor_c.numel() != 0 ? tensor_c.stride(0) : 0);
    LayoutOutput layout_d(tensor_d.stride(0));
    auto tensor_a_device_ref =
        cutlass::TensorRef<ElementInputA, LayoutInputA>(
            (ElementInputA*)tensor_a.data_ptr(), layout_a);
    auto tensor_b_device_ref =
        cutlass::TensorRef<ElementInputB, LayoutInputB>(
            (ElementInputB*)tensor_b.data_ptr(), layout_b);
    auto tensor_c_device_ref =
        cutlass::TensorRef<ElementOutput, LayoutOutput>(
            (ElementOutput*)(tensor_c.numel() != 0 ?
                             tensor_c.data_ptr() : tensor_d.data_ptr()),
            layout_c);
    auto tensor_d_device_ref =
        cutlass::TensorRef<ElementOutput, LayoutOutput>(
            (ElementOutput*)tensor_d.data_ptr(), layout_d);
    auto tensor_e_reordered_device_ref =
        cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>(
            (ElementInputE*)meta.data_ptr(),
            ReorderedLayoutInputE::packed({length_m, meta_ncols}));
    ElementComputeEpilogue alpha(1);
    ElementComputeEpilogue beta(tensor_c.numel() != 0 ? 1 : 0);
    constexpr int split_k_slices = 1;

    // Create a tuple of CUTLASS sparse GEMM kernel arguments.
    typename Gemm::Arguments arguments{
        problem_size,
        tensor_a_device_ref,
        tensor_b_device_ref,
        tensor_c_device_ref,
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
    auto workspace = tensor_a.new_empty({(int64_t)workspace_size},
                                        at::TensorOptions().dtype(at::kByte));

    // Initialize CUTLASS sparse GEMM object.
    status = gemm_op.initialize(arguments, workspace.data_ptr(),
                                at::cuda::getCurrentCUDAStream());
    CUTLASS_STATUS_CHECK(status);

    // Perform sparse GEMM operation.
    status = gemm_op.run(at::cuda::getCurrentCUDAStream());
    CUTLASS_STATUS_CHECK(status);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return tensor_d;
}

// Dispatch according to the input tensors layouts combination.
template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ElementComputeEpilogue,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOp,
    bool EnableRowMajorRowMajorLayouts,
    bool EnableRowMajorColumnMajorLayouts,
    bool EnableColumnMajorRowMajorLayouts,
    bool EnableColumnMajorColumnMajorLayouts>
Tensor two_four_sgemm_cutlass_dispatch_layouts(
    const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c,
    const Tensor& meta) {
    // Determine layouts (row-major or column-major) of input tensors.
    const auto strides_a = tensor_a.strides();
    auto tensor_a_row_major = strides_a[1] == 1;
    auto tensor_a_stride = tensor_a_row_major ? strides_a[0] : strides_a[1];
    const auto strides_b = tensor_b.strides();
    auto tensor_b_row_major = strides_b[1] == 1;
    auto tensor_b_stride = tensor_b_row_major ? strides_b[0] : strides_b[1];

    // Perform dispatching.
    if constexpr (EnableRowMajorRowMajorLayouts) {
        if (tensor_a_row_major && tensor_b_row_major) {
            return two_four_sgemm_cutlass<
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
                tensor_a,
                tensor_a_stride,
                tensor_b,
                tensor_b_stride,
                tensor_c,
                meta);
        }
    }
    if constexpr (EnableRowMajorColumnMajorLayouts) {
        if (tensor_a_row_major && !tensor_b_row_major) {
            return two_four_sgemm_cutlass<
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
                tensor_a,
                tensor_a_stride,
                tensor_b,
                tensor_b_stride,
                tensor_c,
                meta);
        }
    }
    if constexpr (EnableColumnMajorRowMajorLayouts) {
        if (!tensor_a_row_major && tensor_b_row_major) {
            return two_four_sgemm_cutlass<
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
                tensor_a,
                tensor_a_stride,
                tensor_b,
                tensor_b_stride,
                tensor_c,
                meta);
        }
    }
    if constexpr (EnableColumnMajorColumnMajorLayouts) {
        if (!tensor_a_row_major && !tensor_b_row_major) {
            return two_four_sgemm_cutlass<
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
                tensor_a,
                tensor_a_stride,
                tensor_b,
                tensor_b_stride,
                tensor_c,
                meta);
        }
    }

    AT_ERROR("two_four_sgemm_cutlass_dispatch_layouts: Combination of ",
             tensor_a_row_major ? "row-major" : "column_major", "and ",
             tensor_b_row_major ? "row-major" : "column_major",
             " layouts for input tensors is not supported");
    return Tensor{};
}

// Dispatch according to the activation functions enabled.
template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ElementComputeEpilogue,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    bool EnableRowMajorRowMajorLayouts,
    bool EnableRowMajorColumnMajorLayouts,
    bool EnableColumnMajorRowMajorLayouts,
    bool EnableColumnMajorColumnMajorLayouts,
    bool EnableActivationNone,
    bool EnableActivationReLU,
    bool EnableActivationSiLU>
Tensor two_four_sgemm_cutlass_dispatch_layouts_activation(
    const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c,
    const Tensor& meta, const c10::string_view& activation) {
    // Perform dispatching.
    if constexpr (EnableActivationNone) {
        if (activation == "none") {
            using EpilogueOp =
                cutlass::epilogue::thread::LinearCombination<
                    ElementOutput,
                    128 / cutlass::sizeof_bits<ElementOutput>::value,
                    ElementAccumulator,
                    ElementComputeEpilogue>;
            return two_four_sgemm_cutlass_dispatch_layouts<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ElementComputeEpilogue,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                EpilogueOp,
                EnableRowMajorRowMajorLayouts,
                EnableRowMajorColumnMajorLayouts,
                EnableColumnMajorRowMajorLayouts,
                EnableColumnMajorColumnMajorLayouts>(
                tensor_a,
                tensor_b,
                tensor_c,
                meta);
        }
    }
    if constexpr (EnableActivationReLU) {
        if (activation == "relu") {
            using EpilogueOp =
                cutlass::epilogue::thread::LinearCombinationRelu<
                    ElementOutput,
                    128 / cutlass::sizeof_bits<ElementOutput>::value,
                    ElementAccumulator,
                    ElementComputeEpilogue>;
            return two_four_sgemm_cutlass_dispatch_layouts<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ElementComputeEpilogue,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                EpilogueOp,
                EnableRowMajorRowMajorLayouts,
                EnableRowMajorColumnMajorLayouts,
                EnableColumnMajorRowMajorLayouts,
                EnableColumnMajorColumnMajorLayouts>(
                tensor_a,
                tensor_b,
                tensor_c,
                meta);
        }
    }
    if constexpr (EnableActivationSiLU) {
        if (activation == "silu") {
            using EpilogueOp =
                cutlass::epilogue::thread::LinearCombinationSilu<
                    ElementOutput,
                    128 / cutlass::sizeof_bits<ElementOutput>::value,
                    ElementAccumulator,
                    ElementComputeEpilogue>;
            return two_four_sgemm_cutlass_dispatch_layouts<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ElementComputeEpilogue,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                EpilogueOp,
                EnableRowMajorRowMajorLayouts,
                EnableRowMajorColumnMajorLayouts,
                EnableColumnMajorRowMajorLayouts,
                EnableColumnMajorColumnMajorLayouts>(
                tensor_a,
                tensor_b,
                tensor_c,
                meta);
        }
    }

    AT_ERROR("two_four_sgemm_cutlass_dispatch_layouts: Activation \"",
             activation, "\" is not supported for given input tensors");
    return Tensor{};
}
#endif

// Perform linear transformation, but using corresponding CUTLASS
// sparse GEMM kernel, to given arguments:
//     output = input * weight.T + bias
// The "input" tensor is a dense tensor, while the "weight" tensor is
// a matrix with 2:4 sparsity pattern.  The "bias" tensor is optional;
// if provided, it should be a vector, with the number of elements
// equal to the number of rows of "weight" matrix.  It is assumed
// that.  It is assummed that "input", after squashing eventual batch
// dimensions with the next-to-last dimension of this tensor, and
// "weight" tensors are supplied either in row-major or column-major
// layouts (different layouts between these two tensors are OK, but
// not all combinations of formats are supported for some datatypes of
// these matrices).  The "meta" argument contains metadata matrix. The
// function returns the output tensor.
//
// There exists numerous limitations of CUTLASS sparse GEMM kernel,
// with regards to sizes and alignments of input tensors, their
// layouts and datatypes, and so on; this is the reason for large
// number of checks throughout the code.
Tensor _sparse_semi_structured_linear(
      const Tensor& input, const Tensor& weight,
      const Tensor& meta, const c10::optional<Tensor>& bias_opt,
      const c10::optional<c10::string_view> activation_opt) {
#ifndef USE_ROCM
    // No need to check that all tensors are on CUDA device, as this
    // is provided by dispatch.

    // Introduce alias names for arguments, according to the CUTLASS
    // naming conventions.  Also, squash the batch dimensions of the
    // input tensor with its next-to-last dimensions.
    const auto input_sizes = input.sizes().vec();
    const auto tensor_a = weight;
    const auto tensor_b =
        input.reshape({-1, input_sizes.back()}).transpose(-1, -2);
    const auto tensor_c = bias_opt.has_value() ? *bias_opt : Tensor{};

    const auto activation =
        activation_opt.has_value() ? *activation_opt : "none";

    // For now, only CC 8.x devices are supported.
    const auto dprops = at::cuda::getCurrentDeviceProperties();
    const auto is_sm8x = dprops->major == 8;
    TORCH_CHECK(is_sm8x,
                "_sparse_semi_structured_linear: Supported only on GPUs with "
                "compute capability 8.x");

    // Validate datatypes of input tensors.
    TORCH_CHECK(tensor_a.dtype() == at::kChar ||
                tensor_a.dtype() == at::kHalf ||
                tensor_a.dtype() == at::kBFloat16,
                "_sparse_semi_structured_linear: The weight datatype ",
                tensor_a.dtype(), " is not supported");
    TORCH_CHECK(tensor_b.dtype() == tensor_a.dtype(),
                "_sparse_semi_structured_linear: Expected input datatype ",
                tensor_a.dtype(), ", but got ", tensor_b.dtype());

    // Validate layouts of input tensors.
    TORCH_CHECK(tensor_a.layout() == Layout::Strided,
                "_sparse_semi_structured_linear: Expected weight argument "
                "to be strided, but got layout ", tensor_a.layout());
    TORCH_CHECK(tensor_a.dim() == 2,
                "_sparse_semi_structured_linear: Expected weight argument "
                "to be 2D tensor, got ", tensor_a.dim(), " dims");
    const auto strides_a = tensor_a.strides();
    TORCH_CHECK((strides_a[0] == 1 || strides_a[1] == 1) &&
                strides_a[0] != strides_a[1],
                "_sparse_semi_structured_linear: Invalid strides for weight "
                "argument: row stride = ", strides_a[0], ", column stride = ",
                strides_a[1]);
    TORCH_CHECK(tensor_b.layout() == Layout::Strided,
                "_sparse_semi_structured_linear: Expected input argument "
                "to be strided, but got layout ", tensor_b.layout());
    TORCH_CHECK(tensor_b.dim() == 2,
                "_sparse_semi_structured_linear: Expected input argument "
                "to be 2D tensor, got ", tensor_b.dim(), " dims");
    const auto strides_b = tensor_b.strides();
    TORCH_CHECK((strides_b[0] == 1 || strides_b[1] == 1) &&
                strides_b[0] != strides_b[1],
                "_sparse_semi_structured_linear: Invalid strides for input "
                "argument: row stride = ", strides_b[0], ", column stride = ",
                strides_b[1]);
    if (tensor_c.numel() != 0) {
        TORCH_CHECK(tensor_c.layout() == Layout::Strided,
                    "_sparse_semi_structured_linear: Expected bias argument "
                    "to be strided, but got layout ", tensor_c.layout());
        TORCH_CHECK(tensor_c.dim() == 1,
                    "_sparse_semi_structured_linear: Expected bias argument "
                    "to be 1D tensor, got ", tensor_c.dim(), " dims");
    }

    // Validate sizes of input tensors.
    TORCH_CHECK(tensor_a.size(1) == tensor_b.size(0) / 2,
                "_sparse_semi_structured_linear: Expected weight argument "
                "to have ", tensor_b.size(0) / 2, " columns, but got ",
                tensor_a.size(1));
    if (tensor_c.numel() != 0) {
        TORCH_CHECK(tensor_c.size(0) == tensor_a.size(0),
                    "_sparse_semi_structured_linear: Expected bias argument "
                    "to have ", tensor_a.size(0), " elements, but got ",
                    tensor_c.size(0));
    }

    // Call wrapper function for CUTLASS sparse GEMM, dispatching on
    // the input datatype, and then on input tensors layouts.
    // According to the input tensors datatypes and layouts,
    // correspnding template arguments are supplied for instantiating
    // the wrapper function.  The tile sizes template arguments are
    // selected according to the CUTLASS profiler results, for number
    // of runs.
    Tensor output;
    AT_DISPATCH_SWITCH(
        tensor_a.scalar_type(),
        "_sparse_semi_structured_linear",
        AT_DISPATCH_CASE(
            at::ScalarType::Char,
            [&]() {
                using ElementInputA = int8_t;
                using ElementInputB = int8_t;
                using ElementOutput = int32_t;
                using ElementAccumulator = int32_t;
                using ElementComputeEpilogue = int32_t;
                using ThreadblockShape =
                    cutlass::gemm::GemmShape<128, 128, 128>;
                using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
                const auto EnableRowMajorRowMajorLayouts = false;
                const auto EnableRowMajorColumnMajorLayouts = true;
                const auto EnableColumnMajorRowMajorLayouts = false;
                const auto EnableColumnMajorColumnMajorLayouts = false;
                const auto EnableActviationNone = true;
                const auto EnableActviationReLU = true;
                const auto EnableActviationSiLU = false;
                output = two_four_sgemm_cutlass_dispatch_layouts_activation<
                    ElementInputA,
                    ElementInputB,
                    ElementOutput,
                    ElementAccumulator,
                    ElementComputeEpilogue,
                    ThreadblockShape,
                    WarpShape,
                    InstructionShape,
                    EnableRowMajorRowMajorLayouts,
                    EnableRowMajorColumnMajorLayouts,
                    EnableColumnMajorRowMajorLayouts,
                    EnableColumnMajorColumnMajorLayouts,
                    EnableActviationNone,
                    EnableActviationReLU,
                    EnableActviationSiLU>(
                    tensor_a,
                    tensor_b,
                    tensor_c,
                    meta,
                    activation);
                return;
            })
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&]() {
                using ElementInputA = cutlass::half_t;
                using ElementInputB = cutlass::half_t;
                using ElementOutput = cutlass::half_t;
                using ElementAccumulator = float;
                using ElementComputeEpilogue = float;
                using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
                using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
                const auto EnableRowMajorRowMajorLayouts = true;
                const auto EnableRowMajorColumnMajorLayouts = true;
                const auto EnableColumnMajorRowMajorLayouts = true;
                const auto EnableColumnMajorColumnMajorLayouts = true;
                const auto EnableActviationNone = true;
                const auto EnableActviationReLU = true;
                const auto EnableActviationSiLU = true;
                output = two_four_sgemm_cutlass_dispatch_layouts_activation<
                    ElementInputA,
                    ElementInputB,
                    ElementOutput,
                    ElementAccumulator,
                    ElementComputeEpilogue,
                    ThreadblockShape,
                    WarpShape,
                    InstructionShape,
                    EnableRowMajorRowMajorLayouts,
                    EnableRowMajorColumnMajorLayouts,
                    EnableColumnMajorRowMajorLayouts,
                    EnableColumnMajorColumnMajorLayouts,
                    EnableActviationNone,
                    EnableActviationReLU,
                    EnableActviationSiLU>(
                    tensor_a,
                    tensor_b,
                    tensor_c,
                    meta,
                    activation);
                return;
            })
            AT_DISPATCH_CASE(
            at::ScalarType::BFloat16,
            [&]() {
                using ElementInputA = cutlass::bfloat16_t;
                using ElementInputB = cutlass::bfloat16_t;
                using ElementOutput = cutlass::bfloat16_t;
                using ElementAccumulator = float;
                using ElementComputeEpilogue = float;
                using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
                using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
                const auto EnableRowMajorRowMajorLayouts = true;
                const auto EnableRowMajorColumnMajorLayouts = true;
                const auto EnableColumnMajorRowMajorLayouts = true;
                const auto EnableColumnMajorColumnMajorLayouts = true;
                const auto EnableActviationNone = true;
                const auto EnableActviationReLU = true;
                const auto EnableActviationSiLU = true;
                output = two_four_sgemm_cutlass_dispatch_layouts_activation<
                    ElementInputA,
                    ElementInputB,
                    ElementOutput,
                    ElementAccumulator,
                    ElementComputeEpilogue,
                    ThreadblockShape,
                    WarpShape,
                    InstructionShape,
                    EnableRowMajorRowMajorLayouts,
                    EnableRowMajorColumnMajorLayouts,
                    EnableColumnMajorRowMajorLayouts,
                    EnableColumnMajorColumnMajorLayouts,
                    EnableActviationNone,
                    EnableActviationReLU,
                    EnableActviationSiLU>(
                    tensor_a,
                    tensor_b,
                    tensor_c,
                    meta,
                    activation);
                return;
            }));

    // Re-introduce batch dimensions into the output, and return.
    auto output_sizes = input_sizes;
    output_sizes.back() = weight.size(0);
    return output.transpose(-1, -2).reshape(output_sizes);
#else
    AT_ERROR("_sparse_semi_structured_linear: ROCm doesn't support CUTLASS");
    return Tensor{};
#endif
}

} // namespace native
} // namespace at

// Following is just for testing purposes.
namespace at {
namespace native {

#ifndef USE_ROCM
// Copied from tools/util/include/host_reorder.h, from CUTLASS source
// tree.  This is for simplicity - namely, this file is not under
// include/cutlass in this tree, as other CUTLASS include files
// needed, so it would require changing PyTorch CMake configuration;
// furthermore, including this file produces build errors in PyTorch
// at the moment.
template <typename Element, typename LayoutDest, typename LayoutSrc>
static void reorder_meta(cutlass::TensorRef<Element, LayoutDest> dest,
                         cutlass::TensorRef<Element, LayoutSrc> src,
                         const int problem_size_m, const int problem_size_k) {
  for (int m = 0; m < problem_size_m; m++) {
    for (int k = 0; k < problem_size_k; k++) {
      // First reorder the rows.
      int group = (sizeof(Element) == 2) ? 32 : 16;
      int interweave = (sizeof(Element) == 2) ? 4 : 2;

      int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
      int dest_col = k;

      // Next swizzle the 2x2 blocks from Z to N.
      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      dest.at({dest_row, dest_col}) = src.at({m, k});
    }
  }
}
#endif

std::tuple<Tensor, Tensor>
_to_sparse_semi_structured(const Tensor& dense) {
#ifndef USE_ROCM
  // Check dimensions of the dense matrix.
  TORCH_CHECK(dense.dim() == 2,
              "_to_sparse_semi_structured: Expected dense argument to be 2D "
              "tensor, got ", dense.dim(), " dims");

  // Determine PyTorch datatype for the metadata matrix.
  auto meta_dtype = at::kChar;
  auto dense_elems_per_meta_elem = 0;
  if (dense.dtype() == at::kChar) {
    meta_dtype = at::kInt;
    dense_elems_per_meta_elem = 32;
  } else if (dense.dtype() == at::kHalf || dense.dtype() == at::kBFloat16) {
    meta_dtype = at::kShort;
    dense_elems_per_meta_elem = 16;
  } else {
    AT_ERROR("_to_sparse_semi_structured: Invalid dense argument datatype ",
             dense.dtype(), "encountered");
  }

  const auto dense_nrows = dense.size(0);
  const auto dense_ncols = dense.size(1);

  if (dense_nrows % 32 != 0) {
    AT_ERROR("_to_sparse_semi_structured: Number of rows of dense matrix must "
             "be divisible by 32, but it is ", dense_nrows);
  }
  if (dense_ncols % dense_elems_per_meta_elem != 0) {
    AT_ERROR("_to_sparse_semi_structured: Number of columns of dense matrix "
             "must be divisible by ", dense_elems_per_meta_elem, ", but it is ",
             dense_ncols);
  }

  const auto dense_cpu = dense.to("cpu");

  const auto mask_cpu = dense_cpu != at::zeros({1}, dense_cpu.options());

  const auto sparse_cpu =
    dense_cpu.masked_select(mask_cpu).view({dense_nrows, dense_ncols / 2});

  const auto meta_nrows = dense_nrows;
  const auto meta_ncols = dense_ncols / dense_elems_per_meta_elem;

  auto meta_cpu = dense_cpu.new_empty({meta_nrows, meta_ncols},
                                      at::TensorOptions().dtype(meta_dtype));
  auto* mask_cpu_ptr = mask_cpu.data_ptr<bool>();
  for (auto i = 0; i < meta_nrows; ++i) {
    for (auto j = 0; j < meta_ncols; ++j) {
      uint64_t meta_val = 0;
      for (auto k = 0; k < dense_elems_per_meta_elem / 4; ++k, mask_cpu_ptr += 4) {
        const auto mask_elems = std::make_tuple(
          mask_cpu_ptr[0],
          mask_cpu_ptr[1],
          mask_cpu_ptr[2],
          mask_cpu_ptr[3]
        );
        auto meta_quadruple = 0;
        if (mask_elems == std::make_tuple(1, 1, 0, 0)) {
          meta_quadruple = 4; // 0100
        } else if (mask_elems == std::make_tuple(1, 0, 1, 0)) {
          meta_quadruple = 8; // 1000
        } else if (mask_elems == std::make_tuple(0, 1, 1, 0)) {
          meta_quadruple = 9; // 1001
        } else if (mask_elems == std::make_tuple(1, 0, 0, 1)) {
          meta_quadruple = 12; // 1100
        } else if (mask_elems == std::make_tuple(0, 1, 0, 1)) {
          meta_quadruple = 13; // 1101
        } else if (mask_elems == std::make_tuple(0, 0, 1, 1)) {
          meta_quadruple = 14; // 1110
        }
        else {
          AT_ERROR("_to_sparse_semi_structured: dense argument does not match "
                   "2:4 sparsity pattern");
        }
        meta_val = meta_val | (meta_quadruple << (4 * k));
      }
      const auto idx = i * meta_ncols + j;
      if (meta_dtype == at::kShort) {
        using MetaElement = int16_t;
        const auto meta_cpu_ptr = meta_cpu.data_ptr<MetaElement>();
        meta_cpu_ptr[idx] = (MetaElement)meta_val;
      } else if (meta_dtype == at::kInt) {
        using MetaElement = int32_t;
        const auto meta_cpu_ptr = meta_cpu.data_ptr<MetaElement>();
        meta_cpu_ptr[idx] = (MetaElement)meta_val;
      }
    }
  }

  auto meta_reordered_cpu = meta_cpu.new_empty({meta_nrows, meta_ncols});
  using MetaLayout = cutlass::layout::RowMajor;
  using MetaReorderedLayout = cutlass::layout::ColumnMajorInterleaved<2>;
  if (meta_dtype == at::kShort) {
    using MetaElement = int16_t;
    auto meta_cpu_ref =
      cutlass::TensorRef<MetaElement, MetaLayout>(
          meta_cpu.data_ptr<MetaElement>(),
          MetaLayout::packed({meta_nrows, meta_ncols}));
    auto meta_reordered_cpu_ref =
      cutlass::TensorRef<MetaElement, MetaReorderedLayout>(
          meta_reordered_cpu.data_ptr<MetaElement>(),
          MetaReorderedLayout::packed({meta_nrows, meta_ncols}));
    reorder_meta(meta_reordered_cpu_ref, meta_cpu_ref, meta_nrows, meta_ncols);
  } else if (meta_dtype == at::kInt) {
    using MetaElement = int32_t;
    auto meta_cpu_ref =
      cutlass::TensorRef<MetaElement, MetaLayout>(
          meta_cpu.data_ptr<MetaElement>(),
          MetaLayout::packed({meta_nrows, meta_ncols}));
    auto meta_reordered_cpu_ref =
      cutlass::TensorRef<MetaElement, MetaReorderedLayout>(
          meta_reordered_cpu.data_ptr<MetaElement>(),
          MetaReorderedLayout::packed({meta_nrows, meta_ncols}));
    reorder_meta(meta_reordered_cpu_ref, meta_cpu_ref, meta_nrows, meta_ncols);
  }

  return std::make_tuple(sparse_cpu.to(dense.device()),
                         meta_reordered_cpu.to(dense.device()));
#else
  AT_ERROR("_to_sparse_semi_structured: ROCm doesn't support CUTLASS");
  return std::make_tuple(Tensor{}, Tensor{});
#endif
}

}  // namespace native
}  // namespace at
