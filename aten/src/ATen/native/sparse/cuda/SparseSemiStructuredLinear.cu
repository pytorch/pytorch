#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/Dispatch.h>

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm_sparse_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#endif

#include <type_traits>
#include <tuple>

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }

namespace {
    enum class Activation{NONE, RELU, SILU};
}
#endif

namespace at::native {

#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
#else
// Wrapper function for CUTLASS sparse GEMM implementation, used
// solely to simplify dispatching from
// _sparse_semi_structured_linear() function below.
template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename Operator,
    typename LayoutInputA,
    typename LayoutInputB,
    bool use_bias,
    Activation activation>
Tensor two_four_sgemm(
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
    using SmArch = cutlass::arch::Sm80; // Only CC 8.x devices are supported at the moment.
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // This choice provides good performance across wide range of operand sizes.
    constexpr int NumStages = 3; // This choice provides good performance across wide range of operand sizes.
    constexpr int NumEVTEpilogueStages = 1;

    constexpr int AlignmentInputA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
    constexpr int AlignmentInputB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
    constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    using ElementComputeEpilogue = ElementAccumulator;
    constexpr int AlignmentComputeEpilogue = 128 / cutlass::sizeof_bits<ElementComputeEpilogue>::value;
    using ElementC = ElementOutput;
    using LayoutC = LayoutOutput;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using BiasTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        ElementC,
        AlignmentC,
        NumEVTEpilogueStages>;
    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        ElementOutput,
        AlignmentOutput,
        NumEVTEpilogueStages>;

    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

    using BiasScalar =
        cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementC>;
    using BiasTensor =
        cutlass::epilogue::threadblock::VisitorColBroadcast<
            BiasTileThreadMap,
            ElementC,
            cute::Stride<cute::_1, cute::_0, int64_t>>;
    using Bias = std::conditional_t<use_bias, BiasTensor, BiasScalar>;
    using BiasArguments = typename Bias::Arguments;

    using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus, ElementComputeEpilogue, ElementComputeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTApplyBias = cutlass::epilogue::threadblock::Sm80EVT<
        ApplyBias,
        Accum,
        Bias>;

    using ApplyActivationNone = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::epilogue::thread::Identity,
        ElementComputeEpilogue,
        ElementComputeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ApplyActivationReLu = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::epilogue::thread::ReLu,
        ElementComputeEpilogue,
        ElementComputeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ApplyActivationSiLu = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::epilogue::thread::SiLu,
        ElementComputeEpilogue,
        ElementComputeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;
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
        cute::Stride<int64_t, cute::_1, int64_t>>;

    using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<
        Output,
        EVTApplyActivation>;

    using Gemm = cutlass::gemm::device::SparseGemmWithVisitor<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EVTOutput,
        SwizzleThreadBlock,
        NumStages,
        AlignmentInputA,
        AlignmentInputB,
        Operator,
        NumEVTEpilogueStages>;

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

    // Determine PyTorch datatype for the metadata matrix.
    auto meta_dtype = at::kChar;
    switch (sizeof(ElementInputE)) {
    case 2:
        meta_dtype = at::kShort;
        break;
    case 4:
        meta_dtype = at::kInt;
        break;
    default:
        TORCH_CHECK(false, "two_four_sgemm: invalid size of meta tensor datatype "
                 "encountered");
    }
    TORCH_CHECK(meta.dtype() == meta_dtype,
                "two_four_sgemm: Expected meta datatype ", meta_dtype,
                ", but got ", meta.dtype());

    // Determine PyTorch datatype for the output matrix.
    auto tensor_d_dtype = at::kChar;
    if constexpr (std::is_same_v<ElementOutput, int8_t>) {
        tensor_d_dtype = at::kChar;
    } else if constexpr (std::is_same_v<ElementOutput, int32_t>) {
        tensor_d_dtype = at::kInt;
    } else if constexpr (std::is_same_v<ElementOutput, cutlass::half_t>) {
        tensor_d_dtype = at::kHalf;
    } else if constexpr (std::is_same_v<ElementOutput, cutlass::bfloat16_t>) {
        tensor_d_dtype = at::kBFloat16;
    } else if constexpr (std::is_same_v<ElementOutput, float>) {
        tensor_d_dtype = at::kFloat;
    } else {
        TORCH_CHECK(false, "two_four_sgemm: invalid datatype for sparse GEMM output ",
                 "encountered");
    }
    if constexpr (use_bias) {
        TORCH_CHECK(tensor_c.dtype() == tensor_d_dtype,
                    "two_four_sgemm: Expected sparse GEMM bias datatype ",
                    tensor_d_dtype, ", but got ", tensor_c.dtype());
    }

    // Create output matrix.
    Tensor tensor_d =
        tensor_a.new_empty({length_m, length_n},
                           at::TensorOptions().dtype(tensor_d_dtype));

    // Prepare arguments for CUTLASS sparse GEMM kernel.
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
    LayoutInputA layout_a(tensor_a_stride);
    LayoutInputB layout_b(tensor_b_stride);
    auto tensor_a_device_ref =
        cutlass::TensorRef<ElementInputA, LayoutInputA>(
            (ElementInputA*)tensor_a.data_ptr(), layout_a);
    auto tensor_b_device_ref =
        cutlass::TensorRef<ElementInputB, LayoutInputB>(
            (ElementInputB*)tensor_b.data_ptr(), layout_b);
    auto tensor_e_reordered_device_ref =
        cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>(
            (ElementInputE*)meta.data_ptr(),
            ReorderedLayoutInputE::packed({length_m, meta_ncols}));

    BiasArguments bias_arguments{
        [&]() -> BiasArguments {
            if constexpr (use_bias) {
                return {(ElementC*)tensor_c.data_ptr(),
                        ElementC(0),
                        {cute::_1{}, cute::_0{}, problem_size.m()}};
            } else {
                return {ElementC(0)};
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
                {},                 // Accum
                bias_arguments,     // Bias
                {}                  // ApplyBias
            },                      // EVTApplyBias
            {}                      // ApplyActivation
        },                          // EVTApplyActivation
        output_arguments,           // Output
    };                              // EVTOutput

    // Create a tuple of CUTLASS sparse GEMM kernel arguments.
    typename Gemm::Arguments arguments{
        problem_size,
        tensor_a_device_ref,
        tensor_b_device_ref,
        tensor_e_reordered_device_ref,
        callback_arguments};

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
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename Operator,
    bool EnableRowMajorRowMajorLayouts,
    bool EnableRowMajorColumnMajorLayouts,
    bool EnableColumnMajorRowMajorLayouts,
    bool EnableColumnMajorColumnMajorLayouts,
    bool use_bias,
    Activation activation>
Tensor two_four_sgemm_dispatch_layouts(
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
            return two_four_sgemm<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                Operator,
                cutlass::layout::RowMajor,
                cutlass::layout::RowMajor,
                use_bias,
                activation>(
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
            return two_four_sgemm<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                Operator,
                cutlass::layout::RowMajor,
                cutlass::layout::ColumnMajor,
                use_bias,
                activation>(
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
            return two_four_sgemm<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                Operator,
                cutlass::layout::ColumnMajor,
                cutlass::layout::RowMajor,
                use_bias,
                activation>(
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
            return two_four_sgemm<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                Operator,
                cutlass::layout::ColumnMajor,
                cutlass::layout::ColumnMajor,
                use_bias,
                activation>(
                tensor_a,
                tensor_a_stride,
                tensor_b,
                tensor_b_stride,
                tensor_c,
                meta);
        }
    }

    TORCH_CHECK(false, "two_four_sgemm_dispatch_layouts: Combination of ",
             tensor_a_row_major ? "row-major" : "column_major", " and ",
             tensor_b_row_major ? "row-major" : "column_major",
             " layouts for input tensors is not supported");
    return Tensor{};
}

// Dispatch according to the bias tensor being provided or not.
template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename Operator,
    bool EnableRowMajorRowMajorLayouts,
    bool EnableRowMajorColumnMajorLayouts,
    bool EnableColumnMajorRowMajorLayouts,
    bool EnableColumnMajorColumnMajorLayouts,
    Activation activation>
Tensor two_four_sgemm_dispatch_layouts_bias(
    const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c,
    const Tensor& meta) {
    if (tensor_c.numel() > 0) {
        return two_four_sgemm_dispatch_layouts<
            ElementInputA,
            ElementInputB,
            ElementOutput,
            ElementAccumulator,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            Operator,
            EnableRowMajorRowMajorLayouts,
            EnableRowMajorColumnMajorLayouts,
            EnableColumnMajorRowMajorLayouts,
            EnableColumnMajorColumnMajorLayouts,
            true,
            activation>(
            tensor_a,
            tensor_b,
            tensor_c,
            meta);
    } else {
        return two_four_sgemm_dispatch_layouts<
            ElementInputA,
            ElementInputB,
            ElementOutput,
            ElementAccumulator,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            Operator,
            EnableRowMajorRowMajorLayouts,
            EnableRowMajorColumnMajorLayouts,
            EnableColumnMajorRowMajorLayouts,
            EnableColumnMajorColumnMajorLayouts,
            false,
            activation>(
            tensor_a,
            tensor_b,
            tensor_c,
            meta);
    }
}

// Dispatch according to the activation functions enabled.
template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename Operator,
    bool EnableRowMajorRowMajorLayouts,
    bool EnableRowMajorColumnMajorLayouts,
    bool EnableColumnMajorRowMajorLayouts,
    bool EnableColumnMajorColumnMajorLayouts,
    bool EnableActivationNone,
    bool EnableActivationReLU,
    bool EnableActivationSiLU>
Tensor two_four_sgemm_dispatch_layouts_bias_activation(
    const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c,
    const Tensor& meta, const std::string_view& activation) {
    // Perform dispatching.
    if constexpr (EnableActivationNone) {
        if (activation == "none") {
            return two_four_sgemm_dispatch_layouts_bias<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                Operator,
                EnableRowMajorRowMajorLayouts,
                EnableRowMajorColumnMajorLayouts,
                EnableColumnMajorRowMajorLayouts,
                EnableColumnMajorColumnMajorLayouts,
                Activation::NONE>(
                tensor_a,
                tensor_b,
                tensor_c,
                meta);
        }
    }
    if constexpr (EnableActivationReLU) {
        if (activation == "relu") {
            return two_four_sgemm_dispatch_layouts_bias<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                Operator,
                EnableRowMajorRowMajorLayouts,
                EnableRowMajorColumnMajorLayouts,
                EnableColumnMajorRowMajorLayouts,
                EnableColumnMajorColumnMajorLayouts,
                Activation::RELU>(
                tensor_a,
                tensor_b,
                tensor_c,
                meta);
        }
    }
    if constexpr (EnableActivationSiLU) {
        if (activation == "silu") {
            return two_four_sgemm_dispatch_layouts_bias<
                ElementInputA,
                ElementInputB,
                ElementOutput,
                ElementAccumulator,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                Operator,
                EnableRowMajorRowMajorLayouts,
                EnableRowMajorColumnMajorLayouts,
                EnableColumnMajorRowMajorLayouts,
                EnableColumnMajorColumnMajorLayouts,
                Activation::SILU>(
                tensor_a,
                tensor_b,
                tensor_c,
                meta);
        }
    }

    TORCH_CHECK(false, "two_four_sgemm_dispatch_layouts: Activation \"", activation,
             "\" is not supported for given input tensors");
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
// that.  It is assumed that "input", after squashing eventual batch
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
      const Tensor& meta, const std::optional<Tensor>& bias_opt,
      const std::optional<std::string_view> activation_opt,
      const std::optional<c10::ScalarType> out_dtype_opt) {
    TORCH_WARN_ONCE("_sparse_semi_structured_linear is deprecated and will be "
                    "removed in a future PyTorch release.  Please use "
                    "_sparse_semi_structured_mm/_sparse_semi_structured_addmm "
                    "instead.");
#if defined(USE_ROCM) || defined(_MSC_VER) || (defined(CUDA_VERSION) && CUDA_VERSION < 11080)
    TORCH_CHECK(false, "_sparse_semi_structured_linear: CUTLASS not supported");
    return Tensor{};
#else
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

    TORCH_CHECK(!out_dtype_opt.has_value() ||
                (tensor_a.dtype() == at::ScalarType::Char &&
                 out_dtype_opt.value() == at::ScalarType::Int),
                "_sparse_semi_structured_linear: Setting out_dtype is only "
                "supported for int8 input and int32 output");

    // For now, only CC 8.x devices are supported.
    const auto dprops = at::cuda::getCurrentDeviceProperties();
    const auto is_sm8x = dprops->major == 8;
    TORCH_CHECK(is_sm8x,
                "_sparse_semi_structured_linear: Supported only on GPUs with "
                "compute capability 8.x");

    // Validate datatypes of input tensors.
    TORCH_CHECK(tensor_a.dtype() == at::kChar ||
                tensor_a.dtype() == at::kHalf ||
                tensor_a.dtype() == at::kBFloat16 ||
                tensor_a.dtype() == at::kFloat,
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
    // corresponding template arguments are supplied for instantiating
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
                using ElementAccumulator = int32_t;
                using ThreadblockShape =
                    cutlass::gemm::GemmShape<128, 128, 128>;
                using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
                using Operator = cutlass::arch::OpMultiplyAddSaturate;
                const auto EnableRowMajorRowMajorLayouts = false;
                const auto EnableRowMajorColumnMajorLayouts = true;
                const auto EnableColumnMajorRowMajorLayouts = false;
                const auto EnableColumnMajorColumnMajorLayouts = false;
                const auto EnableActivationNone = true;
                const auto EnableActivationReLU = true;
                const auto EnableActivationSiLU = false;
                if (out_dtype_opt.has_value()) {
                  using ElementOutput = int32_t;
                  output = two_four_sgemm_dispatch_layouts_bias_activation<
                      ElementInputA,
                      ElementInputB,
                      ElementOutput,
                      ElementAccumulator,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      Operator,
                      EnableRowMajorRowMajorLayouts,
                      EnableRowMajorColumnMajorLayouts,
                      EnableColumnMajorRowMajorLayouts,
                      EnableColumnMajorColumnMajorLayouts,
                      EnableActivationNone,
                      EnableActivationReLU,
                      EnableActivationSiLU>(
                      tensor_a,
                      tensor_b,
                      tensor_c,
                      meta,
                      activation);
                } else {
                  using ElementOutput = int8_t;
                  output = two_four_sgemm_dispatch_layouts_bias_activation<
                      ElementInputA,
                      ElementInputB,
                      ElementOutput,
                      ElementAccumulator,
                      ThreadblockShape,
                      WarpShape,
                      InstructionShape,
                      Operator,
                      EnableRowMajorRowMajorLayouts,
                      EnableRowMajorColumnMajorLayouts,
                      EnableColumnMajorRowMajorLayouts,
                      EnableColumnMajorColumnMajorLayouts,
                      EnableActivationNone,
                      EnableActivationReLU,
                      EnableActivationSiLU>(
                      tensor_a,
                      tensor_b,
                      tensor_c,
                      meta,
                      activation);
                }
                return;
            })
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&]() {
                using ElementInputA = cutlass::half_t;
                using ElementInputB = cutlass::half_t;
                using ElementOutput = cutlass::half_t;
                using ElementAccumulator = float;
                using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
                using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
                using Operator = cutlass::arch::OpMultiplyAdd;
                const auto EnableRowMajorRowMajorLayouts = true;
                const auto EnableRowMajorColumnMajorLayouts = true;
                const auto EnableColumnMajorRowMajorLayouts = true;
                const auto EnableColumnMajorColumnMajorLayouts = true;
                const auto EnableActivationNone = true;
                const auto EnableActivationReLU = true;
                const auto EnableActivationSiLU = true;
                output = two_four_sgemm_dispatch_layouts_bias_activation<
                    ElementInputA,
                    ElementInputB,
                    ElementOutput,
                    ElementAccumulator,
                    ThreadblockShape,
                    WarpShape,
                    InstructionShape,
                    Operator,
                    EnableRowMajorRowMajorLayouts,
                    EnableRowMajorColumnMajorLayouts,
                    EnableColumnMajorRowMajorLayouts,
                    EnableColumnMajorColumnMajorLayouts,
                    EnableActivationNone,
                    EnableActivationReLU,
                    EnableActivationSiLU>(
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
                using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
                using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
                using Operator = cutlass::arch::OpMultiplyAdd;
                const auto EnableRowMajorRowMajorLayouts = true;
                const auto EnableRowMajorColumnMajorLayouts = true;
                const auto EnableColumnMajorRowMajorLayouts = true;
                const auto EnableColumnMajorColumnMajorLayouts = true;
                const auto EnableActivationNone = true;
                const auto EnableActivationReLU = true;
                const auto EnableActivationSiLU = true;
                output = two_four_sgemm_dispatch_layouts_bias_activation<
                    ElementInputA,
                    ElementInputB,
                    ElementOutput,
                    ElementAccumulator,
                    ThreadblockShape,
                    WarpShape,
                    InstructionShape,
                    Operator,
                    EnableRowMajorRowMajorLayouts,
                    EnableRowMajorColumnMajorLayouts,
                    EnableColumnMajorRowMajorLayouts,
                    EnableColumnMajorColumnMajorLayouts,
                    EnableActivationNone,
                    EnableActivationReLU,
                    EnableActivationSiLU>(
                    tensor_a,
                    tensor_b,
                    tensor_c,
                    meta,
                    activation);
                return;
            })
            AT_DISPATCH_CASE(
            at::ScalarType::Float,
            [&]() {
                using ElementInputA = float;
                using ElementInputB = float;
                using ElementOutput = float;
                using ElementAccumulator = float;
                using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
                using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
                using Operator = cutlass::arch::OpMultiplyAdd;
                const auto EnableRowMajorRowMajorLayouts = true;
                const auto EnableRowMajorColumnMajorLayouts = true;
                const auto EnableColumnMajorRowMajorLayouts = true;
                const auto EnableColumnMajorColumnMajorLayouts = true;
                const auto EnableActivationNone = true;
                const auto EnableActivationReLU = true;
                const auto EnableActivationSiLU = true;
                output = two_four_sgemm_dispatch_layouts_bias_activation<
                    ElementInputA,
                    ElementInputB,
                    ElementOutput,
                    ElementAccumulator,
                    ThreadblockShape,
                    WarpShape,
                    InstructionShape,
                    Operator,
                    EnableRowMajorRowMajorLayouts,
                    EnableRowMajorColumnMajorLayouts,
                    EnableColumnMajorRowMajorLayouts,
                    EnableColumnMajorColumnMajorLayouts,
                    EnableActivationNone,
                    EnableActivationReLU,
                    EnableActivationSiLU>(
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
#endif
}

} // namespace at::native
