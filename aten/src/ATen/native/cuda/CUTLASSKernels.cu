/*
The following source file implements a sparse linear operator using CUTLASS
*/

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/Dispatch.h>

#include <ATen/native/cuda/my_gemm_sparse.h>

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/util/device_memory.h>

#include <type_traits>
#include <tuple>

#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }

namespace at {
namespace native {

template<typename T>
__global__ void two_four_create_meta_kernel(
      const int length_m, const int length_k, const int stride_nz,
      const bool* nz, const int stride_meta, T* meta, int* error) {
    const auto k = blockDim.x * blockIdx.x + threadIdx.x;
    const auto m = blockDim.y * blockIdx.y + threadIdx.y;

    // FIXME: use shared memory, eventually also other optimizations!

    const auto in_range = m < length_m && k < length_k / 4;
    unsigned mask = __ballot_sync(0xffffffff, in_range);
    if (!in_range) {
        return;
    }

    T val = 0;
    const auto pos0 = nz[m * stride_nz + k * 4];
    const auto pos1 = nz[m * stride_nz + k * 4 + 1];
    const auto pos2 = nz[m * stride_nz + k * 4 + 2];
    const auto pos3 = nz[m * stride_nz + k * 4 + 3];
    const auto pos_tuple = std::make_tuple(pos0, pos1, pos2, pos3);

    // See
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-sparse-matrix-storage
    // There are only 6 valid configurations (4 choose 2) and for each
    // there is a special number.
    if (pos_tuple == std::make_tuple(1, 1, 0, 0)) {
        val = 4; // 0100
    } else if (pos_tuple == std::make_tuple(1, 0, 1, 0)) {
        val = 8; // 1000
    } else if (pos_tuple == std::make_tuple(0, 1, 1, 0)) {
        val = 9; // 1001
    } else if (pos_tuple == std::make_tuple(1, 0, 0, 1)) {
        val = 12; // 1100
    } else if (pos_tuple == std::make_tuple(0, 1, 0, 1)) {
        val = 13; // 1101
    } else if (pos_tuple == std::make_tuple(0, 0, 1, 1)) {
        val = 14; // 1110
    } else {
        atomicExch(error, 1);
    }

    auto tile_size = 2 * sizeof(T);
    for (auto i = 1; i < tile_size; i *= 2) {
        val |= __shfl_down_sync(mask, val, i) << (4 * i);
    }
    if (k % tile_size == 0) {
        meta[m * stride_meta + k / tile_size] = val;
    }
}

template <typename Element, typename LayoutSrc, typename LayoutDest>
__global__ void reorder_meta_kernel(
      const int length_m, const int length_k,
      const cutlass::TensorRef<Element, LayoutSrc> src,
      cutlass::TensorRef<Element, LayoutDest> dst) {
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    const int m = blockDim.y * blockIdx.y + threadIdx.y;

    if (m >= length_m || k >= length_k) {
        return;
    }

    // First reorder the rows.
    int group = (sizeof(Element) == 2) ? 32 : 16;
    int interweave = (sizeof(Element) == 2) ? 4 : 2;

    int dst_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
    int dst_col = k;

    // Next swizzle the 2x2 blocks from Z to N.
    if (((dst_row % 2) == 0) && ((dst_col % 2) == 1)) {
      ++dst_row;
      --dst_col;
    } else if (((dst_row % 2) == 1) && ((dst_col % 2) == 0)) {
      --dst_row;
      ++dst_col;
    }
    dst.at({dst_row, dst_col}) = src.at({m, k});
}

template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ElementComputeEpilogue,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename LayoutInputA,
    typename LayoutInputB>
std::tuple<Tensor, Tensor> cutlass_sgemm(
    const Tensor& tensor_a,
    const Tensor& tensor_b,
    const Tensor& tensor_c,
    const Tensor& mask_or_meta,
    const at::IntArrayRef::value_type& tensor_a_stride,
    const at::IntArrayRef::value_type& tensor_b_stride) {
    using LayoutOutput = cutlass::layout::RowMajor;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;
    using SwizzleThreadBlock =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    constexpr int NumStages = 3;
    using Gemm = cutlass::gemm::device::MySparseGemm<
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

    // Data type and layout of meta data matrix E are inferred from
    // template Gemm.
    using ElementInputE = typename Gemm::ElementE;
    using LayoutInputE = cutlass::layout::RowMajor;
    using ReorderedLayoutInputE = typename Gemm::LayoutE;

    constexpr auto kSparse = Gemm::kSparse;
    constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;

    const int length_m = tensor_a.size(0);
    const int length_k = tensor_b.size(0);
    const int length_n = tensor_b.size(1);
    const auto meta_ncols = length_k / kSparse / kElementsPerElementE;

    auto tensor_d = tensor_c.new_empty({length_m, length_n});

    // FIXME: check against the alignments, and not lengths, and take
    // into account datatypes other than half and int8.
    // Alignment related checks, CUTLASS will report error if these
    // not satisfied.
    constexpr auto input_a_is_half =
        std::is_same<ElementInputA, cutlass::half_t>();
    TORCH_CHECK(length_m % 32 == 0);
    TORCH_CHECK(length_k % (input_a_is_half ? 64 : 128) == 0);
    TORCH_CHECK(length_n % (input_a_is_half ? 8 : 16) == 0);

    Tensor meta_reordered;
    if (mask_or_meta.dtype() == at::kBool) {
        auto mask = mask_or_meta;

        TORCH_CHECK(mask.layout() == Layout::Strided, "torch._cutlass_linear: Expected mask argument to be strided, but got layout ", mask.layout());

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
            AT_ERROR("torch._cutlass_linear: invalid size of meta tensor datatype encountered");
        }

        auto meta = mask.new_empty({length_m, meta_ncols}, at::TensorOptions().dtype(meta_dtype));
        auto error = mask.new_zeros({1}, at::TensorOptions().dtype(at::kInt));

        // FIXME: verify that mask is in row major format, or pass
        // both strides to the kernel below.

        two_four_create_meta_kernel<<<
            dim3((length_k + 63) / 64, (length_m + 15) / 16),
            dim3(16, 16),
            0,
            at::cuda::getCurrentCUDAStream()>>> (
                length_m, length_k, mask.stride(0), (bool*)mask.data_ptr(),
                meta.stride(0), (ElementInputE*)meta.data_ptr(),
                (int*)error.data_ptr());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        TORCH_CHECK(error.item().equal(0), "Mask matrix is not 2:4 sparse");

        meta_reordered = meta.new_empty(meta.sizes());

        auto meta_device_ref = cutlass::TensorRef<ElementInputE, cutlass::layout::RowMajor>((ElementInputE*)meta.data_ptr(), LayoutInputE::packed({length_m, meta_ncols}));
        auto meta_reordered_device_ref = cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>((ElementInputE*)meta_reordered.data_ptr(), ReorderedLayoutInputE::packed({length_m, meta_ncols}));

        reorder_meta_kernel<<<
            dim3((meta_ncols + 15) / 16, (length_m + 15) / 16),
            dim3(16, 16),
            0,
            at::cuda::getCurrentCUDAStream()>>> (
                length_m, meta_ncols, meta_device_ref,
                meta_reordered_device_ref);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else {
        meta_reordered = mask_or_meta;
    }

    TORCH_CHECK(tensor_a.size(1) == length_k / kSparse);

    TORCH_CHECK(
        tensor_b.size(0) % kSparse == 0,
        "Expected tensor_b.size(0) of value ",
        tensor_b.size(0),
        " to be evenly divisible by ",
        kSparse,
        " but got.");
    TORCH_CHECK(
        tensor_a.size(1) * kSparse == tensor_b.size(0),
        "Expected tensor_a.size(1) of value ",
        tensor_a.size(1),
        " to match tensor_b.size(0) of value ",
        tensor_b.size(0),
        " to match after being multiplied by ",
        kSparse);

    TORCH_CHECK(
        tensor_a.device() == tensor_b.device(),
        "Check 0: Expected all Tensors to live on the GPU.");
    TORCH_CHECK(
        tensor_b.device() == tensor_c.device(),
        "Check 1: Expected all Tensors to live on the GPU.");
    TORCH_CHECK(
        tensor_c.device() == tensor_d.device(),
        "Check 2: Expected all Tensors to live on the GPU.");

    auto meta_reordered_device_ref = cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>((ElementInputE*)meta_reordered.data_ptr(), ReorderedLayoutInputE::packed({length_m, meta_ncols}));

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    LayoutInputA layout_a(tensor_a_stride);
    LayoutInputB layout_b(tensor_b_stride);
    LayoutOutput layout_c(tensor_c.stride(0));
    LayoutOutput layout_d(tensor_d.stride(0));
    auto tensor_a_device_ref = cutlass::TensorRef<ElementInputA, LayoutInputA>((ElementInputA*)tensor_a.data_ptr(), layout_a);
    auto tensor_b_device_ref = cutlass::TensorRef<ElementInputB, LayoutInputB>((ElementInputB*)tensor_b.data_ptr(), layout_b);
    auto tensor_c_device_ref = cutlass::TensorRef<ElementOutput, LayoutOutput>((ElementOutput*)tensor_c.data_ptr(), layout_c);
    auto tensor_d_device_ref = cutlass::TensorRef<ElementOutput, LayoutOutput>((ElementOutput*)tensor_d.data_ptr(), layout_d);
    auto tensor_e_reordered_device_ref = meta_reordered_device_ref;

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha(1);
    ElementComputeEpilogue beta(1);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later
    // passed as arguments to launch instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        problem_size, // <- problem size of matrix multiplication
        tensor_a_device_ref, // <- reference to matrix A on device
        tensor_b_device_ref, // <- reference to matrix B on device
        tensor_c_device_ref, // <- reference to matrix C on device
        tensor_d_device_ref, // <- reference to matrix D on device
        tensor_e_reordered_device_ref, // <- reference to matrix E on device
        {alpha, beta}, // <- tuple of alpha and beta
        split_k_slices}; // <- k-dimension split factor

    Gemm gemm_op;
    cutlass::Status status;

    // FIXME: the can_implement() report "Misaligned Operand" error
    // here because of tensor_c seemingly having wrong size.
    // status = gemm_op.can_implement(arguments);
    // CUTLASS_STATUS_CHECK(status);

    // FIXME: try to do the allocation using PyTorch here.
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get(), at::cuda::getCurrentCUDAStream());
    CUTLASS_STATUS_CHECK(status);

    status = gemm_op.run(at::cuda::getCurrentCUDAStream());
    CUTLASS_STATUS_CHECK(status);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(tensor_d, meta_reordered);
}

// FIXME: Pull back in device and cuda version constraints.
std::tuple<Tensor, Tensor> _cutlass_linear(
      const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c,
      const Tensor& mask_or_meta) {
    TORCH_CHECK(tensor_a.layout() == Layout::Strided, "torch._cutlass_linear: Expected tensor_a argument to be strided, but got layout ", tensor_a.layout());
    TORCH_CHECK(tensor_a.dim() == 2, "torch._cutlass_linear: Expected tensor_a argument to be 2D tensor, got ", tensor_a.dim(), " dims");
    const auto strides_a = tensor_a.strides();
    TORCH_CHECK((strides_a[0] == 1 || strides_a[1] == 1) && strides_a[0] != strides_a[1], "torch._cutlass_linear: Invalid strides for tensor_a argument: row stride = ", strides_a[0], ", column stride = ", strides_a[1]);

    TORCH_CHECK(tensor_b.layout() == Layout::Strided, "torch._cutlass_linear: Expected tensor_b argument to be strided, but got layout ", tensor_b.layout());
    TORCH_CHECK(tensor_b.dim() == 2, "torch._cutlass_linear: Expected tensor_b argument to be 2D tensor, got ", tensor_b.dim(), " dims");
    const auto strides_b = tensor_b.strides();
    TORCH_CHECK((strides_b[0] == 1 || strides_b[1] == 1) && strides_b[0] != strides_b[1], "torch._cutlass_linear: Invalid strides for tensor_b argument: row stride = ", strides_b[0], ", column stride = ", strides_b[1]);

    TORCH_CHECK(tensor_c.layout() == Layout::Strided, "torch._cutlass_linear: Expected tensor_c argument to be strided, but got layout ", tensor_c.layout());
    TORCH_CHECK(tensor_c.dim() == 2, "torch._cutlass_linear: Expected tensor_c argument to be 2D tensor, got ", tensor_c.dim(), " dims");
    const auto strides_c = tensor_c.strides();
    TORCH_CHECK(strides_c[0] == 1 && strides_c[1] == 1, "torch._cutlass_linear: Invalid strides for tensor_c argument: row stride = ", strides_c[0], ", column stride = ", strides_c[1]);

    auto tensor_a_row_major = strides_a[1] == 1;
    auto tensor_a_stride = tensor_a_row_major ? strides_a[0] : strides_a[1];
    auto tensor_b_row_major = strides_b[1] == 1;
    auto tensor_b_stride = tensor_b_row_major ? strides_b[0] : strides_b[1];

    std::tuple<Tensor, Tensor> result;
    AT_DISPATCH_SWITCH(
        tensor_a.scalar_type(),
        "_cutlass_linear",
        AT_DISPATCH_CASE(
            at::ScalarType::Char,
            [&]() {
                using ElementInputA = int8_t;
                using ElementInputB = int8_t;
                using ElementOutput = int32_t;
                using ElementAccumulator = int32_t;
                using ElementComputeEpilogue = int32_t;
                using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
                using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
                if (tensor_a_row_major && !tensor_b_row_major) {
                    result = cutlass_sgemm<
                        ElementInputA,
                        ElementInputB,
                        ElementOutput,
                        ElementAccumulator,
                        ElementComputeEpilogue,
                        ThreadblockShape,
                        WarpShape,
                        InstructionShape,
                        cutlass::layout::RowMajor,
                        cutlass::layout::ColumnMajor>(
                        tensor_a,
                        tensor_b,
                        tensor_c,
                        mask_or_meta,
                        tensor_a_stride,
                        tensor_b_stride);
                    return;
                }

                AT_ERROR("torch._cutlass_linear: the combination of tensor_a in ", tensor_a_row_major ? "row-major" : "column_major", " layout and tensor_b in ", tensor_b_row_major ? "row-major" : "column_major", " layout is not supported");
            })
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&]() {
                using ElementInputA = cutlass::half_t;
                using ElementInputB = cutlass::half_t;
                using ElementOutput = cutlass::half_t;
                using ElementAccumulator = float;
                using ElementComputeEpilogue = cutlass::half_t;
                using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
                using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
                using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
                if (tensor_a_row_major && tensor_b_row_major) {
                    result = cutlass_sgemm<
                        ElementInputA,
                        ElementInputB,
                        ElementOutput,
                        ElementAccumulator,
                        ElementComputeEpilogue,
                        ThreadblockShape,
                        WarpShape,
                        InstructionShape,
                        cutlass::layout::RowMajor,
                        cutlass::layout::RowMajor>(
                        tensor_a,
                        tensor_b,
                        tensor_c,
                        mask_or_meta,
                        tensor_a_stride,
                        tensor_b_stride);
                    return;
                }
                if (tensor_a_row_major && !tensor_b_row_major) {
                    result = cutlass_sgemm<
                        ElementInputA,
                        ElementInputB,
                        ElementOutput,
                        ElementAccumulator,
                        ElementComputeEpilogue,
                        ThreadblockShape,
                        WarpShape,
                        InstructionShape,
                        cutlass::layout::RowMajor,
                        cutlass::layout::ColumnMajor>(
                        tensor_a,
                        tensor_b,
                        tensor_c,
                        mask_or_meta,
                        tensor_a_stride,
                        tensor_b_stride);
                    return;
                }
                if (!tensor_a_row_major && tensor_b_row_major) {
                    result = cutlass_sgemm<
                        ElementInputA,
                        ElementInputB,
                        ElementOutput,
                        ElementAccumulator,
                        ElementComputeEpilogue,
                        ThreadblockShape,
                        WarpShape,
                        InstructionShape,
                        cutlass::layout::ColumnMajor,
                        cutlass::layout::RowMajor>(
                        tensor_a,
                        tensor_b,
                        tensor_c,
                        mask_or_meta,
                        tensor_a_stride,
                        tensor_b_stride);
                    return;
                }
                if (!tensor_a_row_major && !tensor_b_row_major) {
                    result = cutlass_sgemm<
                        ElementInputA,
                        ElementInputB,
                        ElementOutput,
                        ElementAccumulator,
                        ElementComputeEpilogue,
                        ThreadblockShape,
                        WarpShape,
                        InstructionShape,
                        cutlass::layout::ColumnMajor,
                        cutlass::layout::ColumnMajor>(
                        tensor_a,
                        tensor_b,
                        tensor_c,
                        mask_or_meta,
                        tensor_a_stride,
                        tensor_b_stride);
                    return;
                }
            }));
    return result;
}

} // namespace native
} // namespace at
