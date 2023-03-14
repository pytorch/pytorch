/*
The following source file implements a sparse linear operator using CUTLASS
*/

#include <ATen/Functions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <torch/custom_class.h>
#include <iostream>

#include <iostream>

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <functional>
#include <limits>
#include <typeinfo>

#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }

namespace at {
namespace native {

__global__ void two_four_create_meta_kernel(
      const int length_m, const int length_k, const int stride_nz,
      const bool* nz, const int stride_meta, uint16_t* meta, int* error) {
    const auto k = blockDim.x * blockIdx.x + threadIdx.x;
    const auto m = blockDim.y * blockIdx.y + threadIdx.y;

    // FIXME: use shared memory, eventually also other optimizations!

    const auto in_range = m < length_m && k < length_k / 4;
    unsigned mask = __ballot_sync(0xffffffff, in_range);
    if (!in_range) {
      return;
    }

    uint16_t val = 0;
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
      // Report an error.
      atomicExch(error, 1);
    }

    val |= __shfl_down_sync(mask, val, 1) << 4;
    val |= __shfl_down_sync(mask, val, 2) << 8;
    if (k % 4 == 0) {
        meta[m * stride_meta + k / 4] = val;
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

template <typename LayoutInputA, typename LayoutInputB>
std::tuple<Tensor, Tensor> do_sgemm(const Tensor& tensor_a,
                                    const Tensor& tensor_b,
                                    const Tensor& tensor_c,
                                    const Tensor& mask_or_meta,
                                    const LayoutInputA& layout_a,
                                    const LayoutInputB& layout_b) {
    auto tensor_d = tensor_a.new_empty({tensor_a.size(0), tensor_b.size(1)});

    using ElementInputA =
        cutlass::half_t; // <- data type of elements in input matrix A
    using ElementInputB =
        cutlass::half_t; // <- data type of elements in input matrix B
    using ElementOutput =
        cutlass::half_t; // <- data type of elements in output matrix

    using LayoutOutput = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::SparseGemm<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 128, 64>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<
            cutlass::half_t,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        6>;

    // Data type and layout of meta data matrix E can be inferred
    // from template Gemm.
    using ElementInputE = typename Gemm::ElementE;
    using LayoutInputE = cutlass::layout::RowMajor;
    using ReorderedLayoutInputE = typename Gemm::LayoutE;

    constexpr int kSparse = Gemm::kSparse;

    Tensor meta_reordered;
    if (mask_or_meta.dtype() == at::kBool) {
        auto mask = mask_or_meta;

        TORCH_CHECK(mask.layout() == Layout::Strided, "torch._cutlass_linear: Expected mask argument to be strided, but got layout ", mask.layout());

        const int length_m = mask.size(0);
        const int length_k = mask.size(1);

        // FIXME: remove these checks!
        TORCH_CHECK(mask.size(0) % 16 == 0);
        TORCH_CHECK(mask.size(1) % 64 == 0);

        auto meta = mask.new_empty({length_m, length_k / 16}, at::TensorOptions().dtype(at::kShort));
        auto error = mask.new_zeros({1}, at::TensorOptions().dtype(at::kInt));

        // FIXME: verify that mask is in row major format, or pass
        // both strides to the kernel below.

        two_four_create_meta_kernel<<<
            dim3((length_k + 63) / 64, (length_m + 15) / 16),
            dim3(16, 16),
            0,
            at::cuda::getCurrentCUDAStream()>>> (
                length_m, length_k, mask.stride(0), (bool*)mask.data_ptr(),
                meta.stride(0), (uint16_t*)meta.data_ptr(),
                (int*)error.data_ptr());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        TORCH_CHECK(error.item().equal(0), "Mask matrix is not 2:4 sparse");

        meta_reordered = meta.new_empty({length_m, length_k / 16});

        auto meta_device_ref = cutlass::TensorRef<uint16_t, cutlass::layout::RowMajor>((uint16_t*)meta.data_ptr(), cutlass::layout::RowMajor(meta.stride(0)));
        auto meta_reordered_device_ref = cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>((ElementInputE*)meta_reordered.data_ptr(), ReorderedLayoutInputE::packed({length_m, length_k / 16}));

        reorder_meta_kernel<<<
            dim3(length_k / 64, length_m / 16),
            dim3(16, 16),
            0,
            at::cuda::getCurrentCUDAStream()>>> (
                length_m, length_k / 16, meta_device_ref,
                meta_reordered_device_ref);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else {
        meta_reordered = mask_or_meta;
    }

    const int length_m = tensor_a.size(0);
    const int length_k = tensor_b.size(0);
    const int length_n = tensor_b.size(1);

    TORCH_CHECK(tensor_a.size(1) == length_k / kSparse);

    // FIXME: remove these checks!
    static_assert(sizeof(uint16_t) == sizeof(ElementInputE));
    TORCH_CHECK(length_m % 16 == 0);
    // TORCH_CHECK(length_k % 16 == 0);
    TORCH_CHECK(length_k % 64 == 0);

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

    auto meta_reordered_device_ref = cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>((ElementInputE*)meta_reordered.data_ptr(), ReorderedLayoutInputE::packed({length_m, length_k / 16}));

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    LayoutOutput layout_c(tensor_c.stride(0));
    LayoutOutput layout_d(tensor_d.stride(0));
    auto tensor_a_device_ref = cutlass::TensorRef<ElementInputA, LayoutInputA>((ElementInputA*)tensor_a.data_ptr(), layout_a);
    auto tensor_b_device_ref = cutlass::TensorRef<ElementInputB, LayoutInputB>((ElementInputB*)tensor_b.data_ptr(), layout_b);
    auto tensor_c_device_ref = cutlass::TensorRef<ElementInputB, LayoutOutput>((ElementOutput*)tensor_c.data_ptr(), layout_c);
    auto tensor_d_device_ref = cutlass::TensorRef<ElementInputB, LayoutOutput>((ElementOutput*)tensor_d.data_ptr(), layout_d);
    auto tensor_e_device_ref = meta_reordered_device_ref;

    // Initialize alpha and beta for dot product computation
    float alpha = 1;
    float beta = 1;

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
            tensor_e_device_ref, // <- reference to matrix E on device
            {alpha, beta}, // <- tuple of alpha and beta
            split_k_slices}; // <- k-dimension split factor

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_STATUS_CHECK(status);

    // FIXME: check is CUTLASS workspace allocation needed here!

    status = gemm_op.initialize(arguments, nullptr, at::cuda::getCurrentCUDAStream());
    CUTLASS_STATUS_CHECK(status);

    status = gemm_op.run(at::cuda::getCurrentCUDAStream());
    CUTLASS_STATUS_CHECK(status);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(tensor_d, meta_reordered);
}

// TODO: Pull back in device and cuda version constraints.
std::tuple<Tensor, Tensor> _cutlass_linear(
      const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c,
      const Tensor& mask_or_meta) {
    TORCH_CHECK(tensor_a.dim() == 2, "torch._cutlass_linear: Expected tensor_a argument to be 2D tensor, got ", tensor_a.dim(), " dims");
    TORCH_CHECK(tensor_a.layout() == Layout::Strided, "torch._cutlass_linear: Expected tensor_a argument to be strided, but got layout ", tensor_a.layout());
    const auto strides_a = tensor_a.strides();
    TORCH_CHECK((strides_a[0] == 1 || strides_a[1] == 1) && strides_a[0] != strides_a[1], "torch._cutlass_linear: Invalid strides for tensor_a argument: row stride = ", strides_a[0], ", column stride = ", strides_a[1]);

    TORCH_CHECK(tensor_b.dim() == 2, "torch._cutlass_linear: Expected tensor_b argument to be 2D tensor, got ", tensor_b.dim(), " dims");
    TORCH_CHECK(tensor_b.layout() == Layout::Strided, "torch._cutlass_linear: Expected tensor_b argument to be strided, but got layout ", tensor_b.layout());
    const auto strides_b = tensor_b.strides();
    TORCH_CHECK((strides_b[0] == 1 || strides_b[1] == 1) && strides_b[0] != strides_b[1], "torch._cutlass_linear: Invalid strides for tensor_b argument: row stride = ", strides_b[0], ", column stride = ", strides_b[1]);

    TORCH_CHECK(tensor_c.dim() == 2, "torch._cutlass_linear: Expected tensor_c argument to be 2D tensor, got ", tensor_c.dim(), " dims");
    TORCH_CHECK(tensor_c.layout() == Layout::Strided, "torch._cutlass_linear: Expected tensor_c argument to be strided, but got layout ", tensor_c.layout());
    const auto strides_c = tensor_c.strides();
    TORCH_CHECK(strides_c[1] == 1 && strides_c[0] != strides_c[1], "torch._cutlass_linear: Invalid strides for tensor_c argument: row stride = ", strides_c[0], ", column stride = ", strides_c[1]);  // Must be in row-major format.

    if (strides_a[1] == 1) {
        auto layout_a = cutlass::layout::RowMajor(strides_a[0]);
        if (strides_b[1] == 1) {
            auto layout_b = cutlass::layout::RowMajor(strides_b[0]);
            return do_sgemm(tensor_a, tensor_b, tensor_c, mask_or_meta, layout_a, layout_b);
        }
        else {
            auto layout_b = cutlass::layout::ColumnMajor(strides_b[1]);
            return do_sgemm(tensor_a, tensor_b, tensor_c, mask_or_meta, layout_a, layout_b);
        }
    }
    else {
        auto layout_a = cutlass::layout::ColumnMajor(strides_a[1]);
        if (strides_b[1] == 1) {
            auto layout_b = cutlass::layout::RowMajor(strides_b[0]);
            return do_sgemm(tensor_a, tensor_b, tensor_c, mask_or_meta, layout_a, layout_b);
        }
        else {
            auto layout_b = cutlass::layout::ColumnMajor(strides_b[1]);
            return do_sgemm(tensor_a, tensor_b, tensor_c, mask_or_meta, layout_a, layout_b);
        }
    }

    return std::make_tuple(Tensor{}, Tensor{});
}

} // namespace native
} // namespace at
