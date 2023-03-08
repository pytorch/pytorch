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

std::tuple<Tensor, Tensor> _cutlass_create_meta(const Tensor& mask) {
    const int length_m = mask.size(0);
    const int length_k = mask.size(1);

    // FIXME: remove these checks!
    TORCH_CHECK(mask.dtype() == at::kBool);
    TORCH_CHECK(mask.size(0) % 16 == 0);
    TORCH_CHECK(mask.size(1) % 64 == 0);

    auto meta = mask.new_empty({length_m, length_k / 16}, at::TensorOptions().dtype(at::kShort));
    auto error = mask.new_zeros({1}, at::TensorOptions().dtype(at::kInt));

    const dim3 block(16, 16);
    const dim3 grid((length_k + 63) / 64, (length_m + 15) / 16);
    const size_t shmem_size = 0;
    two_four_create_meta_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>> (
        length_m, length_k, mask.stride(0), (bool*)mask.data_ptr(),
        meta.stride(0), (uint16_t*)meta.data_ptr(), (int*)error.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    TORCH_CHECK(error.item().equal(0), "Mask matrix is not 2:4 sparse");

    return {meta, meta.new_empty({length_m, length_k / 16})};
}

// TODO: Pull back in device and cuda version constraints.
Tensor _cutlass_linear(const Tensor& sparse, const Tensor& dense,
                       const Tensor& meta, const Tensor& meta_reordered) {
    // The code section below describes datatype for input, output
    // matrices and computation between elements in input matrices,
    // which will all be used as template parameters for
    // cutlass::gemm::device::SparseGemm
    using ElementInputA =
        cutlass::half_t; // <- data type of elements in input matrix A
    using ElementInputB =
        cutlass::half_t; // <- data type of elements in input matrix B
    using ElementOutput =
        cutlass::half_t; // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and
    // output matrices.  Row Major for Matrix A, Column Major for
    // Matrix B and Row Major for Matrix C
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
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
    TORCH_CHECK(
        dense.size(0) % kSparse == 0,
        "Expected dense.size(0) of value ",
        dense.size(0),
        " to be evenly divisible by ",
        kSparse,
        " but got.");
    TORCH_CHECK(
        sparse.size(1) * kSparse == dense.size(0),
        "Expected sparse.size(1) of value ",
        sparse.size(1),
        " to match dense.size(0) of value ",
        dense.size(0),
        " to match after being multiplied by ",
        kSparse);

    const int length_m = sparse.size(0);
    const int length_k = dense.size(0);
    const int length_n = dense.size(1);

    // FIXME: remove these checks!
    static_assert(sizeof(uint16_t) == sizeof(ElementInputE));
    TORCH_CHECK(length_m % 16 == 0);
    // TORCH_CHECK(length_k % 16 == 0);
    TORCH_CHECK(length_k % 64 == 0);

    auto meta_device_ref = cutlass::TensorRef<uint16_t, cutlass::layout::RowMajor>((uint16_t*)meta.data_ptr(), cutlass::layout::RowMajor(meta.stride(0)));
    auto meta_reordered_device_ref = cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>((ElementInputE*)meta_reordered.data_ptr(), ReorderedLayoutInputE::packed({length_m, length_k / 16}));

    const dim3 block(16, 16);
    const dim3 grid(length_k / 64, length_m / 16);
    const size_t shmem_size = 0;
    reorder_meta_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>> (
        length_m, length_k / 16, meta_device_ref, meta_reordered_device_ref);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto result = sparse.new_empty({sparse.size(0), dense.size(1)});

    auto tensor_a = sparse;
    auto tensor_b = dense;
    auto tensor_c = result;
    auto tensor_d = result;

    TORCH_CHECK(tensor_a.size(0) == length_m);
    TORCH_CHECK(tensor_a.size(1) == length_k / kSparse);
    TORCH_CHECK(tensor_b.size(0) == length_k);
    TORCH_CHECK(tensor_b.size(1) == length_n);

    TORCH_CHECK(
        tensor_a.device() == tensor_b.device(),
        "Check 0: Expected all Tensors to live on the GPU.");
    TORCH_CHECK(
        tensor_b.device() == tensor_c.device(),
        "Check 1: Expected all Tensors to live on the GPU.");
    TORCH_CHECK(
        tensor_c.device() == tensor_d.device(),
        "Check 2: Expected all Tensors to live on the GPU.");

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    LayoutInputA layout_a(tensor_a.stride(0));
    LayoutInputB layout_b(tensor_b.stride(0));
    LayoutOutput layout_c(tensor_c.stride(0));
    LayoutOutput layout_d(tensor_d.stride(0));
    auto tensor_a_device_ref = cutlass::TensorRef<ElementInputA, LayoutInputA>((ElementInputA*)tensor_a.data_ptr(), layout_a);
    auto tensor_b_device_ref = cutlass::TensorRef<ElementInputB, LayoutInputB>((ElementInputB*)tensor_b.data_ptr(), layout_b);
    auto tensor_c_device_ref = cutlass::TensorRef<ElementInputB, LayoutOutput>((ElementOutput*)tensor_c.data_ptr(), layout_c);
    auto tensor_d_device_ref = cutlass::TensorRef<ElementInputB, LayoutOutput>((ElementOutput*)tensor_d.data_ptr(), layout_d);
    auto tensor_e_device_ref = meta_reordered_device_ref;

    // Initialize alpha and beta for dot product computation
    float alpha = 1;
    float beta = 0;

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

    return result;
}

} // namespace native
} // namespace at
