/*
The following source file implements a sparse linear operator using cusparseLt
*/

#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <torch/custom_class.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>

#include <iostream>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>
#include <cuda_runtime.h>

#include <typeinfo>
#include <limits>

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices, which will all be used as template parameters for cutlass::gemm::device::SparseGemm
using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Row Major for
// Matrix A, Column Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::SparseGemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        float,
        float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

// Data type and layout of meta data matrix E can be inferred from template Gemm.
using ElementInputE = typename Gemm::ElementE;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = typename Gemm::LayoutE;

// Below property is defined in include/cutlass/arch/sp_mma_sm80.h
// 50% Sparsity on Ampere
constexpr int kSparse = Gemm::kSparse;
// How many elements of A are covered per ElementE
constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
// The size of individual meta data
constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;

int run(
    at::Tensor tensor_a,
    at::Tensor tensor_b,
    at::Tensor tensor_c,
    at::Tensor tensor_d,
    at::Tensor meta) {
  // tensor a is m x (k // kSparse); tensor b is k x n
  const int length_m = tensor_a.size(0);
  const int length_k = tensor_b.size(0);
  const int length_n = tensor_b.size(1);

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

  TORCH_CHECK(tensor_c.size(0) == tensor_a.size(0));
  TORCH_CHECK(tensor_d.size(0) == tensor_a.size(0));

  TORCH_CHECK(tensor_c.size(1) == tensor_b.size(1));
  TORCH_CHECK(tensor_d.size(1) == tensor_b.size(1));

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  TORCH_CHECK(meta.size(0) == problem_size.m(), "meta.size(0) expected to be ", problem_size.m(), " but got ", meta.size(0));
  TORCH_CHECK(meta.size(1) == (problem_size.k() / kSparse / kElementsPerElementE), "meta.size(0) expected to be ", (problem_size.k() / kSparse / kElementsPerElementE), " but got ", meta.size(0));
  TORCH_CHECK(meta.dtype() == at::kShort, "Expected meta dtype to be int16, but got ", meta.dtype());

  TORCH_CHECK(tensor_a.device() == tensor_b.device(), "Check 0: Expected all Tensors to live on the GPU.");
  TORCH_CHECK(tensor_b.device() == tensor_c.device(), "Check 1: Expected all Tensors to live on the GPU.");
  TORCH_CHECK(tensor_c.device() == tensor_d.device(), "Check 2: Expected all Tensors to live on the GPU.");
  TORCH_CHECK(tensor_d.device() == meta.device(),     "Check 3: Expected all Tensors to live on the GPU.");

  // TODO
  // Try various valid 2:4 meta configurations
  // 0x4 0100
  // 0x8 1000
  // 0x9 1001
  // 0xc 1100
  // 0xd 1101
  // 0xe 1110

  // Initialize alpha and beta for dot product computation
  float alpha = 1;
  float beta  = 0;

  LayoutInputA layout_a(tensor_a.stride(0));
  LayoutInputB layout_b(tensor_b.stride(0));
  LayoutOutput layout_c(tensor_c.stride(0));
  LayoutOutput layout_d(tensor_d.stride(0));

  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  auto tensor_a_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutInputA>((cutlass::half_t*)tensor_a.data_ptr<at::Half>(), layout_a);
  auto tensor_b_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutInputB>((cutlass::half_t*)tensor_b.data_ptr<at::Half>(), layout_b);
  auto tensor_c_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutOutput>((cutlass::half_t*)tensor_c.data_ptr<at::Half>(), layout_c);
  auto tensor_d_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutOutput>((cutlass::half_t*)tensor_d.data_ptr<at::Half>(), layout_d);

  ReorderedLayoutInputE layout_e;
  auto tensor_e_reordered_pt_device_ref = cutlass::TensorRef<uint16_t, ReorderedLayoutInputE>((uint16_t*)meta.data_ptr<int16_t>(), layout_e);

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a_device_ref,  // <- reference to matrix A on device
                                     tensor_b_device_ref,  // <- reference to matrix B on device
                                     tensor_c_device_ref,  // <- reference to matrix C on device
                                     tensor_d_device_ref,  // <- reference to matrix D on device
                                     // tensor_e_reordered.device_ref(),  // <- reference to matrix E on device
                                     tensor_e_reordered_pt_device_ref,  // <- reference to matrix E on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  Gemm gemm_op;

  cutlass::Status status =
      gemm_op.initialize(arguments, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status != cutlass::Status::kErrorWorkspaceNull,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to workspace.");
  TORCH_CHECK(
      status != cutlass::Status::kErrorInternal,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to internal error.");
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to initialize CUTLASS Grouped GEMM kernel.");

  status = gemm_op.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to run CUTLASS Grouped GEMM kernel.");

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace at {
namespace native {

// TODO: Pull back in device and cuda version constraints.
Tensor _cusparselt_linear(const Tensor& sparse, const Tensor& dense, const Tensor& meta) {
  auto result = sparse.new_empty({sparse.size(0), dense.size(1)}); //.fill_(1);
  run(sparse, dense, result, result, meta);
  return result;
}

}
}
