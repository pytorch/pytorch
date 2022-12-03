/*
The following source file implements a sparse linear operator using cusparseLt
*/

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

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }




/**
Please check example 07, 08 and 17 for the basics of dense tensor op gemm kernels.  NVIDIA Ampere
architecture also supports structured sparse tensor op for tf32, fp16, int8 and int4.

Sparse GEMM kernels needs to takes an additional E matrix which stores the meta data.  The format of
meta data is different for every data types.   CUTLASS templates can automatically infer it based on
input A and B.  Check code below.

Moreover, matrix E needs to be preprocessed so that it can use ldmatrix to load into the registers
efficiently.
*/


// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices, which will all be used as template parameters for cutlass::gemm::device::SparseGemm
using ElementAccumulator = float;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Row Major for
// Matrix A, Column Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
// What should we these be for fp16 pyspeech shapse?
// current settings are from fp16 cutlass sparse gemm unit test
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 64>;  // <- threadblock tile M = 128, N = 128, K = 256
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 256
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M = 16, N = 8, K = 128

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementAccumulator>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::SparseGemm<ElementInputA,
                                               LayoutInputA,
                                               ElementInputB,
                                               LayoutInputB,
                                               ElementOutput,
                                               LayoutOutput,
                                               ElementAccumulator,
                                               MMAOp,
                                               SmArch,
                                               ShapeMMAThreadBlock,
                                               ShapeMMAWarp,
                                               ShapeMMAOp,
                                               EpilogueOp,
                                               SwizzleThreadBlock,
                                               NumStages>;

// Data type and layout of meta data matrix E can be inferred from template Gemm.
using ElementInputE = typename Gemm::ElementE;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = typename Gemm::LayoutE;

// Blow property is defined in include/cutlass/arch/sp_mma_sm80.h
// 50% Sparsity on Ampere
constexpr int kSparse = Gemm::kSparse;
// How many elements of A are covered per ElementE
constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
// The size of individual meta data
constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;

int run() {
  // tensor a is m x k; tensor b is k x n
  const int length_m = 512;
  const int length_n = 512;
  const int length_k = 1024;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse));  // <- Create matrix A with dimensions M x (K / 2) -- this is compressed
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a_uncompressed(
      problem_size.mk());  // <- Create uncompressed matrix A with dimensions M x K for reference computing

  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Create matrix E with dimensions M x (K / 2 / kElementsPerElementE). This one is used by reference computing.
  cutlass::HostTensor<ElementInputE, LayoutInputE> tensor_e(
      cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
  // Same size as the above.  The above one needs to be reordered and stored in this one.
  cutlass::HostTensor<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
      cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));

  // Fill input and output matrices on host using CUTLASS helper functions
  // why does the output matrix need to be filled?
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(2),
      ElementInputA(-2),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(2),
      ElementInputB(-2),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(2),
      ElementOutput(-2),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomSparseMeta(
      tensor_e.host_view(),
      1,
      kMetaSizeInBits);   // <- Fill matrix E on host with uniform-distribution random meta data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Reorder the meta data matrix so that we can use ldmatrix to load them to tensor core
  // instructions.
  cutlass::reorder_meta(tensor_e_reordered.host_ref(), tensor_e.host_ref(),
                        {problem_size.m(), problem_size.n(),
                         problem_size.k() / kSparse / kElementsPerElementE});

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_e_reordered.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     tensor_e_reordered.device_ref(),  // <- reference to matrix E on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);

  // uncompress tensor_a based on meta data tensor_e. We need it for reference computing.
  cutlass::uncompress(tensor_a_uncompressed.host_ref(), tensor_a.host_ref(),
                      tensor_e.host_ref(), problem_size.m(), problem_size.k());

  // Create instantiation for host reference gemm kernel
  // this takes the activation/input matrix, which could be problematic?
  cutlass::reference::host::Gemm<ElementInputA,
                                 LayoutInputA,
                                 ElementInputB,
                                 LayoutInputB,
                                 ElementOutput,
                                 LayoutOutput,
                                 ElementComputeEpilogue,
                                 ElementComputeEpilogue,
                                 typename Gemm::Operator>
      gemm_host;

  // Launch host reference gemm kernel
  gemm_host(problem_size,
            alpha,
            tensor_a_uncompressed.host_ref(),
            tensor_b.host_ref(),
            beta,
            tensor_c.host_ref(),
            tensor_ref_d.host_ref());

  // Copy output data from CUTLASS host for comparison
  tensor_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;

  return (passed ? 0  : -1);
}

namespace at {
namespace native {

Tensor _cusparselt_linear(const Tensor& sparse, const Tensor& dense) {
  std::cout << "Calling _cusparselt_linear" << std::endl;

  // bool notSupported = false;

  // Ampere Sparse Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.1.
  //
  // CUTLASS must be compiled with CUDA 11.1 Toolkit to run these examples.

  // if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 1))) {
  //   std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.1 Toolkit or later." << std::endl;
  //   notSupported = true;
  // }

  // cudaDeviceProp props;

  // cudaError_t error = cudaGetDeviceProperties(&props, 0);
  // if (error != cudaSuccess) {
  //   std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
  //   return -1;
  // }

  // if (props.major * 10 + props.minor < 80) {
  //   std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
  //             << std::endl;
  //   notSupported = true;
  // }

  // if (notSupported) {
  //   // Returning zero so this test passes on older Toolkits. Its actions are no-op.
  //   return 0;
  // }

  std::cout << "run: " << run() << std::endl;
  return sparse.mm(dense);
}

}
}
