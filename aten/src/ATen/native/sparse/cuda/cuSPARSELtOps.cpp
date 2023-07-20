#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Half.h>
#include <cusparse.h>
#include <cusparseLt.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/Dispatch.h>
#include <cstdint>
#include <iostream>

namespace at {
namespace native {

cusparseLtHandle_t handle;
uint32_t alignment = 16;

int num_streams = 0;
cudaStream_t stream = nullptr;
cudaStream_t* streams = nullptr;
cusparseLtPruneAlg_t pruning_algo = CUSPARSELT_PRUNE_SPMMA_STRIP;
constexpr static auto order = CUSPARSE_ORDER_ROW;


Tensor _cslt_compress(
    const Tensor& sparse_input
)
{
  cudaDataType type = CUDA_R_16F;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;

  // We create the tensor to store the compressed sparse matrix (non-pruned
  // elements + mask) in python with the same dtype as the sparse input tensor
  // so we know this wil be correct.
  if (sparse_input.dtype() == c10::ScalarType::BFloat16) {
    type = CUDA_R_16BF;
  } else if (sparse_input.dtype() == c10::ScalarType::Float) {
    type = CUDA_R_32F;
    compute_type = CUSPARSE_COMPUTE_TF32_FAST;
  } else if (sparse_input.dtype() == c10::ScalarType::Char) {
    type = CUDA_R_8I;
    compute_type = CUSPARSE_COMPUTE_32I;
  }

  auto compression_factor = 9;
  auto original_size = sparse_input.numel();
  auto compressed_tensor = sparse_input.new_empty(original_size * compression_factor / 16);

  TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));

  // assert weight.contiguous()
  // For PyTorch, we assume row major order

  int64_t m = sparse_input.size(0);
  int64_t k = sparse_input.size(1);
  int64_t lda = k;

  cusparseLtMatDescriptor_t sparse_input_descriptor;

  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      m,
      k,
      lda,
      alignment,
      type,
      order,
      CUSPARSELT_SPARSITY_50_PERCENT));

  // compress input
  //--------------------------------------------------------------------------
  size_t compressed_size, compressed_buffer_size;
  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
      &handle,
      &sparse_input_descriptor,
      &compressed_size,
      &compressed_buffer_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);

  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
      &handle,
      &sparse_input_descriptor,
      true,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      sparse_input.data_ptr(),
      compressed_tensor.data_ptr(),
      compressedBufferPtr.get(),
      stream));

  return compressed_tensor;
}

Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const Tensor& bias
)
{
  // cupsarselt constructs
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  cusparseLtMatDescriptor_t dense_input_descriptor;
  cusparseLtMatDescriptor_t res_descriptor;

  float alpha = 1.0;
  float beta = 0.0;

  cudaDataType type = CUDA_R_16F;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;

  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  bool transpose_result = false;
  bool transpose_dense = !dense_B.is_contiguous();

  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  // kind of a hack
  int64_t m = (compressed_A.numel() * 16 / 9  ) / k;

  // create result tensor
  auto res = (transpose_result) ? dense_B.new_empty({n, m})
                                : dense_B.new_empty({m, n});

  cusparseOrder_t result_order =
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;

  cusparseOperation_t opB = (transpose_dense)
      ? CUSPARSE_OPERATION_TRANSPOSE
      : CUSPARSE_OPERATION_NON_TRANSPOSE;


  int64_t num_B_rows = (transpose_dense) ? n : k;
  int64_t num_B_cols = (transpose_dense) ? k : n;

  int64_t ldb = (transpose_dense) ? num_B_cols : num_B_rows;
  int64_t ldc = (transpose_result) ? m: n;

  //initialize sparse descriptor
  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      m,
      k,
      k,
      alignment,
      type,
      order,
      CUSPARSELT_SPARSITY_50_PERCENT));
  // initalize dense input descriptor
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &dense_input_descriptor,
      num_B_rows,
      num_B_cols,
      ldb,
      alignment,
      type,
      order));

  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      ldc,
      alignment,
      type,
      result_order));

  // intialize matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      opB,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &res_descriptor,
      &res_descriptor,
      compute_type));


  // set bias pointer for matmut, need to assign to get location
  void* dBias = bias.data_ptr();
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias)));

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
      &handle,
      &plan,
      &alpha,
      compressed_A.data_ptr(),
      dense_B.data_ptr(),
      &beta,
      res.data_ptr(),
      res.data_ptr(),
      workspacePtr.get(),
      streams,
      num_streams));

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));

  return res;
}

} // namespace native
} // namespace at
