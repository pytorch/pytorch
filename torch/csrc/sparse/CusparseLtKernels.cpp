#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Half.h>
#include <cusparse.h>
#include <torch/csrc/sparse/cusparseLt.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

// create a container that holds relevant data for cusparselt matmul
namespace torch {

struct CusparseLt : torch::CustomClassHolder {
  constexpr static auto order{CUSPARSE_ORDER_ROW};
  at::Tensor sparse_compressed;
  // cupsarselt constructs
  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  cusparseLtMatDescriptor_t dense_input_descriptor;
  cusparseLtMatDescriptor_t res_descriptor;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  uint32_t alignment{16};
  float alpha{1.0};
  float beta{0.0};
  int num_streams{0};
  cudaStream_t stream{nullptr};
  cudaStream_t* streams{nullptr};
  void* d_workspace{nullptr};
  void* d_A_compressed{nullptr};
  int alg_id{7777};
  int64_t num_A_rows;

  cusparseLtPruneAlg_t pruning_algo{CUSPARSELT_PRUNE_SPMMA_STRIP};
  cusparseOperation_t opA;
  cudaDataType type = CUDA_R_16F;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;

  // struct functions / constructor
  at::Tensor cusparselt_mm(
      const at::Tensor& input,
      bool transpose_dense,
      bool transpose_result);
  at::Tensor cusparselt_addmm(
      const at::Tensor& input,
      const at::Tensor& bias,
      bool transpose_dense,
      bool transpose_result);
  at::Tensor cusparselt_helper(
      const at::Tensor& input,
      void* dBias,
      int64_t biasStride,
      bool transpose_dense,
      bool transpose_result);
  void compress(const at::Tensor& sparse_input, bool transpose_sparse);
  CusparseLt(const at::Tensor& sparse_compressed)
      : sparse_compressed{sparse_compressed} {
    // Check CUDA compatibility, currently only supported on 8.0, 8.6, 8.9
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        prop->major >= 8 && prop->minor >= 0,
        "cuSPARSELt is only supported only on GPU deviced with compute capability == 8.0, 8.6, 8.9! (current: ",
        prop->major,
        prop->minor,
        ")");

    // Initialized cuSPARSELt handle
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));

    // We create the tensor to store the compressed sparse matrix (non-pruned
    // elements + mask) in python with the same dtype as the sparse input tensor
    // so we know this wil be correct.
    if (sparse_compressed.dtype() == torch::kBFloat16) {
      type = CUDA_R_16BF;
    } else if (sparse_compressed.dtype() == torch::kFloat32) {
      type = CUDA_R_32F;
      compute_type = CUSPARSE_COMPUTE_TF32_FAST;
    } else if (sparse_compressed.dtype() == torch::kInt8) {
      type = CUDA_R_8I;
      compute_type = CUSPARSE_COMPUTE_32I;
    }
  };
};

void CusparseLt::compress(
    const at::Tensor& sparse_input,
    bool is_sparse_input_transposed) {
  int64_t m = sparse_input.size(0);
  int64_t k = sparse_input.size(1);

  bool is_rowmajor = (order == CUSPARSE_ORDER_ROW);
  opA = (is_sparse_input_transposed) ? CUSPARSE_OPERATION_TRANSPOSE
                                     : CUSPARSE_OPERATION_NON_TRANSPOSE;

  num_A_rows = (is_sparse_input_transposed) ? k : m;
  int64_t num_A_cols = (is_sparse_input_transposed) ? m : k;
  int64_t lda = (is_rowmajor) ? num_A_cols : num_A_rows;

  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      num_A_rows,
      num_A_cols,
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

  void* d_compressedBuffer = nullptr;
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);
  d_compressedBuffer = compressedBufferPtr.get();

  TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
      &handle,
      &sparse_input_descriptor,
      true,
      opA,
      sparse_input.data_ptr(),
      sparse_compressed.data_ptr(),
      d_compressedBuffer,
      stream));
}

at::Tensor CusparseLt::cusparselt_mm(
    const at::Tensor& input,
    bool transpose_dense,
    bool transpose_result) {
  return CusparseLt::cusparselt_helper(
      input, nullptr, 0, transpose_dense, transpose_result);
}

at::Tensor CusparseLt::cusparselt_addmm(
    const at::Tensor& input,
    const at::Tensor& bias,
    bool transpose_dense,
    bool transpose_result) {
  return CusparseLt::cusparselt_helper(
      input, bias.data_ptr(), 0, transpose_dense, transpose_result);
}

at::Tensor CusparseLt::cusparselt_helper(
    const at::Tensor& input,
    void* dBias,
    int64_t biasStride,
    bool transpose_dense,
    bool transpose_result) {
  cusparseLtMatmulDescriptor_t matmul;

  int64_t k = input.size(0);
  int64_t n = input.size(1);

  // create tensor
  auto res = (transpose_result) ? input.new_empty({n, num_A_rows})
                                : input.new_empty({num_A_rows, n});

  cusparseOrder_t result_order =
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;
  cusparseOperation_t opB = (transpose_dense)
      ? CUSPARSE_OPERATION_TRANSPOSE
      : CUSPARSE_OPERATION_NON_TRANSPOSE;

  int64_t num_B_rows = (transpose_dense) ? n : k;
  int64_t num_B_cols = (transpose_dense) ? k : n;
  int64_t num_C_rows = num_A_rows;
  int64_t num_C_cols = n;

  int64_t ldb = num_B_cols;
  int64_t ldc = (transpose_result) ? num_C_rows : num_C_cols;

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
      num_C_rows,
      num_C_cols,
      ldc,
      alignment,
      type,
      result_order));

  // intialize matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      opA,
      opB,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &res_descriptor,
      &res_descriptor,
      compute_type));

  // set bias pointer for matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias)));
  // set bias stride
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
      &handle,
      &matmul,
      CUSPARSELT_MATMUL_BIAS_STRIDE,
      &biasStride,
      sizeof(biasStride)));

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);

  if (alg_id == 7777) {
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulSearch(
        &handle,
        &plan,
        &alpha,
        sparse_compressed.data_ptr(),
        input.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        streams,
        num_streams));
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg_id,
        sizeof(alg_id)));
  } else {
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
        &handle,
        &plan,
        &alpha,
        sparse_compressed.data_ptr(),
        input.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        d_workspace,
        streams,
        num_streams));
  }

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));

  return res;
}

} // namespace torch
