/*
The following source file implements a sparse linear operator using cusparseLt
*/
#include <torch/custom_class.h>
#include <torch/torch.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cusparseLt.h>
#include <cusparse.h>

#define CHECK_CUDA(func)                                                                   \
  {                                                                                        \
    cudaError_t status = (func);                                                           \
    TORCH_CHECK(status == cudaSuccess, "CUDA API failed at line %d with error: %s (%d)\n", \
          __LINE__,                                                                        \
          cudaGetErrorString(status),                                                      \
          status)                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                               \
  {                                                                                        \
    cusparseStatus_t status = (func);                                                      \
    TORCH_CHECK((status == CUSPARSE_STATUS_SUCCESS),                                       \
          "CUSPARSE API failed at line %d with error: %s (%d)\n",                          \
          __LINE__,                                                                        \
          cusparseGetErrorString(status),                                                  \
          status);                                                                         \
  }

// create a container that holds relevant data for cusparselt linear
struct CusparseLtLinear : torch::CustomClassHolder {
  // define constants
  constexpr static auto order{CUSPARSE_ORDER_ROW};
  // this tensor is magic, will segfault when removed? 
  at::Tensor weight_compressed;
  // cupsarselt constructs
  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t sparse_input_descriptor, dense_input_descriptor, res_descriptor;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;
  float alpha{1.0};
  float beta{0.0};
  unsigned alignment{16};
  int64_t num_A_rows;
  int num_streams{0};
  cudaStream_t stream{nullptr};
  cudaStream_t* streams{nullptr};
  void* d_workspace{nullptr};
  int alg_id{7777};
  int* d_valid;

  cusparseLtPruneAlg_t pruning_algo{CUSPARSELT_PRUNE_SPMMA_STRIP};
  cusparseOperation_t opA;
  cudaDataType type = CUDA_R_16F;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;

  at::Tensor cusparselt_mm(const at::Tensor& input);
  at::Tensor cusparselt_addmm(const at::Tensor& input, const at::Tensor& bias);
  at::Tensor cusparselt_helper(const at::Tensor& input, void* dBias, int64_t biasStride);
  void set_compressed(const at::Tensor& weight);

  // defining constructor, which will prune and compress the weights
  CusparseLtLinear(const at::Tensor& weight_compressed)
  : weight_compressed{weight_compressed}
  {
    // CUDA VERSION CHECK
    // --------------------------------------------------------------------------
    int major_cc, minor_cc;
    CHECK_CUDA(
      cudaDeviceGetAttribute(
        &major_cc,
        cudaDevAttrComputeCapabilityMajor,
        0) )
    CHECK_CUDA(
      cudaDeviceGetAttribute(
        &minor_cc,
        cudaDevAttrComputeCapabilityMinor,
        0) )

    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6) &&
        !(major_cc == 8 && minor_cc == 9)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6, 8.9 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return;
    }

    // handle initilization
    //--------------------------------------------------------------------------
    CHECK_CUSPARSE( cusparseLtInit(&handle) )

    // set matrix dtype and compute type
    if (weight_compressed.dtype() == torch::kInt8) {
      type = CUDA_R_8I;
      compute_type = CUSPARSE_COMPUTE_32I;
    }
  };

};


void CusparseLtLinear::set_compressed(const at::Tensor& weight) {
  // SETTING UP VALUES 
  //--------------------------------------------------------------------------
  int64_t m = weight.size(0);
  int64_t k = weight.size(1);

  bool is_rowmajor = (order == CUSPARSE_ORDER_ROW);
  bool is_sparse_input_transposed = false;
  opA = (is_sparse_input_transposed) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

  num_A_rows     = (is_sparse_input_transposed) ? k : m;
  auto     num_A_cols     = (is_sparse_input_transposed) ? m : k;
  auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;

  
  CHECK_CUDA( cudaMalloc((void**)&d_valid, sizeof(*d_valid)) )

  CHECK_CUSPARSE(
    cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      num_A_rows,
      num_A_cols,
      lda,
      alignment,
      type,
      order,
      CUSPARSELT_SPARSITY_50_PERCENT) )
  
  // prune weights
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE(
    cusparseLtSpMMAPrune2(
      &handle,
      &sparse_input_descriptor, 
      true, 
      opA,
      weight.data_ptr(),
      weight.data_ptr(),
      pruning_algo,
      stream) )
  CHECK_CUSPARSE(
    cusparseLtSpMMAPruneCheck2(
      &handle,
      &sparse_input_descriptor,
      true,
      opA,
      weight.data_ptr(),
      d_valid,
      stream) )

  int is_valid;
  cudaDeviceSynchronize();
  CHECK_CUDA(
    cudaMemcpyAsync(
      &is_valid,
      d_valid,
      sizeof(is_valid),
      cudaMemcpyDeviceToHost,
      stream) )

  CHECK_CUDA( cudaStreamSynchronize(stream) )

  TORCH_CHECK(is_valid == 0, "!!!! The matrix has been pruned in a wrong way. "
              "cusparseLtMatmul will not provide correct results");
 
  // compress weight
  //--------------------------------------------------------------------------
  size_t compressed_size, compressed_buffer_size;
  CHECK_CUSPARSE(
    cusparseLtSpMMACompressedSize2(
      &handle,
      &sparse_input_descriptor,
      &compressed_size,
      &compressed_buffer_size) )
  
  void* dA_compressedBuffer = nullptr;

  CHECK_CUDA( cudaMalloc((void**)&dA_compressedBuffer, compressed_buffer_size) )

  CHECK_CUSPARSE(
    cusparseLtSpMMACompress2(
      &handle,
      &sparse_input_descriptor,
      true,
      opA,
      weight.data_ptr(),
      weight_compressed.data_ptr(),
      dA_compressedBuffer,
      stream) )


}

at::Tensor CusparseLtLinear::cusparselt_mm(const at::Tensor& input) {
    return CusparseLtLinear::cusparselt_helper(input, nullptr, 0);
}

at::Tensor CusparseLtLinear::cusparselt_addmm(const at::Tensor& input, const at::Tensor& bias) {
    return CusparseLtLinear::cusparselt_helper(input, bias.data_ptr(), 0);
}

at::Tensor CusparseLtLinear::cusparselt_helper(const at::Tensor& input, void* dBias, int64_t biasStride) {
  // create tensor
  cusparseLtMatmulDescriptor_t matmul;

  //if (second){
      //int64_t m = input.size(0);
      //int64_t k = input.size(1);
      //auto res = input.new_empty(m, n)
  //}
  auto res = input.new_empty({num_A_rows, input.size(1)});

  int64_t k = input.size(0);
  int64_t n = input.size(1);

  bool is_dense_input_transposed = !input.is_contiguous();
  auto opB = is_dense_input_transposed? CUSPARSE_OPERATION_TRANSPOSE: CUSPARSE_OPERATION_NON_TRANSPOSE;

  auto     num_B_rows     = (is_dense_input_transposed) ? n : k;
  auto     num_B_cols     = (is_dense_input_transposed) ? k : n;
  auto     num_C_rows     = num_A_rows;
  auto     num_C_cols     = n;

  bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
  auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
  auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;

  // B and C mat init
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE(
    cusparseLtDenseDescriptorInit(
      &handle,
      &dense_input_descriptor,
      num_B_rows,
      num_B_cols,
      ldb,
      alignment,
      type,
      order) )

  CHECK_CUSPARSE(
    cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      num_C_rows,
      num_C_cols,
      ldc,
      alignment,
      type,
      order) )

  // matmul, algorithm selection, and plan initialization
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE(
    cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      opA,
      opB,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &res_descriptor,
      &res_descriptor,
      compute_type) )

  // SET BIAS POINTER
  // --------------------------------------------------------------------------
  CHECK_CUSPARSE(
    cusparseLtMatmulDescSetAttribute(
      &handle,
      &matmul,
      CUSPARSELT_MATMUL_BIAS_POINTER,
      &dBias,
      sizeof(dBias)) )

  CHECK_CUSPARSE(
    cusparseLtMatmulDescSetAttribute(
      &handle,
      &matmul,
      CUSPARSELT_MATMUL_BIAS_STRIDE,
      &biasStride,
      sizeof(biasStride)) )

  CHECK_CUSPARSE(
    cusparseLtMatmulAlgSelectionInit(
      &handle,
      &alg_sel,
      &matmul,
      CUSPARSELT_MATMUL_ALG_DEFAULT) )

  CHECK_CUSPARSE(
      cusparseLtMatmulPlanInit(
        &handle,
        &plan,
        &matmul,
        &alg_sel) )

  size_t workspace_size;
  CHECK_CUSPARSE(
      cusparseLtMatmulGetWorkspace(
        &handle,
        &plan,
        &workspace_size) )
  CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )

  if (alg_id == 7777) {
    CHECK_CUSPARSE(
      cusparseLtMatmulSearch(
        &handle,
        &plan,
        &alpha,
        weight_compressed.data_ptr(),
        input.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        d_workspace,
        streams,
        num_streams) )
    CHECK_CUSPARSE(
      cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg_id,
        sizeof(alg_id)) )
  }
  else {
    CHECK_CUSPARSE(
      cusparseLtMatmulAlgSetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg_id,
        sizeof(alg_id)) )
  }

  CHECK_CUSPARSE(
    cusparseLtMatmul(
      &handle,
      &plan,
      &alpha,
      weight_compressed.data_ptr(),
      input.data_ptr(),
      &beta,
      res.data_ptr(),
      res.data_ptr(),
      d_workspace,
      streams,
      num_streams) )

  
  CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&dense_input_descriptor) )
  CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&res_descriptor) )
  CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )

  return res;

}

TORCH_LIBRARY(cusparselt, m) {
  m.class_<CusparseLtLinear>("CusparseLtLinear")
    .def(torch::init<const at::Tensor&>())
    .def("cusparselt_mm", &CusparseLtLinear::cusparselt_mm)
    .def("cusparselt_addmm", &CusparseLtLinear::cusparselt_addmm)
    .def("set_compressed", &CusparseLtLinear::set_compressed)
  ;
}
