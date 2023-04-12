/*
The following source file implements a sparse linear operator using cusparseLt
*/
#include <torch/custom_class.h>
#include <torch/torch.h>
#include "c10/core/ScalarType.h"
#include "c10/util/Half.h"
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
  cusparseLtMatDescriptor_t weight_descriptor, activation_descriptor, res_descriptor;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;
  void* dBias;  
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

  cusparseLtPruneAlg_t pruning_algo;
  cusparseOperation_t opA{CUSPARSE_OPERATION_NON_TRANSPOSE};
  cudaDataType type = CUDA_R_16F;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;

  at::Tensor masked_mm(const at::Tensor& input);
  void set_compressed(const at::Tensor& weight);

  // defining constructor, which will prune and compress the weights
  CusparseLtLinear(const at::Tensor& weight_compressed,
                   const at::Tensor& bias)
  : weight_compressed{weight_compressed},
    dBias{bias.data_ptr()},
    pruning_algo{CUSPARSELT_PRUNE_SPMMA_STRIP}
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
  bool isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);

  num_A_rows     = (isA_transposed) ? k : m;
  auto     num_A_cols     = (isA_transposed) ? m : k;
  auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;

  
  CHECK_CUDA( cudaMalloc((void**)&d_valid, sizeof(*d_valid)) )

  CHECK_CUSPARSE(
    cusparseLtStructuredDescriptorInit(
      &handle,
      &weight_descriptor,
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
      &weight_descriptor, 
      true, 
      opA,
      weight.data_ptr(),
      weight.data_ptr(),
      pruning_algo,
      stream) )
  CHECK_CUSPARSE(
    cusparseLtSpMMAPruneCheck2(
      &handle,
      &weight_descriptor,
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
      &weight_descriptor,
      &compressed_size,
      &compressed_buffer_size) )
  
  void* dA_compressedBuffer = nullptr;

  // CHECK_CUDA( cudaMalloc((void**)&dA_compressed, compressed_size) )
  CHECK_CUDA( cudaMalloc((void**)&dA_compressedBuffer, compressed_buffer_size) )

  CHECK_CUSPARSE(
    cusparseLtSpMMACompress2(
      &handle,
      &weight_descriptor,
      true,
      opA,
      weight.data_ptr(),
      weight_compressed.data_ptr(),
      dA_compressedBuffer,
      stream) )

}

// this function assumes the weight tensor already has the mask applied
at::Tensor CusparseLtLinear::masked_mm(const at::Tensor& input) {
  // create tensor
  auto res = input.new_empty({input.size(0), num_A_rows, input.size(2)});

  int num_batches = (int)input.size(0);
  int64_t k = input.size(1);
  int64_t n = input.size(2);

  bool isB_transposed = !input.is_contiguous();
  auto opB = isB_transposed? CUSPARSE_OPERATION_TRANSPOSE: CUSPARSE_OPERATION_NON_TRANSPOSE;

  auto     num_B_rows     = (isB_transposed) ? n : k;
  auto     num_B_cols     = (isB_transposed) ? k : n;
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
      &activation_descriptor,
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

  // set options
  int64_t batch_strideA = 0;
  int64_t batch_strideB = k * n;
  int64_t batch_strideC = num_A_rows * n;

  CHECK_CUSPARSE(
    cusparseLtMatDescSetAttribute(
      &handle,
      &weight_descriptor,
      CUSPARSELT_MAT_NUM_BATCHES,
      &num_batches,
      sizeof(num_batches)) )

  CHECK_CUSPARSE(
    cusparseLtMatDescSetAttribute(
      &handle,
      &activation_descriptor,
      CUSPARSELT_MAT_NUM_BATCHES,
      &num_batches,
      sizeof(num_batches)) )

  CHECK_CUSPARSE(
    cusparseLtMatDescSetAttribute(
      &handle,
      &res_descriptor,
      CUSPARSELT_MAT_NUM_BATCHES,
      &num_batches,
      sizeof(num_batches)) )

  CHECK_CUSPARSE(
    cusparseLtMatDescSetAttribute(
      &handle,
      &weight_descriptor,
      CUSPARSELT_MAT_BATCH_STRIDE,
      &batch_strideA,
      sizeof(batch_strideA)) )

  CHECK_CUSPARSE(
    cusparseLtMatDescSetAttribute(
      &handle,
      &activation_descriptor,
      CUSPARSELT_MAT_BATCH_STRIDE,
      &batch_strideB,
      sizeof(batch_strideB)) )

  CHECK_CUSPARSE(
    cusparseLtMatDescSetAttribute(
      &handle,
      &res_descriptor,
      CUSPARSELT_MAT_BATCH_STRIDE,
      &batch_strideC,
      sizeof(batch_strideC)) )

  // matmul, algorithm selection, and plan initialization
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE(
    cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      opA,
      opB,
      &weight_descriptor,
      &activation_descriptor,
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

  
  CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&activation_descriptor) )
  CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&res_descriptor) )
  CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )

  return res;

}

TORCH_LIBRARY(cusparselt, m) {
  m.class_<CusparseLtLinear>("CusparseLtLinear")
    .def(torch::init<const at::Tensor&, const at::Tensor&>())
    .def("masked_mm", &CusparseLtLinear::masked_mm)
    .def("set_compressed", &CusparseLtLinear::set_compressed)
  ;
}
