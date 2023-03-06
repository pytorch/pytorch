/*
The following source file implements a sparse linear operator using cusparseLt
*/

#include "c10/core/ScalarType.h"
#include "c10/util/Half.h"
#include <torch/custom_class.h>
#include <cusparseLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
//#include <cuda_runtime_api.h>
#include <cusparse.h>

#define CHECK_CUDA(func)                                      \
  {                                                           \
    cudaError_t status = (func);                              \
    TORCH_CHECK(status == cudaSuccess, "CUDA API failed at line %d with error: %s (%d)\n", \
          __LINE__,                                           \
          cudaGetErrorString(status),                         \
          status)                                             \
  }

#define CHECK_CUSPARSE(func)                                      \
  {                                                               \
    cusparseStatus_t status = (func);                             \
    TORCH_CHECK((status == CUSPARSE_STATUS_SUCCESS),             \
          "CUSPARSE API failed at line %d with error: %s (%d)\n", \
          __LINE__,                                               \
          cusparseGetErrorString(status),                         \
          status);                                                \
  }

// create a container that holds relevant data for cusparselt linear
// TODO: template this class based on dtype or figure out another way to
// make dtype variable
//
struct CusparseLtLinear : torch::CustomClassHolder {
  at::Tensor weight;
  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t weight_descriptor, activation_descriptor, matC;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseOperation_t opA;
  cusparseOperation_t opB;
  c10::Half *dA, *dB, *dC, *dD, *dA_compressed, *dBias; 
  float alpha{1.0};
  float beta{0.0};
  int num_streams{0};
  cudaStream_t stream{nullptr};
  cudaStream_t* streams{nullptr};
  void* d_workspace{nullptr};
  void* dA_compressedBuffer{nullptr};
  int* d_valid;

  CusparseLtLinear() = delete;
  CusparseLtLinear(const at::Tensor& weight) : weight{weight}{};

  void init(const at::Tensor& activation, const at::Tensor& res, const at::Tensor& bias);
  void prune();
  void compress();
  void search_matmul_algo();
  void masked_mm();
};


// https://docs.nvidia.com/cuda/cusparselt/getting_started.html
// A, B, C, D in the above link corresponds to weight, activation, offset, and output
// this function does all the cuSPARSELt initial preparation stuff
void CusparseLtLinear::init(const at::Tensor& activation, const at::Tensor& res,
                            const at::Tensor& bias) {

  int major_cc, minor_cc;
  CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                     cudaDevAttrComputeCapabilityMajor, 0) )
  CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                     cudaDevAttrComputeCapabilityMinor, 0) )
  if (!(major_cc == 8 && minor_cc == 0) &&
      !(major_cc == 8 && minor_cc == 6) &&
      !(major_cc == 8 && minor_cc == 9)) {
      std::printf("\ncusparseLt is supported only on GPU devices with"
                  " compute capability == 8.0, 8.6, 8.9 current: %d.%d\n\n",
                   major_cc, minor_cc);
      return;
  }

  // SETTING UP VALUES 
  //--------------------------------------------------------------------------
  int num_batches = weight.size(0);
  // m & k are for weight I think, k & n are for activation
  int64_t m = weight.size(1);
  int64_t k = weight.size(2);
  int64_t n = activation.size(1); // this is assuming num_batches > 1
  //int64_t batch_strideA = 0; // setting this to 0 allows broadcasting of A (weight) tensor for multi-batch gemm
  int64_t batch_strideA = m * k + 128; 
  int64_t batch_strideB = k * n + 128;
  int64_t batch_strideC = m * n + 128;

  opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  constexpr auto order = CUSPARSE_ORDER_ROW;
  constexpr auto type = CUDA_R_16F;
  constexpr auto compute_type = CUSPARSE_COMPUTE_16F;

  // TODO: may need to adjust logic if transpose is passed in
  // TODO: make variable names more descriptive of weight, activation, bias, etc..
  bool is_rowmajor = (order == CUSPARSE_ORDER_ROW);
  bool isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
  bool isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
  auto     num_A_rows     = (isA_transposed) ? k : m;
  auto     num_A_cols     = (isA_transposed) ? m : k;
  auto     num_B_rows     = (isB_transposed) ? n : k;
  auto     num_B_cols     = (isB_transposed) ? k : n;
  auto     num_C_rows     = m;
  auto     num_C_cols     = n;
  // TODO: is alignment dtype dependent? 16 was the default setting on spmma2 example for fp16
  unsigned alignment      = 16;
  auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
  auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
  auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
  //auto     C_size         = num_batches * batch_strideC;
  // TODO: make this a function of dtype when dtype is a user input
  //auto     C_size_bytes   = num_batches * batch_strideC * sizeof(__half);

  // logging
  //std::cout << num_batches << std::endl;
  //std::cout << m << std::endl;
  //std::cout << k << std::endl;
  //std::cout << n << std::endl;
  //std::cout<< "isA_transposed:" << isA_transposed <<std::endl;
  //std::cout<< "isB_transposed:" << isB_transposed <<std::endl;

  dA = weight.data_ptr<c10::Half>();
  dB = activation.data_ptr<c10::Half>();
  dC = res.data_ptr<c10::Half>();
  dD = res.data_ptr<c10::Half>();
  dBias = bias.data_ptr<c10::Half>();

  // TODO: we may consider removing C or improving the usability;
  // right now, we assume it's not used (beta is initialized to 0)
  //__half *hC = (__half*)malloc(C_size_bytes);
  //for (int i = 0; i < C_size; i++)
      //hC[i] = static_cast<__half>(0.0);

  //CHECK_CUDA(cudaMalloc((void**)&dC, C_size_bytes))
  //CHECK_CUDA(cudaMemcpy(dC, hC, C_size_bytes, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMalloc((void**)&d_valid, sizeof(*d_valid)))
  
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE(cusparseLtInit(&handle))

  // matrix descriptor initilization
  //
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      &handle, &weight_descriptor, num_A_rows, num_A_cols,
      lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT))

  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
      &handle, &activation_descriptor, num_B_rows, num_B_cols, ldb, alignment, type, order))

  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
      &handle, &matC, num_C_rows, num_C_cols, ldc, alignment, type, order))


  // SET NUM BATCHES
  // SET BATCH STRIDE
  // if batch_strideA = 0, the matrix multiplication performs a broadcast of
  // the matrix A
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &weight_descriptor,
                                                CUSPARSELT_MAT_NUM_BATCHES,
                                                &num_batches, sizeof(num_batches)) )
  CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &activation_descriptor,
                                                CUSPARSELT_MAT_NUM_BATCHES,
                                                &num_batches, sizeof(num_batches)) )
  CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &matC,
                                                CUSPARSELT_MAT_NUM_BATCHES,
                                                &num_batches, sizeof(num_batches)) )
  CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &weight_descriptor,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideA,
                                                sizeof(batch_strideA)) )
  CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &activation_descriptor,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideB,
                                                sizeof(batch_strideB)) )
  CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&handle, &matC,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideC,
                                                sizeof(batch_strideC)) )


  //--------------------------------------------------------------------------
  // matmul, algorithm selection, and plan initialization
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                          &handle, &matmul, opA, opB,
                                          &weight_descriptor, &activation_descriptor, &matC, &matC,
                                          compute_type) )

  // SET BIAS POINTER
  //--------------------------------------------------------------------------
  CHECK_CUSPARSE( cusparseLtMatmulDescSetAttribute(&handle, &matmul,
                                                   CUSPARSELT_MATMUL_BIAS_POINTER,
                                                   &dBias, sizeof(dBias)) )


  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))

  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(
      &handle, &plan, &matmul, &alg_sel))

  size_t workspace_size = 0;
  CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size) )
  CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )
}

// TODO: make this a user input
// see https://docs.nvidia.com/cuda/cusparselt/types.html for pruning_algo choices
void CusparseLtLinear::prune() {
  constexpr cusparseLtPruneAlg_t pruning_algo = CUSPARSELT_PRUNE_SPMMA_STRIP;
  
  CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul,
                                        dA, dA,
                                        pruning_algo, stream) )

  CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul,
                                            dA, d_valid, stream) )

  int is_valid;
  cudaDeviceSynchronize();
  CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(is_valid),
                              cudaMemcpyDeviceToHost, stream) )
  CHECK_CUDA( cudaStreamSynchronize(stream) )

  TORCH_CHECK(is_valid == 0, "!!!! The matrix has been pruned in a wrong way. "
              "cusparseLtMatmul will not provide correct results");
}

void CusparseLtLinear::compress() {
  
  size_t compressed_size, compressed_buffer_size;
  CHECK_CUSPARSE(
      cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compressed_buffer_size))

  CHECK_CUDA(cudaMalloc((void**)&dA_compressed, compressed_size))
  CHECK_CUDA(cudaMalloc((void**)&dA_compressedBuffer, compressed_buffer_size))

  CHECK_CUSPARSE(
    cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, dA_compressedBuffer, stream))
}

void CusparseLtLinear::search_matmul_algo() {

  CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                          dA_compressed, dB, &beta,
                                          dC, dD, d_workspace,
                                          streams, num_streams) )
}

  // TODO: cache alg_id?
  //int alg_id;
  //CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                          //&handle, &alg_sel,
                                          //CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                          //&alg_id, sizeof(alg_id)) )

// this function assumes the weight tensor already has the mask applied
void CusparseLtLinear::masked_mm() {

  CHECK_CUSPARSE( cusparseLtMatmul(
      &handle,
      &plan,
      &alpha,
      dA_compressed,
      dB,
      &beta,
      dC,
      dD,
      d_workspace,
      streams,
      num_streams) )
}

TORCH_LIBRARY(cusparselt, m) {
  m.class_<CusparseLtLinear>("CusparseLtLinear")
    .def(torch::init<const at::Tensor&>())
    .def("init", &CusparseLtLinear::init)
    .def("prune", &CusparseLtLinear::prune)
    .def("compress", &CusparseLtLinear::compress)
    .def("search_matmul_algo", &CusparseLtLinear::search_matmul_algo)
    .def("masked_mm", &CusparseLtLinear::masked_mm)
  ;
}
