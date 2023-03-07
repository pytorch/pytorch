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
struct CusparseLtLinear : torch::CustomClassHolder {
  at::Tensor weight;
  cusparseLtHandle_t handle;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  c10::Half *dA, *dB, *dC, *dD, *dA_compressed, *dBias; 
  float alpha{1.0};
  float beta{0.0};
  int  num_streams{0};
  cudaStream_t stream{nullptr};
  cudaStream_t* streams{nullptr};
  void* d_workspace{nullptr};
  int* d_valid;

  CusparseLtLinear() = delete;
  CusparseLtLinear(const at::Tensor& weight) : weight{weight}{};

  void init(const at::Tensor& res, const at::Tensor& input, const at::Tensor& bias, const at::Tensor& zeros);
  void prune();
  void compress();
  void search_matmul_algo();
  void masked_mm(const at::Tensor& input);
  //void masked_mm(const at::Tensor& input);
};


// https://docs.nvidia.com/cuda/cusparselt/getting_started.html
// A, B, C, D in the above link corresponds to weight, input, offset, and output
// this function does all the cuSPARSELt initial preparation stuff
void CusparseLtLinear::init(const at::Tensor& res, 
                            const at::Tensor& input, 
                            const at::Tensor& bias,
                            const at::Tensor& zeros) {

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
  // m & k are for weight I think, k & n are for input
  //--------------------------------------------------------------------------
  int64_t m = weight.size(0);
  int64_t k = weight.size(1);
  int64_t n = input.size(1);

  bool isB_transposed = !input.is_contiguous();

  constexpr auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opB = isB_transposed? CUSPARSE_OPERATION_TRANSPOSE: CUSPARSE_OPERATION_NON_TRANSPOSE;
  constexpr auto order = CUSPARSE_ORDER_ROW;
  constexpr auto type = CUDA_R_16F;
  constexpr auto compute_type = CUSPARSE_COMPUTE_16F;

  // TODO: may need to adjust logic if transpose is passed in
  bool is_rowmajor = (order == CUSPARSE_ORDER_ROW);
  bool isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);

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
  
  dA = weight.data_ptr<c10::Half>();
  dB = input.data_ptr<c10::Half>();
  dC = zeros.data_ptr<c10::Half>();
  dD = res.data_ptr<c10::Half>();
  dBias = bias.data_ptr<c10::Half>();

  //--------------------------------------------------------------------------
  CHECK_CUDA(cudaMalloc((void**)&d_valid, sizeof(*d_valid)))
  
  // matrix descriptor initilization
  //--------------------------------------------------------------------------
  cusparseLtMatDescriptor_t weight_descriptor, activation_descriptor, matC;
  
  CHECK_CUSPARSE(cusparseLtInit(&handle))
  CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
      &handle, &weight_descriptor, num_A_rows, num_A_cols,
      lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT))
  CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
      &handle, &activation_descriptor, num_B_rows, num_B_cols, ldb, alignment, type, order))
  CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
      &handle, &matC, num_C_rows, num_C_cols, ldc, alignment, type, order))
  
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

  //--------------------------------------------------------------------------
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
  
  void* dA_compressedBuffer = nullptr;

  CHECK_CUDA(cudaMalloc((void**)&dA_compressed, compressed_size))
  CHECK_CUDA(cudaMalloc((void**)&dA_compressedBuffer, compressed_buffer_size))

  CHECK_CUSPARSE(
    cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, dA_compressedBuffer, stream))
}

void CusparseLtLinear::search_matmul_algo() {
  CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                          dA_compressed, dB, &beta,
                                          dC, dD, nullptr,
                                          streams, num_streams) )
}

// this function assumes the weight tensor already has the mask applied
void CusparseLtLinear::masked_mm(const at::Tensor& input) {

  dB = input.data_ptr<c10::Half>();

  CHECK_CUSPARSE( cusparseLtMatmul(
      &handle,
      &plan,
      &alpha,
      dA_compressed,
      dB,
      //input.data_ptr<c10::Half>(),
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
