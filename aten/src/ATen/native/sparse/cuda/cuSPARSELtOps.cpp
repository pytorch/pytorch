#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Half.h>
#include <cusparse.h>
#include <cstdint>
#include <iostream>

#if AT_CUSPARSELT_ENABLED()

#include <cusparseLt.h>

namespace at::native {

cusparseLtHandle_t handle;
bool handle_initialized = false;

at::Tensor _cslt_compress(const Tensor& sparse_input)
{
    if (!handle_initialized){
        TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
        handle_initialized = true;
    }
    // create sparse descriptor, dtype
    cusparseLtMatDescriptor_t sparse_input_descriptor;
    cudaDataType type;
    auto compression_factor = 9;

    switch(
        sparse_input.scalar_type()
    )
    {
        case at::ScalarType::Char:
            type = CUDA_R_8I;
            compression_factor = 10;
            break;
        case at::ScalarType::Half:
            type = CUDA_R_16F;
            break;
        case at::ScalarType::BFloat16:
            type = CUDA_R_16BF;
            break;
        case at::ScalarType::Float:
            type = CUDA_R_32F;
            break;
        default:
            TORCH_CHECK(false, "Unsupported dtype for cuSPARSELt compressed matrix");
            break;
    }

    // create a new compressed tensor with the same dtype as
    auto compressed_tensor = sparse_input.new_empty(sparse_input.numel() * compression_factor / 16);

    TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
        &handle,
        &sparse_input_descriptor,
        sparse_input.size(0),
        sparse_input.size(1),
        sparse_input.size(1),
        16,
        type,
        CUSPARSE_ORDER_ROW,
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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


at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result
)
{
  if (!handle_initialized){
      TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
      handle_initialized = true;
  }
  // cupsarselt constructs
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  bool mixed_dtype_mode = false;
  float alpha = 1.0;
  float beta = 0.0;
  cudaDataType input_type;
  cudaDataType output_type;
  cusparseComputeType compute_type;
  auto compression_factor = 9;
  c10::ScalarType pytorch_output_type;


  switch(compressed_A.scalar_type())
  {
    case at::ScalarType::Char:
        input_type = CUDA_R_8I;
        output_type = CUDA_R_8I;
        compute_type = CUSPARSE_COMPUTE_32I;
        compression_factor = 10;

        break;
    case at::ScalarType::Half:
        input_type = CUDA_R_16F;
        output_type = CUDA_R_16F;
        compute_type = CUSPARSE_COMPUTE_16F;
        break;
    case at::ScalarType::BFloat16:
        input_type = CUDA_R_16BF;
        output_type = CUDA_R_16BF;
        compute_type = CUSPARSE_COMPUTE_16F;
        break;
    case at::ScalarType::Float:
        input_type = CUDA_R_32F;
        output_type = CUDA_R_32F;
        compute_type = CUSPARSE_COMPUTE_TF32;
        break;
    default:
        TORCH_CHECK(false, "Unsupported dtype for cuSPARSE compressed matrix multiplication.");
        break;
  }

  // special check for int8 int8 -> fp16 support
  if (out_dtype_opt.has_value()) {
    ScalarType out_dtype = out_dtype_opt.value();
    if (input_type == CUDA_R_8I and out_dtype == at::ScalarType::Half)
    {
        output_type = CUDA_R_16F;
        mixed_dtype_mode = true;
        pytorch_output_type = out_dtype;
    }
    else if (input_type == CUDA_R_8I and out_dtype == at::ScalarType::Int)
    {
        output_type = CUDA_R_32I;
        mixed_dtype_mode = true;
        pytorch_output_type = out_dtype;
    }
    else
    {
        TORCH_CHECK(false, "Setting out_dtype is only supported for int8 input and fp16 output.");
    }
  }

  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = (compressed_A.numel() * 16 / compression_factor  ) / k;

  //initialize sparse descriptor
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      m,
      k,
      k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));

  // initalize dense input descriptor
  cusparseLtMatDescriptor_t dense_input_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &dense_input_descriptor,
      (dense_B.is_contiguous()) ? k : n,
      (dense_B.is_contiguous()) ? n : k,
      (dense_B.is_contiguous()) ? n : k,
      16,
      input_type,
      CUSPARSE_ORDER_ROW));

  // create result tensor
  at::Tensor res;
  if (mixed_dtype_mode)
  {
      res = (transpose_result) ? at::empty({n, m}, c10::TensorOptions().dtype(pytorch_output_type).device(dense_B.device()))
                               : at::empty({m, n}, c10::TensorOptions().dtype(pytorch_output_type).device(dense_B.device()));
  }
  else
  {
      res = (transpose_result) ? dense_B.new_empty({n, m})
                               : dense_B.new_empty({m, n});
  }


  cusparseLtMatDescriptor_t res_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      (transpose_result) ? m: n,
      16,
      output_type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // intialize matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &res_descriptor,
      &res_descriptor,
      compute_type));

  // set bias pointer for matmut, need to assign to get location
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias)));
  }

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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
      // jank because of the way we want this to be an array of streams
      &stream,
      1));


  //destroy descriptors
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  // destroy plan
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));

  return res;
}

} // namespace at::native

#else // No cuSPARSELt support, throw error if these functions are called.

namespace at::native {

at::Tensor _cslt_compress(const Tensor& sparse_input){
    TORCH_CHECK(false, "cuSPARSELT not supported on your machine.");
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<c10::ScalarType> out_dtype,
    bool transpose_result)
{
    TORCH_CHECK(false, "cuSPARSELT not supported on your machine.");
}

} // namespace at::native

#endif
