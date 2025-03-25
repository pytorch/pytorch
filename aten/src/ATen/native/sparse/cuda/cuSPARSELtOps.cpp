#include <ATen/native/sparse/cuda/cuSPARSELtOps.h>

#if AT_CUSPARSELT_ENABLED()

namespace at::native {

// Ideally we would use the same DeviceThreadHandlePool mechanism as used in
// aten/src/ATen/cuda/CuSparseHandlePool.cpp which would handle this for us.
// However, the cuSPARSELt handle signature is different from that of
// cuSPARSE/cuBLAS, so it's not possible to reuse the existing pooling
// mechanism. Instead we have to handle our handles ourselves, which is why
// these variables are thread local. Once cuSPARSELt updates their handle
// signature to be consistent with the rest of CUDA, we can switch to using
// DeviceThreadHandlePool.
thread_local cusparseLtHandle_t handle;
thread_local bool handle_initialized = false;

at::Tensor _cslt_compress(const Tensor& sparse_input) {
  if (!handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
    handle_initialized = true;
  }
  // create sparse descriptor, dtype
  cusparseLtMatDescriptor_t sparse_input_descriptor;
  cudaDataType type;
  auto compression_factor = 9;

  switch (sparse_input.scalar_type()) {
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
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602
    case at::ScalarType::Float8_e4m3fn:
      type = CUDA_R_8F_E4M3;
      compression_factor = 10;
      break;
#endif
    default:
      TORCH_CHECK(false, "Unsupported dtype for cuSPARSELt compressed matrix");
      break;
  }

  // create a new compressed tensor with the same dtype as
  auto compressed_tensor =
      sparse_input.new_empty(sparse_input.numel() * compression_factor / 16);

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

std::tuple<at::Tensor, int64_t, int64_t, bool, int64_t> _cslt_sparse_mm_impl(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int alg_id,
    int split_k,
    bool split_k_one_kernel,
    bool search_alg_id) {
  if (!handle_initialized) {
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
    handle_initialized = true;
  }
  // cupsarselt constructs
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;

  int tensor_alpha_mode = 0;
  float alpha = 1.0;
  float beta = 0.0;
  cudaDataType input_type;
  cudaDataType output_type;
  cudaDataType C_type;
  cusparseComputeType compute_type;
  auto compression_factor = 9;

  switch (compressed_A.scalar_type()) {
    case at::ScalarType::Char:
      input_type = CUDA_R_8I;
      output_type = CUDA_R_8I;
      C_type = CUDA_R_8I;
      compute_type = CUSPARSE_COMPUTE_32I;
      compression_factor = 10;
      break;

// cuSPARSELt v0.5.2 onwards changes CUSPARSE_COMPUTE_TF32, CUSPARSE_COMPUT_16F
// to CUSPARSE_COMPUTE_32F
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 502
    case at::ScalarType::Half:
      input_type = CUDA_R_16F;
      output_type = CUDA_R_16F;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    case at::ScalarType::BFloat16:
      input_type = CUDA_R_16BF;
      output_type = CUDA_R_16BF;
      C_type = CUDA_R_16BF;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
    case at::ScalarType::Float:
      input_type = CUDA_R_32F;
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
      compute_type = CUSPARSE_COMPUTE_32F;
      break;
// if cuSPARSELt >= 6.2.3, we can add Float8 support
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602
    case at::ScalarType::Float8_e4m3fn:
      input_type = CUDA_R_8F_E4M3;
      output_type = CUDA_R_8F_E4M3;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_32F;
      compression_factor = 10;
      break;
#endif
// cuSPARSELt <= v0.5.2 uses CUSPARSE_COMPUTE_TF32, CUSPARSE_COMPUTE_16F
#else
    case at::ScalarType::Half:
      input_type = CUDA_R_16F;
      output_type = CUDA_R_16F;
      C_type = CUDA_R_16F;
      compute_type = CUSPARSE_COMPUTE_16F;
      break;
    case at::ScalarType::BFloat16:
      input_type = CUDA_R_16BF;
      output_type = CUDA_R_16BF;
      C_type = CUDA_R_16BF;
      compute_type = CUSPARSE_COMPUTE_16F;
      break;
    case at::ScalarType::Float:
      input_type = CUDA_R_32F;
      output_type = CUDA_R_32F;
      C_type = CUDA_R_32F;
      compute_type = CUSPARSE_COMPUTE_TF32;
      break;
#endif
    default:
      TORCH_CHECK(
          false,
          "Unsupported dtype for cuSPARSELt compressed matrix multiplication.");
      break;
  }
  ScalarType out_dtype = dense_B.scalar_type();
  // special check for mixed dtype support for 8 bit dtypes
  // cslt 0.5.2+: int8 int8 -> {fp16, bf16, int32} support
  if (out_dtype_opt.has_value()) {
    out_dtype = out_dtype_opt.value();
    if (input_type == CUDA_R_8I) {
      switch (out_dtype) {
        case at::ScalarType::Half:
          C_type = CUDA_R_16F;
          output_type = CUDA_R_16F;
          break;
        case at::ScalarType::BFloat16:
          C_type = CUDA_R_16BF;
          output_type = CUDA_R_16BF;
          break;
        case at::ScalarType::Int:
          C_type = CUDA_R_32I;
          output_type = CUDA_R_32I;
          break;
        default:
          TORCH_CHECK(
              false,
              "Unsupported out_dtype passed, must be one of {fp16, bf16, int32} for int8 inputs");
          break;
      }
    }
// cslt 0.6.2+: fp8 fp8 -> {fp8, fp16, bf16, fp32} support
#if defined(CUSPARSELT_VERSION) && CUSPARSELT_VERSION >= 602
    else if (input_type == CUDA_R_8F_E4M3) {
      switch (out_dtype) {
        case at::ScalarType::Float8_e4m3fn:
          output_type = CUDA_R_8F_E4M3;
          C_type = CUDA_R_16F;
          break;
        case at::ScalarType::Half:
          output_type = CUDA_R_16F;
          C_type = CUDA_R_16F;
          break;
        case at::ScalarType::BFloat16:
          output_type = CUDA_R_16BF;
          C_type = CUDA_R_16BF;
          break;
        case at::ScalarType::Float:
          output_type = CUDA_R_32F;
          C_type = CUDA_R_32F;
          break;
        default:
          TORCH_CHECK(
              false,
              "Unsupported out_dtype passed, must be one of {fp16, bf16, float32} for fp8 inputs");
          break;
      }
    }
#endif
    else {
      TORCH_CHECK(
          false, "out_dtype support only available for int8/fp8 inputs");
    }
  }

  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = (compressed_A.numel() * 16 / compression_factor) / k;

  // initialize sparse descriptor
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

  // initialize dense input descriptor
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
  auto res_tensor_options =
      c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = (transpose_result) ? at::empty({n, m}, res_tensor_options)
                                      : at::empty({m, n}, res_tensor_options);

  cusparseLtMatDescriptor_t res_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      (transpose_result) ? m : n,
      16,
      output_type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // For float8, need fp16 C_descriptor, can't use FP8 for this matrix
  cusparseLtMatDescriptor_t C_descriptor;
  TORCH_CUDASPARSE_CHECK(cusparseLtDenseDescriptorInit(
      &handle,
      &C_descriptor,
      m,
      n,
      (transpose_result) ? m : n,
      16,
      C_type,
      (transpose_result) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));

  // initialize matmul
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                : CUSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &C_descriptor,
      &res_descriptor,
      compute_type));

  // set bias pointer for matmul, need to assign to get location
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        CUSPARSELT_MATMUL_BIAS_POINTER,
        &dBias,
        sizeof(dBias)));
  }

  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

  // set matmul search params
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
      &handle,
      &alg_sel,
      CUSPARSELT_MATMUL_ALG_CONFIG_ID,
      &alg_id,
      sizeof(alg_id)));

  cusparseLtSplitKMode_t splitKMode;
  int max_alg_id;
  if (split_k != 1) {
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K,
        &split_k,
        sizeof(split_k)));

    splitKMode = split_k_one_kernel ? CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL
                                    : CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS;
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgSetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K_MODE,
        &splitKMode,
        sizeof(splitKMode)));
  }

  // set tensor_alpha_mode and alpha pointer for matmul
  const auto alpha_tensor = alpha_opt.has_value() ? *alpha_opt : Tensor{};
  auto alpha_ptr = &alpha;
  if (alpha_opt.has_value()) {
    if (alpha_tensor.numel() == 1) {
      alpha = alpha_tensor.item<float>();
    } else {
      tensor_alpha_mode = 1;
      TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
          &handle,
          &matmul,
          CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING,
          &tensor_alpha_mode,
          sizeof(tensor_alpha_mode)));
      alpha_ptr = static_cast<float*>(alpha_tensor.data_ptr());
    }
  }

  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (search_alg_id) {
    // run matmul search
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulSearch(
        &handle,
        &plan,
        alpha_ptr,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        // jank because of the way we want this to be an array of streams
        &stream,
        1));

    // get matmul params used
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg_id,
        sizeof(alg_id)));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K,
        &split_k,
        sizeof(split_k)));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K_MODE,
        &splitKMode,
        sizeof(splitKMode)));

    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulAlgGetAttribute(
        &handle,
        &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
        &max_alg_id,
        sizeof(max_alg_id)));

  } else {
    // do normal matmul
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmul(
        &handle,
        &plan,
        alpha_ptr,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        // jank because of the way we want this to be an array of streams
        &stream,
        1));
  }

  // destroy descriptors
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  TORCH_CUDASPARSE_CHECK(
      cusparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_CUDASPARSE_CHECK(cusparseLtMatDescriptorDestroy(&res_descriptor));
  // destroy plan
  TORCH_CUDASPARSE_CHECK(cusparseLtMatmulPlanDestroy(&plan));

  return {
      res,
      alg_id,
      split_k,
      splitKMode == CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL,
      max_alg_id};
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int64_t alg_id,
    int64_t split_k,
    bool split_k_one_kernel) {
  auto result = _cslt_sparse_mm_impl(
      compressed_A,
      dense_B,
      bias_opt,
      alpha_opt,
      out_dtype_opt,
      transpose_result,
      (int)alg_id,
      (int)split_k,
      split_k_one_kernel,
      false);
  return std::get<0>(result);
}

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result) {
  TORCH_WARN_ONCE(
      "torch._cslt_sparse_mm_search is deprecated and will be removed in a future PyTorch release. Please use torch._C._cusparselt.mm_search instead.");
  int alg_id_int = 0;
  int split_k = 1;
  bool split_k_one_kernel = true;
  auto result = _cslt_sparse_mm_impl(
      compressed_A,
      dense_B,
      bias_opt,
      alpha_opt,
      out_dtype_opt,
      transpose_result,
      alg_id_int,
      split_k,
      split_k_one_kernel,
      true);
  return (int64_t)std::get<1>(result);
}

} // namespace at::native

#else // No cuSPARSELt support, throw error if these functions are called.

namespace at::native {

at::Tensor _cslt_compress(const Tensor& sparse_input) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result,
    int64_t alg_id,
    int64_t split_k,
    bool split_k_one_kernel) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result) {
  TORCH_CHECK(false, "cuSPARSELt not supported on your machine.");
}

} // namespace at::native

#endif
