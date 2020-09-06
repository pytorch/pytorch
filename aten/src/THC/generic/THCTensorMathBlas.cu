#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathBlas.cu"
#else

#include <ATen/cuda/CUDAContext.h>
#include <ATen/NamedTensorUtils.h>

#define ERROR_ONLY_FP_TYPES(func) \
  THError("%s for CUDA tensors only supports floating-point types. Try converting the tensors with .float()", func);

__global__ void createBatchGemmBuffer3(const scalar_t** buffer1, const scalar_t ** buffer2, const scalar_t ** buffer3, scalar_t* data1,
                                       scalar_t * data2, scalar_t * data3, int64_t stride1, int64_t stride2, int64_t stride3, int64_t num_batches) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_batches) {
    buffer1[idx] = data1 + idx * stride1;
    buffer2[idx] = data2 + idx * stride2;
    buffer3[idx] = data3 + idx * stride3;
  }
}

void THCTensor_(baddbmm)(THCState *state, THCTensor *result, THCTensor *t,
                         THCTensor *batch1, THCTensor *batch2,
                         scalar_t beta, scalar_t alpha) {
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_BFLOAT16)
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, result, t, batch1, batch2));
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, t) == 3, 4, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, batch2) == 3, 7, "expected 3D tensor");
  THArgCheck(THCTensor_(size)(state, t, 0) == THCTensor_(size)(state, batch1, 0), 6,
             "equal number of batches expected");
  THArgCheck(THCTensor_(size)(state, t, 0) == THCTensor_(size)(state, batch2, 0), 7,
             "equal number of batches expected");
  auto maybe_outnames = at::namedinference::compute_baddbmm_outnames(result, batch1, batch2, t);
  {
    at::NoNamesGuard guard;
  THArgCheck(THCTensor_(size)(state, t, 1) == THCTensor_(size)(state, batch1, 1), 6,
             "wrong matrix size");
  THArgCheck(THCTensor_(size)(state, t, 2) == THCTensor_(size)(state, batch2, 2), 7,
             "wrong matrix size");
  THArgCheck(THCTensor_(size)(state, batch1, 2) == THCTensor_(size)(state, batch2, 1), 6,
             "wrong matrix size");

  if (t != result) {
    THCTensor_(resizeAs)(state, result, t);
    if (ScalarConvert<scalar_t, double>::to(beta) != 0.0) {
      THCTensor_(copy)(state, result, t);
    }
  }

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  int64_t lda, ldb, ldc;
  THCTensor *result_, *batch1_, *batch2_;
  if (result->stride(1) == 1 &&
   (result->size(2) == 1 || result->stride(2) >= std::max<int64_t>(1, result->size(1))))
  {
    transpose_result = false;
    result_ = result;
    ldc = result_->stride(2);
  }
  else if (result->stride(2) == 1 &&
   (result->size(1) == 1 || result->stride(1) >= std::max<int64_t>(1, result->size(2))))
  {
    transpose_result = true;

    THCTensor *swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result_ = result;
    ldc = result_->stride(1);
  }
  else
  {
    transpose_result = false;

    THCTensor *transp_r_ = THCTensor_(newTranspose)(state, result, 1, 2);
    result_ = THCTensor_(newClone)(state, transp_r_);
    THCTensor_(free)(state, transp_r_);
    THCTensor_(transpose)(state, result_, NULL, 1, 2);

    ldc = result_->stride(2);
  }

  const int64_t m = result->size(transpose_result ? 2 : 1);
  const int64_t n = result->size(transpose_result ? 1 : 2);
  const int64_t k = batch1->size(transpose_result ? 1 : 2);

  if (batch1->stride(transpose_result ? 2 : 1) == 1 &&
   batch1->stride(transpose_result ? 1 : 2) >= std::max<int64_t>(1, m))
  {
    transpose_batch1 = 'n';
    batch1_ = batch1;
    lda = batch1_->stride(transpose_result ? 1 : 2);
  }
  else if (batch1->stride(transpose_result ? 1 : 2) == 1 &&
   batch1->stride(transpose_result ? 2 : 1) >= std::max<int64_t>(1, k))
  {
    transpose_batch1 = 't';
    batch1_ = batch1;
    lda = batch1_->stride(transpose_result ? 2 : 1);
  }
  else
  {
    transpose_batch1 = transpose_result ? 'n' : 't';
    // batch1_ is later freed if batch1_ != batch1
    if (THCTensor_(isContiguous)(state, batch1)) {
      batch1_ = batch1;
    } else {
      batch1_ = THCTensor_(newContiguous)(state, batch1);
    }
    lda = batch1_->stride(1);
  }

  if (batch2->stride(transpose_result ? 2 : 1) == 1 &&
   batch2->stride(transpose_result ? 1 : 2) >= std::max<int64_t>(1, k))
  {
    transpose_batch2 = 'n';
    batch2_ = batch2;
    ldb = batch2_->stride(transpose_result ? 1 : 2);
  }
  else if (batch2->stride(transpose_result ? 1 : 2) == 1 &&
   batch2->stride(transpose_result ? 2 : 1) >= std::max<int64_t>(1, n))
  {
    transpose_batch2 = 't';
    batch2_ = batch2;
    ldb = batch2_->stride(transpose_result ? 2 : 1);
  }
  else
  {
    transpose_batch2 = transpose_result ? 'n' : 't';
    // batch2_ is later freed if batch2_ != batch2
    if (THCTensor_(isContiguous)(state, batch2)) {
      batch2_ = batch2;
    } else {
      batch2_ = THCTensor_(newContiguous)(state, batch2);
    }
    ldb = batch2_->stride(1);
  }
  int64_t num_batches = result_->size(0);

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  // Compute pointers to matrices in each batch.
#if CUDA_VERSION < 8000 && !defined __HIP_PLATFORM_HCC__
  size_t matrices_size = num_batches * sizeof(scalar_t*);

//   Copy pointers to device.
  auto d_matrices1 = static_cast<const scalar_t**>(THCudaMalloc(state, matrices_size));
  auto d_matrices2 = static_cast<const scalar_t**>(THCudaMalloc(state, matrices_size));
  auto d_result_matrices = static_cast<scalar_t**>(THCudaMalloc(state, matrices_size));

  const int64_t block = 512;
  const int64_t grid = (num_batches + block - 1) / block;

  createBatchGemmBuffer3<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
    d_matrices1, d_matrices2, (const scalar_t**)d_result_matrices, THCTensor_(data)(state, batch1_),
    THCTensor_(data)(state, batch2_), THCTensor_(data)(state, result_),
    batch1_->stride(0), batch2_->stride(0), result_->stride(0), num_batches);

#ifdef THC_REAL_IS_FLOAT
  THCudaBlas_SgemmBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      d_matrices1, lda,
      d_matrices2, ldb,
      beta,
      d_result_matrices, ldc,
      num_batches);
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_DgemmBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      d_matrices1, lda,
      d_matrices2, ldb,
      beta,
      d_result_matrices, ldc,
      num_batches);
#endif //THC_REAL

  THCudaFree(state, d_matrices1);
  THCudaFree(state, d_matrices2);
  THCudaFree(state, d_result_matrices);

#else
#ifdef THC_REAL_IS_FLOAT
  THCudaBlas_SgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride(0),
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride(0),
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride(0),
      num_batches);
#elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_DgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride(0),
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride(0),
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride(0),
      num_batches);
#endif //THC_REAL
#endif //CUDA_VERSION

#elif defined(THC_REAL_IS_HALF)

#if CUDA_VERSION < 9010 && !defined(__HIP_PLATFORM_HCC__)
  // Currently no HgemmBatched in Cublas
  for (int64_t i = 0; i < num_batches; ++i) {
    THCudaBlas_Hgemm(
        state,
        transpose_batch1,
        transpose_batch2,
        result_->size(transpose_result ? 2 : 1),
        result_->size(transpose_result ? 1 : 2),
        batch1_->size(transpose_result ? 1 : 2),
        alpha,
        THCTensor_(data)(state, batch1_) + i * batch1_->stride(0), lda,
        THCTensor_(data)(state, batch2_) + i * batch2_->stride(0), ldb,
        beta,
        THCTensor_(data)(state, result_) + i * result_->stride(0), ldc);
  }
#else
#ifndef __HIP_PLATFORM_HCC__
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5){
#endif

  THCudaBlas_HgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride(0),
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride(0),
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride(0),
      num_batches);
#ifndef __HIP_PLATFORM_HCC__
   } else {
      for (int64_t i = 0; i < num_batches; ++i) {
        THCudaBlas_Hgemm(
        state,
        transpose_batch1,
        transpose_batch2,
        result_->size(transpose_result ? 2 : 1),
        result_->size(transpose_result ? 1 : 2),
        batch1_->size(transpose_result ? 1 : 2),
        alpha,
        THCTensor_(data)(state, batch1_) + i * batch1_->stride(0), lda,
        THCTensor_(data)(state, batch2_) + i * batch2_->stride(0), ldb,
        beta,
        THCTensor_(data)(state, result_) + i * result_->stride(0), ldc);
      }
   }
#endif
#endif //CUDA_VERSION

#elif defined(THC_REAL_IS_BFLOAT16)
#if defined(__HIP_PLATFORM_HCC__)
  THCudaBlas_BgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size(transpose_result ? 2 : 1),
      result_->size(transpose_result ? 1 : 2),
      batch1_->size(transpose_result ? 1 : 2),
      alpha,
      THCTensor_(data)(state, batch1_), lda, batch1_->stride(0),
      THCTensor_(data)(state, batch2_), ldb, batch2_->stride(0),
      beta,
      THCTensor_(data)(state, result_), ldc, result_->stride(0),
      num_batches);
#endif // __HIP_PLATFORM_HCC__
#endif

  if (batch1_ != batch1) {
    THCTensor_(free)(state, batch1_);
  }

  if (batch2_ != batch2) {
    THCTensor_(free)(state, batch2_);
  }

  if (result_ != result) {
    THCTensor_(freeCopyTo)(state, result_, result);
  }

#if defined(THC_REAL_IS_BFLOAT16) && !defined(__HIP_PLATFORM_HCC__)
  // To avoid "variable was set but never used" warning
  [&transpose_batch1, &transpose_batch2, &lda, &ldb, &ldc]{}();
  TORCH_CHECK(false, "BgemmStridedBatched is not supported with at::BFloat16 type");
#endif
  }
#if !defined(THC_REAL_IS_BFLOAT16) || defined(__HIP_PLATFORM_HCC__)
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
#endif

#else
  ERROR_ONLY_FP_TYPES("baddbmm");
#endif
}

#endif
