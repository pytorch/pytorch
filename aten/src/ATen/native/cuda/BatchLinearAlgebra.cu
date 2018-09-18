#include "ATen/Context.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/PinnedMemoryAllocator.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/native/cuda/MiscUtils.h"

#include "THC.h" // for USE_MAGMA

#ifdef USE_MAGMA
#include <magma.h>
#include <magma_types.h>
#endif

namespace at {
namespace native {

#ifdef USE_MAGMA
template<class scalar_t>
void magmaGesvBatched(
    magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, scalar_t** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, magma_queue_t queue) {
  AT_ERROR("gesv only takes float or double Tensors");
}

template<class scalar_t>
void magmaGetrfBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    magma_queue_t queue) {
  AT_ERROR("getrf only takes float or double Tensors");
}

template<class scalar_t>
void magmaGetriBatched(
    magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, scalar_t** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, magma_queue_t queue) {
  AT_ERROR("getri only takes float or double Tensors");
}

template<class scalar_t>
void magmaPotrfBatched(
    magma_uplo_t uplo, magma_int_t n, scalar_t **dA_array, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue) {
  AT_ERROR("potrf only takes float or double Tensors");
}

template<>
void magmaGesvBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, double** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, magma_queue_t queue) {
  magma_dgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batch_count, queue);
}

template<>
void magmaGesvBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, float** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, magma_queue_t queue) {
  magma_sgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batch_count, queue);
}

template<>
void magmaGetrfBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    magma_queue_t queue) {
    magma_dgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, queue);
}

template<>
void magmaGetrfBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    magma_queue_t queue) {
    magma_sgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, queue);
}

template<>
void magmaGetriBatched<double>(
    magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, double** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, magma_queue_t queue) {
    magma_dgetri_outofplace_batched(n, dA_array, ldda, ipiv_array, dinvA_array, lddia, info_array, batchsize, queue);
}

template<>
void magmaGetriBatched<float>(
    magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, float** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, magma_queue_t queue) {
    magma_sgetri_outofplace_batched(n, dA_array, ldda, ipiv_array, dinvA_array, lddia, info_array, batchsize, queue);
}

template<>
void magmaPotrfBatched<double>(
    magma_uplo_t uplo, magma_int_t n, double **dA_array, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue) {
  magma_dpotrf_batched(uplo, n, dA_array, ldda, info_array, batchCount, queue);
}

template<>
void magmaPotrfBatched<float>(
    magma_uplo_t uplo, magma_int_t n, float **dA_array, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue) {
  magma_spotrf_batched(uplo, n, dA_array, ldda, info_array, batchCount, queue);
}
#endif

#define ALLOCATE_ARRAY(name, type, size, dummy_tensor) \
  auto storage_##name = pin_memory<type>(size, dummy_tensor); \
  name = static_cast<type*>(storage_##name.data());

template <typename scalar_t>
static void apply_gesv(Tensor& b, Tensor& A, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("gesv: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  magma_int_t* info_array;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** A_array;
  scalar_t** b_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, b);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, b);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, b);
  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, b);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size, b);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  magmaGesvBatched<scalar_t>(
      n, nrhs, A_array, n, ipiv_array, b_array, n,
      info_array, batch_size, createMagmaQueue(b));

  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

LINALG_HELPER_2_ARGS(gesv, self, A, cuda)

template <typename scalar_t>
static void apply_inverse(Tensor &self, Tensor &self_inv, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("inverse: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto self_data = self.data<scalar_t>();
  auto self_mat_stride = matrixStride(self);
  auto self_inv_data = self_inv.data<scalar_t>();
  auto self_inv_mat_stride = matrixStride(self_inv);

  magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");

  magma_int_t* info_array;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** self_array;
  scalar_t** self_inv_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, self);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, self);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, self);
  ALLOCATE_ARRAY(self_array, scalar_t*, batch_size, self);
  ALLOCATE_ARRAY(self_inv_array, scalar_t*, batch_size, self_inv);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    self_array[i] = &self_data[i * self_mat_stride];
    self_inv_array[i] = &self_inv_data[i * self_inv_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  magma_queue_t inverse_magma_queue = createMagmaQueue(self);

  magmaGetrfBatched<scalar_t>(
    n, n, self_array, n, ipiv_array, info_array,
    batch_size, inverse_magma_queue);

  magmaGetriBatched<scalar_t>(
    n, self_array, n, ipiv_array, self_inv_array,
    n, info_array, batch_size, inverse_magma_queue);

  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

// Because this is out-of-place inverse, the predefined macros will
// not work
Tensor _inverse_helper_cuda(const Tensor& self) {
  std::vector<int64_t> infos(batchCount(self), 0);
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto self_inv_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "inverse", [&]{
    apply_inverse<scalar_t>(
      self_working_copy, self_inv_working_copy, infos);
  });
  batchCheckErrors(infos, "inverse");
  return self_inv_working_copy;
}

template <typename scalar_t>
static void apply_potrf(Tensor& A, bool upper) {
#ifndef USE_MAGMA
AT_ERROR("potrf: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto A_data = A.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);

  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");

  magma_int_t* info_array;
  scalar_t** A_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, A);
  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, A);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
  }

  magmaPotrfBatched<scalar_t>(upper ? MagmaUpper : MagmaLower, n, A_array, n,
			      info_array, batch_size, createMagmaQueue(A));

  for (int64_t i = 0; i < batch_size; i++) {
    AT_CHECK(info_array[i] == 0, "potrf: the leading minor of order ", info_array[i], " of argument ", i, " is not positive definite");
  }
#endif
}

Tensor potrf_cuda(const Tensor& self, bool upper) {
  Tensor result;
  if (upper) { // work around magma not doing upper
    result = cloneBatchedColumnMajor(self.transpose(-1, -2));
  } else {
    result = cloneBatchedColumnMajor(self);
  }
  AT_DISPATCH_FLOATING_TYPES(result.type(), "potrf", [&]{
      apply_potrf<scalar_t>( result, false);
  });
  if (upper) {
    result.transpose_(-1, -2).triu_();
  } else {
    result.tril_();
  }
  return result;
}


template <typename T, bool upper>
struct BatchTensorTriOp {
  BatchTensorTriOp(T *start_, int64_t stride_batch_, int64_t stride_row_, int64_t stride_col_, int64_t k_)
    : start(start_), stride_row(stride_row_), stride_col(stride_col_), k(k_), stride_batch(stride_batch_),
      stride_max(stride_row_ > stride_col_? stride_row_ : stride_col_), stride_min(stride_row_ < stride_col_? stride_row_ : stride_col_) {}

  __device__ __forceinline__ int mask(T *result) {
    ptrdiff_t n = result - start;
    if (stride_batch > stride_max) {
      n = n % stride_batch;
    } else if (stride_batch > stride_min) {
      n = n - n % stride_max + n % stride_batch; // eliminate batch part
    } // if stride_batch < stride min, the divisions below will eliminate batch
    int64_t row, col;
    if (stride_row > stride_col)
    {
      row = (int64_t) (n / stride_row);
      col = (int64_t) ((n % stride_row) / stride_col);
    }
    else
    {
      row = (int64_t) ((n % stride_col) / stride_row);
      col = (int64_t) (n / stride_col);
    }

    return upper ? (col - row >= k) : (col - row <= k);
  }

  __device__ __forceinline__ void operator()(T& result, T& self) {
    result = mask(&result) ? self : scalar_cast<T>(0);
  }

  __device__ __forceinline__ void operator()(T& v) {
    if (!mask(&v))
      v = scalar_cast<T>(0);
  }

  const T *start;
  const int64_t stride_row, stride_col, k, stride_batch, stride_max, stride_min;
};

template <typename scalar_t, bool inplace, bool upper>
void apply_triu_tril(Tensor& result, const Tensor& self, int64_t k) {
  auto n = self.size(-2);
  auto m = self.size(-1);
  auto self_batched = self.view({-1, n, m});
  auto result_batched = result.view({-1, n, m});
  auto batch_size = self_batched.size(0);
  AT_CHECK(result_batched.size(0) == batch_size, "matrix sizes don't match");

  if (! inplace) {
    at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t> (result_batched, self_batched,
       BatchTensorTriOp<scalar_t, upper>(result_batched.data<scalar_t>(), result_batched.stride(0), result_batched.stride(1), result_batched.stride(2), k));
  } else {
    // it would be nicer to use CUDA_tensor_apply1 if it existed...
    at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t> (result_batched, result_batched,
       BatchTensorTriOp<scalar_t, upper>(result_batched.data<scalar_t>(), result_batched.stride(0), result_batched.stride(1), result_batched.stride(2), k));
  }
}

Tensor& tril_cuda_(Tensor &self, int64_t k) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "tril", [&] {
      apply_triu_tril<scalar_t, true, false>(self, self, k);
    });
  return self;
}

Tensor& tril_cuda_out(Tensor &result, const Tensor& self, int64_t k) {
  result.resize_as_(self);
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "tril", [&] {
      apply_triu_tril<scalar_t, false, false>(result, self, k);
    });
  return result;
}

Tensor& triu_cuda_(Tensor &self, int64_t k) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "triu", [&] {
      apply_triu_tril<scalar_t, true, true>(self, self, k);
    });
  return self;
}

Tensor& triu_cuda_out(Tensor &result, const Tensor& self, int64_t k) {
  result.resize_as_(self);
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "triu", [&] {
      apply_triu_tril<scalar_t, false, true>(result, self, k);
    });
  return result;
}

}}  // namespace at::native

#undef ALLOCATE_ARRAY
