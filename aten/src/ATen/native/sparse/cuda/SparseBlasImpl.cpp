#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDASparseDescriptors.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/sparse/cuda/SparseBlasImpl.h>
#include <ATen/native/sparse/cuda/SparseBlasLegacy.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/MaybeOwned.h>

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace cuda {

namespace {

c10::MaybeOwned<Tensor> inline prepare_dense_matrix_for_cusparse(
    const Tensor& tensor) {
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
  // CUDA < 11.0 doesn't support row-major layout, return column-major in this case
  if (is_blas_compatible_column_major_order(tensor)) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(cloneBatchedColumnMajor(tensor));
  }
#else
  if (is_blas_compatible_row_major_order(tensor) ||
      is_blas_compatible_column_major_order(tensor)) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
#endif
}

// This function is used for old CUDA Toolkit versions that doesn't support new cuSPARSE Generic API
void addmm_out_legacy(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.is_sparse_csr());
  auto nnz = mat1._nnz();
  auto m = mat1.size(0);
  auto k = mat1.size(1);
  auto n = mat2.size(1);
  auto crow_indices = mat1.crow_indices().to(kInt);
  auto col_indices = mat1.col_indices().to(kInt);
  auto values = mat1.values();
  at::native::s_addmm_out_csr_sparse_dense_cuda_worker(nnz, m, n, k, result, beta, result, alpha, crow_indices, col_indices, values, mat2);
}

c10::MaybeOwned<Tensor> inline prepare_dense_vector_for_cusparse(
    const Tensor& tensor) {
  if (tensor.is_non_overlapping_and_dense()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

void inline indices_to_32_bit_inplace(const Tensor& input) {
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())->set_member_tensors(
      input.crow_indices().to(kInt),
      input.col_indices().to(kInt),
      input.values(),
      input.sizes());
}

void inline col_indices_and_values_resize_(const Tensor& input, int64_t nnz) {
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())->set_member_tensors(
      input.crow_indices(),
      input.col_indices().resize_({nnz}),
      input.values().resize_({nnz}),
      input.sizes());
}

void spmm(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_CUSPARSE_GENERIC_API()
  addmm_out_legacy(mat1, mat2, beta, alpha, result);
#else
  c10::MaybeOwned<Tensor> result_ = prepare_dense_matrix_for_cusparse(result);
  c10::MaybeOwned<Tensor> mat2_ = prepare_dense_matrix_for_cusparse(mat2);

  // Here subscript "c" stands for column-major, substript "r" stands for
  // row-major order Both orders are supported by cuSPARSE. For mixed input we
  // need to cast 'mat2' to order of 'result'. We compute
  // result = mat1 @ op(mat2) + result.
  // If order of 'mat2' and 'result' matches, the op is
  // identity; op(mat2) == mat2. If 'result' is column-major and 'mat2' is
  // row-major we pass 'mat2' as column-major and compute
  // result_c = mat1 @ transpose(mat2_c) + result_c; mat2_r==transpose(mat2_c)
  // if 'result' is row-major and 'mat2' is column-major we pass 'mat2'
  // as row-major and compute
  // result_r = mat1 @ transpose(mat2_r) + result_r; mat2_c==transpose(mat2_r)
  IntArrayRef result_strides = result_->strides();
  IntArrayRef mat2_strides = mat2_->strides();
  auto ndim = result_->dim();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim == 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.dim() == 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat2.dim() == 2);
  bool is_result_row_major = (result_strides[ndim - 1] == 1);
  bool is_mat2_row_major = (mat2_strides[ndim - 1] == 1);
  bool transpose_B = (is_result_row_major ^ is_mat2_row_major);

  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = transpose_B ? CUSPARSE_OPERATION_TRANSPOSE
                                        : CUSPARSE_OPERATION_NON_TRANSPOSE;

  // CUDA < 11.0 doesn't support 64-bit indices and doesn't raise an error about this
  // silently returning incorrect results
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
  auto mat1_32 = at::native::_sparse_csr_tensor_unsafe(
      mat1.crow_indices().to(kInt),
      mat1.col_indices().to(kInt),
      mat1.values(),
      mat1.sizes(),
      mat1.scalar_type(),
      mat1.layout(),
      mat1.device());
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(mat1_32);
  auto algorithm = CUSPARSE_MM_ALG_DEFAULT;
#else
  // TODO: update this to support COO sparse layout
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(mat1);
  auto algorithm = CUSPARSE_SPMM_CSR_ALG2;
#endif

  auto descB = at::cuda::sparse::CuSparseDnMatDescriptor(
      transpose_B ? mat2_->transpose(-2, -1) : *mat2_);
  auto descC = at::cuda::sparse::CuSparseDnMatDescriptor(*result_);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      result.scalar_type(),
      "addmm_out_sparse_csr_impl_cuda",
      [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();

        size_t buffer_size;
        TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            algorithm,
            &buffer_size // output
            ));

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto work_data = allocator.allocate(buffer_size);

        TORCH_CUDASPARSE_CHECK(cusparseSpMM(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            algorithm,
            work_data.get()));
      });

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif // !AT_USE_CUSPARSE_GENERIC_API()
}

void spgemm(
    const at::sparse_csr::SparseCsrTensor& A,
    const at::sparse_csr::SparseCsrTensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const at::sparse_csr::SparseCsrTensor& C) {
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
  TORCH_CHECK(
      false,
      "Calling addmm with sparse GPU tensors requires compiling ",
      "PyTorch with CUDA 11+. ",
      "Please use PyTorch built with newer CUDA version.");
#else
  IntArrayRef A_sizes = A.sizes();
  auto ndim = A.dim();
  auto m = A_sizes[ndim - 2];
  auto k = A_sizes[ndim - 1];

  IntArrayRef B_sizes = B.sizes();
  auto n = B_sizes[ndim - 1];

  // Only 32-bit indices are supported
  auto A_32 = at::native::_sparse_csr_tensor_unsafe(A.crow_indices().to(kInt), A.col_indices().to(kInt), A.values(), A.sizes(), A.scalar_type(), A.layout(), A.device());
  auto B_32 = at::native::_sparse_csr_tensor_unsafe(B.crow_indices().to(kInt), B.col_indices().to(kInt), B.values(), B.sizes(), B.scalar_type(), B.layout(), B.device());

  // Modify C tensor in-place to swap indices tensors with 32-bit variants
  indices_to_32_bit_inplace(C);

  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(A_32);
  auto descB = at::cuda::sparse::CuSparseSpMatCsrDescriptor(B_32);
  auto descC = at::cuda::sparse::CuSparseSpMatCsrDescriptor(C);

  auto spgemm_desc = at::cuda::sparse::CuSparseSpGEMMDescriptor();
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      C.scalar_type(),
      "addmm_out_sparse_csr_impl_cuda",
      [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        auto compute_type = at::cuda::getCudaDataType<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();

        // It's required to call workEstimation twice
        size_t buffer_size1 = 0;
        TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_workEstimation(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemm_desc.descriptor(),
            &buffer_size1,
            nullptr));

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto buffer1 = allocator.allocate(buffer_size1);

        TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_workEstimation(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemm_desc.descriptor(),
            &buffer_size1,
            buffer1.get()));

        // It's required to call compute twice
        size_t buffer_size2 = 0;
        TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_compute(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemm_desc.descriptor(),
            &buffer_size2,
            nullptr));

        auto buffer2 = allocator.allocate(buffer_size2);

        TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_compute(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemm_desc.descriptor(),
            &buffer_size2,
            buffer2.get()));

        // Get how many specified elements are there in C
        int64_t C_num_rows, C_num_cols, C_nnz;
        std::tie(C_num_rows, C_num_cols, C_nnz) = descC.get_size();

        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(C_num_rows == m);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(C_num_cols == n);

        // Resize result using nnz information from cusolver
        col_indices_and_values_resize_(C, C_nnz);

        // Update matC with the new pointers
        descC.set_tensor(C);

        // Copy the data into C
        TORCH_CUDASPARSE_CHECK(cusparseSpGEMM_copy(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemm_desc.descriptor()));
      });
#endif
}

} // anonymous namespace

void addmm_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  if (mat2.layout() == kStrided && result.layout() == kStrided) {
    return spmm(mat1, mat2, beta, alpha, result);
  } else if (mat2.is_sparse_csr() && result.is_sparse_csr()) {
    return spgemm(mat1, mat2, beta, alpha, result);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Received unexpected tensor layouts as input.");
  }
}

/*
  Computes a sparse matrix-dense vector product defined as
  y <- alpha*op(A)*x + beta*y

  Args:
  * `mat` - Tensor storing sparse m x n matrix A.
  * `vec` - Tensor storing dense vector x of size n.
  * `result` - [in] Tensor storing dense vector y of size m.
               [out] result of the operation.
*/
void addmv_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_CUSPARSE_GENERIC_API()
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse GPU tensor requires compiling ",
      "PyTorch with CUDA 10.2+ (CUDA 11+ on Windows). ",
      "Please use PyTorch built with newer CUDA version.");
#else
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;

  c10::MaybeOwned<Tensor> result_ = prepare_dense_vector_for_cusparse(result);
  c10::MaybeOwned<Tensor> vec_ = prepare_dense_vector_for_cusparse(vec);

  // TODO: update this to support COO sparse layout
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(mat);
  auto descX = at::cuda::sparse::CuSparseDnVecDescriptor(*vec_);
  auto descY = at::cuda::sparse::CuSparseDnVecDescriptor(*result_);

  // cusparseSpMVAlg_t was updated in cuda 11.2.1 (cusparse 11.4.0)
#if CUSPARSE_VERSION >= 11400
  cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;
#else
  cusparseSpMVAlg_t alg = CUSPARSE_MV_ALG_DEFAULT;
#endif

  // There is no dispatch for kHalf and kBFloat16 types because cusparse
  // computes garbage in this case, latest checked version of cuda is 11.3
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(),
      "addmv_out_sparse_csr_cuda_impl",
      [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();

        size_t buffer_size;
        TORCH_CUDASPARSE_CHECK(cusparseSpMV_bufferSize(
            handle,
            opA,
            &alpha_,
            descA.descriptor(),
            descX.descriptor(),
            &beta_,
            descY.descriptor(),
            compute_type,
            alg,
            &buffer_size // output
            ));

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto work_data = allocator.allocate(buffer_size);

        TORCH_CUDASPARSE_CHECK(cusparseSpMV(
            handle,
            opA,
            &alpha_,
            descA.descriptor(),
            descX.descriptor(),
            &beta_,
            descY.descriptor(),
            compute_type,
            alg,
            work_data.get()));
      });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif
}

} // namespace cuda
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
