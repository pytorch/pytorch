#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDASparseBlas.h>
#include <ATen/cuda/CUDASparseDescriptors.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/sparse/cuda/SparseBlasImpl.h>
#include <ATen/native/sparse/cuda/SparseBlasLegacy.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/empty_strided.h>
#endif

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/MaybeOwned.h>

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace cuda {

namespace {

c10::MaybeOwned<Tensor> prepare_column_major_matrix_for_cusparse(
    const Tensor& tensor) {
  if (is_blas_compatible_column_major_order(tensor)) {
    return at::native::expect_resolved_conj(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(cloneBatchedColumnMajor(tensor));
  }
}

c10::MaybeOwned<Tensor> inline prepare_dense_matrix_for_cusparse(
    const Tensor& tensor) {
#if defined(CUDA_VERSION) && CUDA_VERSION < 11000
  // CUDA < 11.0 doesn't support row-major layout, return column-major in this case
  return prepare_column_major_matrix_for_cusparse(tensor);
#else
  if (is_blas_compatible_row_major_order(tensor) ||
      is_blas_compatible_column_major_order(tensor)) {
    return at::native::expect_resolved_conj(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
#endif
}

Tensor copy_strided(const Tensor& tensor, IntArrayRef strides) {
  Tensor result = at::empty_strided(tensor.sizes(), strides, tensor.options());
  result.copy_(tensor);
  return result;
}

c10::MaybeOwned<Tensor> prepare_dense_matrix_for_cusparse(
    const Tensor& tensor,
    IntArrayRef strides) {
  if (tensor.strides().equals(strides)) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(copy_strided(tensor, strides));
  }
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
  auto mat2_ = at::native::expect_resolved_conj(mat2);
  auto result_ = at::native::expect_resolved_conj(result);
  at::native::s_addmm_out_csr_sparse_dense_cuda_worker(nnz, m, n, k, result, beta, *result_, alpha, crow_indices, col_indices, values, *mat2_);
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
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

void block_sparse_triangular_solve_vec(
    const at::sparse_csr::SparseCsrTensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
#if !AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()
  TORCH_CHECK(
      false,
      "Calling triangular solver with block sparse GPU tensors requires compiling ",
      "PyTorch with ROCm 4.5.0+. ",
      "Please use PyTorch built with newer ROCm version.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.is_sparse_csr());
  // values is expected to be a blocks of sparse matrix
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.values().dim() == 3);
  // blocks are expected to be square
  TORCH_INTERNAL_ASSERT(A.values().size(2) == A.values().size(1));
  // only block of size > 1 is supported in cuSPARSE
  TORCH_INTERNAL_ASSERT(A.values().size(-1) > 1);
  // blocks are expected to be in row- or column-major order
  TORCH_INTERNAL_ASSERT(
      A.values().is_contiguous() ||
      A.values().transpose(-2, -1).is_contiguous());

  // cuSPARSE can't work with empty sparse matrices
  if (A._nnz() == 0) {
    X.fill_(NAN);
    return;
  }

  const cusparseDirection_t block_layout = A.values().is_contiguous()
      ? CUSPARSE_DIRECTION_ROW
      : CUSPARSE_DIRECTION_COLUMN;

  c10::MaybeOwned<Tensor> X_ = prepare_dense_matrix_for_cusparse(X);
  c10::MaybeOwned<Tensor> B_ = prepare_dense_matrix_for_cusparse(B);

  auto block_size = cuda_int_cast(A.values().size(2), "block_size");
  auto nnzb = cuda_int_cast(A._nnz(), "nnzb");
  auto mb = cuda_int_cast(A.size(0), "mb") / block_size;

  auto desc = at::cuda::sparse::CuSparseMatDescriptor(upper, unitriangular);
  cusparseOperation_t opA = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                      : CUSPARSE_OPERATION_NON_TRANSPOSE;

  auto info = at::cuda::sparse::CuSparseBsrsv2Info();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      X.scalar_type(), "block_sparse_triangular_solve_vec", [&] {
        scalar_t alpha = 1;
        auto values = A.values();
        auto values_data_ptr = values.data_ptr<scalar_t>();
        auto crow_indices = A.crow_indices().to(kInt);
        auto crow_indices_data_ptr = crow_indices.data_ptr<int>();
        auto col_indices = A.col_indices().to(kInt);
        auto col_indices_data_ptr = col_indices.data_ptr<int>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();
        int buffer_size = 0;

        at::cuda::sparse::bsrsv2_bufferSize(
            handle,
            block_layout,
            opA,
            mb,
            nnzb,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            info.descriptor(),
            &buffer_size);

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto work_data = allocator.allocate(buffer_size);

        at::cuda::sparse::bsrsv2_analysis(
            handle,
            block_layout,
            opA,
            mb,
            nnzb,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            info.descriptor(),
            CUSPARSE_SOLVE_POLICY_NO_LEVEL,
            work_data.get());

        at::cuda::sparse::bsrsv2_solve(
            handle,
            block_layout,
            opA,
            mb,
            nnzb,
            &alpha,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            info.descriptor(),
            B_->data_ptr<scalar_t>(),
            X_->data_ptr<scalar_t>(),
            CUSPARSE_SOLVE_POLICY_NO_LEVEL,
            work_data.get());
      });
  if (!X.is_same(*X_)) {
    X.copy_(*X_);
  }
#endif
}

void block_sparse_triangular_solve_mat(
    const at::sparse_csr::SparseCsrTensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
#if !AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()
  TORCH_CHECK(
      false,
      "Calling triangular solver with block sparse GPU tensors requires compiling ",
      "PyTorch with ROCm 4.5.0+. ",
      "Please use PyTorch built with newer ROCm version.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.is_sparse_csr());
  // values is expected to be a blocks of sparse matrix
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.values().dim() == 3);
  // blocks are expected to be square
  TORCH_INTERNAL_ASSERT(A.values().size(2) == A.values().size(1));
  // only block of size > 1 is supported in cuSPARSE
  TORCH_INTERNAL_ASSERT(A.values().size(-1) > 1);
  // blocks are expected to be in row- or column-major order
  TORCH_INTERNAL_ASSERT(
      A.values().is_contiguous() ||
      A.values().transpose(-2, -1).is_contiguous());

  // cuSPARSE can't work with empty sparse matrices
  if (A._nnz() == 0) {
    X.fill_(NAN);
    return;
  }

  const cusparseDirection_t block_layout = A.values().is_contiguous()
      ? CUSPARSE_DIRECTION_ROW
      : CUSPARSE_DIRECTION_COLUMN;

  c10::MaybeOwned<Tensor> X_ = prepare_column_major_matrix_for_cusparse(X);
  c10::MaybeOwned<Tensor> B_ = prepare_column_major_matrix_for_cusparse(B);

  int ldb = cuda_int_cast(B_->stride(-1), "ldb");
  int ldx = cuda_int_cast(X_->stride(-1), "ldx");

  cusparseOperation_t opX = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opA = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                      : CUSPARSE_OPERATION_NON_TRANSPOSE;

  auto block_size = cuda_int_cast(A.values().size(2), "block_size");
  auto nnzb = cuda_int_cast(A._nnz(), "nnzb");
  auto mb = cuda_int_cast(A.size(0), "mb") / block_size;
  auto n = cuda_int_cast(B.size(-1), "n");

  auto desc = at::cuda::sparse::CuSparseMatDescriptor(upper, unitriangular);
  auto info = at::cuda::sparse::CuSparseBsrsm2Info();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      X.scalar_type(), "block_sparse_triangular_solve_vec", [&] {
        scalar_t alpha = 1;
        auto values = A.values();
        auto values_data_ptr = values.data_ptr<scalar_t>();
        auto crow_indices = A.crow_indices().to(kInt);
        auto crow_indices_data_ptr = crow_indices.data_ptr<int>();
        auto col_indices = A.col_indices().to(kInt);
        auto col_indices_data_ptr = col_indices.data_ptr<int>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();
        int buffer_size = 0;

        at::cuda::sparse::bsrsm2_bufferSize(
            handle,
            block_layout,
            opA,
            opX,
            mb,
            n,
            nnzb,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            info.descriptor(),
            &buffer_size);

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto work_data = allocator.allocate(buffer_size);

        at::cuda::sparse::bsrsm2_analysis(
            handle,
            block_layout,
            opA,
            opX,
            mb,
            n,
            nnzb,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            info.descriptor(),
            CUSPARSE_SOLVE_POLICY_NO_LEVEL,
            work_data.get());

        at::cuda::sparse::bsrsm2_solve(
            handle,
            block_layout,
            opA,
            opX,
            mb,
            n,
            nnzb,
            &alpha,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            info.descriptor(),
            B_->data_ptr<scalar_t>(),
            ldb,
            X_->data_ptr<scalar_t>(),
            ldx,
            CUSPARSE_SOLVE_POLICY_NO_LEVEL,
            work_data.get());
      });
  if (!X.is_same(*X_)) {
    X.copy_(*X_);
  }
#endif
}

void block_sparse_mv(
    const at::sparse_csr::SparseCsrTensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat.is_sparse_csr());
  // values is expected to be a blocks of sparse matrix
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat.values().dim() == 3);
  // blocks are expected to be square
  TORCH_INTERNAL_ASSERT(mat.values().size(2) == mat.values().size(1));
  // only block of size > 1 is supported in cuSPARSE
  TORCH_INTERNAL_ASSERT(mat.values().size(-1) > 1);
  // blocks are expected to be in row- or column-major order
  TORCH_INTERNAL_ASSERT(
      mat.values().is_contiguous() ||
      mat.values().transpose(-2, -1).is_contiguous());

  const cusparseDirection_t block_layout = mat.values().is_contiguous()
      ? CUSPARSE_DIRECTION_ROW
      : CUSPARSE_DIRECTION_COLUMN;

  c10::MaybeOwned<Tensor> result_ = prepare_dense_vector_for_cusparse(result);
  c10::MaybeOwned<Tensor> vec_ = prepare_dense_vector_for_cusparse(vec);

  auto block_size = cuda_int_cast(mat.values().size(2), "block_size");
  auto nnzb = cuda_int_cast(mat._nnz(), "nnzb");
  auto mb = cuda_int_cast(mat.size(0), "mb") / block_size;
  auto nb = cuda_int_cast(mat.size(1), "nb") / block_size;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "block_sparse_mv", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();
        auto desc = at::cuda::sparse::CuSparseMatDescriptor();
        auto values = mat.values();
        auto values_data_ptr = values.data_ptr<scalar_t>();
        auto crow_indices = mat.crow_indices().to(kInt);
        auto crow_indices_data_ptr = crow_indices.data_ptr<int>();
        auto col_indices = mat.col_indices().to(kInt);
        auto col_indices_data_ptr = col_indices.data_ptr<int>();
        at::cuda::sparse::bsrmv(
            handle,
            block_layout,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            mb,
            nb,
            nnzb,
            &alpha_,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            vec_->data_ptr<scalar_t>(),
            &beta_,
            result_->data_ptr<scalar_t>());
      });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

void block_sparse_mm(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.is_sparse_csr());
  // values is expected to be a blocks of sparse matrix
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.values().dim() == 3);
  // blocks are expected to be square
  TORCH_INTERNAL_ASSERT(mat1.values().size(2) == mat1.values().size(1));
  // only block of size > 1 is supported in cuSPARSE
  TORCH_INTERNAL_ASSERT(mat1.values().size(-1) > 1);
  // blocks are expected to be in row- or column-major order
  TORCH_INTERNAL_ASSERT(
      mat1.values().is_contiguous() ||
      mat1.values().transpose(-2, -1).is_contiguous());

  const cusparseDirection_t block_layout = mat1.values().is_contiguous()
      ? CUSPARSE_DIRECTION_ROW
      : CUSPARSE_DIRECTION_COLUMN;

  c10::MaybeOwned<Tensor> mat2_ = prepare_dense_matrix_for_cusparse(mat2);

  // cuSPARSE expects column-major strides for result and we can't manipulate
  // transpose flag of mat1
  c10::MaybeOwned<Tensor> result_ =
      prepare_column_major_matrix_for_cusparse(result);

  IntArrayRef result_strides = result_->strides();
  IntArrayRef mat2_strides = mat2_->strides();
  auto ndim = result_->dim();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim == 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.dim() == 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat2.dim() == 2);

  bool is_mat2_row_major = (mat2_strides[ndim - 1] == 1);
  int ldb = is_mat2_row_major ? cuda_int_cast(mat2_strides[ndim - 2], "ldb")
                              : cuda_int_cast(mat2_strides[ndim - 1], "ldb");
  int ldc = cuda_int_cast(result_strides[ndim - 1], "ldc");
  auto block_size = cuda_int_cast(mat1.values().size(2), "block_size");
  auto nnzb = cuda_int_cast(mat1._nnz(), "nnzb");
  auto mb = cuda_int_cast(mat1.size(0), "mb") / block_size;
  auto kb = cuda_int_cast(mat1.size(1), "nb") / block_size;
  auto n = cuda_int_cast(mat2.size(1), "n");

  // according to cuSPARSE documentation, opA can only be NON_TRANSPOSE
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = is_mat2_row_major
      ? CUSPARSE_OPERATION_TRANSPOSE
      : CUSPARSE_OPERATION_NON_TRANSPOSE;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "block_sparse_mm", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();
        auto desc = at::cuda::sparse::CuSparseMatDescriptor();

        auto values = mat1.values();
        auto values_data_ptr = values.data_ptr<scalar_t>();
        auto crow_indices = mat1.crow_indices().to(kInt);
        auto crow_indices_data_ptr = crow_indices.data_ptr<int>();
        auto col_indices = mat1.col_indices().to(kInt);
        auto col_indices_data_ptr = col_indices.data_ptr<int>();

        at::cuda::sparse::bsrmm(
            handle,
            block_layout,
            opA,
            opB,
            mb,
            n,
            kb,
            nnzb,
            &alpha_,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            mat2_->data_ptr<scalar_t>(),
            ldb,
            &beta_,
            result_->data_ptr<scalar_t>(),
            ldc);
      });

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

void spmm(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  if (mat1.values().dim() == 3 && mat1.values().size(-1) > 1) {
    return block_sparse_mm(mat1, mat2, beta, alpha, result);
  }
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
      transpose_B ? mat2_->mT() : *mat2_);
  auto descC = at::cuda::sparse::CuSparseDnMatDescriptor(*result_);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      result.scalar_type(),
      "spmm",
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
  // older versions of cusparse on Windows segfault for complex128 dtype
#if defined(_WIN32) && defined(CUSPARSE_VERSION) && CUSPARSE_VERSION < 11400
  TORCH_CHECK(
      !(A.scalar_type() == ScalarType::ComplexDouble),
      "Sparse multiplication with complex128 dtype inputs is not supported with current CUDA version. Please upgrade to CUDA Toolkit 11.2.1+");
#endif

  IntArrayRef A_sizes = A.sizes();
  auto ndim = A.dim();
  auto m = A_sizes[ndim - 2];

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
      "spgemm",
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

        // Resize result using nnz information from cusparse
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
  if (mat.values().dim() == 3 && mat.values().size(-1) > 1) {
    return block_sparse_mv(mat, vec, beta, alpha, result);
  }
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

  // SpMV doesn't support uniform precision computation
  // For float16/bfloat16 inputs compute_type must be CUDA_R_32F
  // and type of alpha, beta must be float
  auto dispatch_scalar_type = result.scalar_type();
  if (dispatch_scalar_type == at::ScalarType::Half ||
      dispatch_scalar_type == at::ScalarType::BFloat16) {
    dispatch_scalar_type = at::ScalarType::Float;
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      dispatch_scalar_type,
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

/*
  Computes C = alpha * A + beta * B

  Args:
  * `A` - [in] sparse Tensor of size m × n.
  * `B` - [in] sparse Tensor of size m × n.
  * `C` - [out] sparse Tensor of size m × n.
*/
void add_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& A,
    const at::sparse_csr::SparseCsrTensor& B,
    const Scalar& alpha,
    const Scalar& beta,
    const at::sparse_csr::SparseCsrTensor& C) {
  IntArrayRef A_sizes = A.sizes();
  auto ndim = A.dim();
  int m = at::native::cuda_int_cast(A_sizes[ndim - 2], "m");
  int n = at::native::cuda_int_cast(A_sizes[ndim - 1], "n");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.sizes().equals(B.sizes()) && A.sizes().equals(C.sizes()));

  // Only 32-bit indices are supported
  auto A_32 = at::native::_sparse_csr_tensor_unsafe(
      A.crow_indices().to(kInt),
      A.col_indices().to(kInt),
      A.values(),
      A.sizes(),
      A.scalar_type(),
      A.layout(),
      A.device());
  auto B_32 = at::native::_sparse_csr_tensor_unsafe(
      B.crow_indices().to(kInt),
      B.col_indices().to(kInt),
      B.values(),
      B.sizes(),
      B.scalar_type(),
      B.layout(),
      B.device());

  // Modify C tensor in-place to swap indices tensors with 32-bit variants
  indices_to_32_bit_inplace(C);

  int nnzA = at::native::cuda_int_cast(A_32._nnz(), "nnzA");
  int nnzB = at::native::cuda_int_cast(B_32._nnz(), "nnzB");

  auto desc = at::cuda::sparse::CuSparseMatDescriptor();

  auto A_crow_indices = A_32.crow_indices();
  auto B_crow_indices = B_32.crow_indices();
  auto C_crow_indices = C.crow_indices();
  auto A_crow_indices_ptr = A_crow_indices.data_ptr<int>();
  auto B_crow_indices_ptr = B_crow_indices.data_ptr<int>();
  auto C_crow_indices_ptr = C_crow_indices.data_ptr<int>();

  auto A_col_indices = A_32.col_indices();
  auto B_col_indices = B_32.col_indices();
  auto C_col_indices = C.col_indices();
  auto A_col_indices_ptr = A_col_indices.data_ptr<int>();
  auto B_col_indices_ptr = B_col_indices.data_ptr<int>();
  auto C_col_indices_ptr = C_col_indices.data_ptr<int>();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      C.scalar_type(), "add_out_sparse_csr_cuda_impl", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();

        auto A_values = A_32.values();
        auto B_values = B_32.values();
        auto C_values = C.values();
        auto A_values_ptr = A_values.data_ptr<scalar_t>();
        auto B_values_ptr = B_values.data_ptr<scalar_t>();
        auto C_values_ptr = C_values.data_ptr<scalar_t>();

        auto handle = at::cuda::getCurrentCUDASparseHandle();
        TORCH_CUDASPARSE_CHECK(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

        size_t buffer_size;
        at::cuda::sparse::csrgeam2_bufferSizeExt<scalar_t>(
            handle,
            m,
            n,
            &alpha_,
            desc.descriptor(),
            nnzA,
            A_values_ptr,
            A_crow_indices_ptr,
            A_col_indices_ptr,
            &beta_,
            desc.descriptor(),
            nnzB,
            B_values_ptr,
            B_crow_indices_ptr,
            B_col_indices_ptr,
            desc.descriptor(),
            C_values_ptr,
            C_crow_indices_ptr,
            C_col_indices_ptr,
            &buffer_size // output
        );

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto work_data = allocator.allocate(buffer_size);

        int nnzC = -1;
        at::cuda::sparse::csrgeam2Nnz<scalar_t>(
            handle,
            m,
            n,
            desc.descriptor(),
            nnzA,
            A_crow_indices_ptr,
            A_col_indices_ptr,
            desc.descriptor(),
            nnzB,
            B_crow_indices_ptr,
            B_col_indices_ptr,
            desc.descriptor(),
            C_crow_indices_ptr,
            &nnzC,
            work_data.get());

        // Resize result using nnz information from cusparse
        col_indices_and_values_resize_(C, nnzC);
        C_col_indices = C.col_indices();
        C_values = C.values();

        C_col_indices_ptr = C_col_indices.data_ptr<int>();
        C_values_ptr = C_values.data_ptr<scalar_t>();

        at::cuda::sparse::csrgeam2<scalar_t>(
            handle,
            m,
            n,
            &alpha_,
            desc.descriptor(),
            nnzA,
            A_values_ptr,
            A_crow_indices_ptr,
            A_col_indices_ptr,
            &beta_,
            desc.descriptor(),
            nnzB,
            B_values_ptr,
            B_crow_indices_ptr,
            B_col_indices_ptr,
            desc.descriptor(),
            C_values_ptr,
            C_crow_indices_ptr,
            C_col_indices_ptr,
            work_data.get());
      });
}

/*
  Solves a system of linear equations whose coefficients are represented in a sparse triangular matrix A:
  op(A) X = B.

  Args:
  * `A` - sparse Tensor of size m × m.
  * `B` - dense Tensor of size m × nrhs.
  * `X` - dense Tensor of size m × nrhs.
  * `upper` - controls whether upper or lower triangular part of A is considered in computations.
  * `transpose` - if true then op(A) = A^T.
  * `unitriangular` - if true then the diagonal elements of A are assumed to be one.
*/
void triangular_solve_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
  if (B.numel() == 0 || X.numel() == 0 || A._nnz() == 0) {
    // If A has no nnz, then A is singular and we can't solve.
    X.fill_(NAN);
    return;
  }
  if (A.values().dim() == 3 && A.values().size(-1) > 1) {
    if (B.size(-1) == 1) {
      return block_sparse_triangular_solve_vec(A, B, X, upper, transpose, unitriangular);
    } else {
      return block_sparse_triangular_solve_mat(A, B, X, upper, transpose, unitriangular);
    }
  }
#if !AT_USE_CUSPARSE_GENERIC_SPSV()
  TORCH_CHECK(
      false,
      "Calling triangular solve on a sparse GPU tensor requires compiling ",
      "PyTorch with at least CUDA 11.3. ",
      "Please use PyTorch built with newer CUDA version.");
#else
  c10::MaybeOwned<Tensor> X_ = prepare_dense_matrix_for_cusparse(X);
  // It should be possible to use mixed memory format
  // but there is a bug in CUDA 11.3.1 version:
  // strides of matrix B are used to write result to matrix X.
  // As a workaround we need to convert matrices to have the same strides.
  c10::MaybeOwned<Tensor> B_ = prepare_dense_matrix_for_cusparse(B, X_->strides());

  // TODO: update this to support COO sparse layout
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(A);
  descA.set_mat_fill_mode(upper);
  descA.set_mat_diag_type(unitriangular);
  cusparseOperation_t opA = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                      : CUSPARSE_OPERATION_NON_TRANSPOSE;

  if (B.size(-1) == 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        X.scalar_type(), "triangular_solve_out_sparse_csr_cuda_impl", [&] {
          scalar_t alpha = 1;
          cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
          auto handle = at::cuda::getCurrentCUDASparseHandle();
          size_t buffer_size;

          auto desc_spsv = at::cuda::sparse::CuSparseSpSVDescriptor();
          auto descB = at::cuda::sparse::CuSparseDnVecDescriptor(*B_);
          auto descX = at::cuda::sparse::CuSparseDnVecDescriptor(*X_);
          TORCH_CUDASPARSE_CHECK(cusparseSpSV_bufferSize(
              handle,
              opA,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSV_ALG_DEFAULT,
              desc_spsv.descriptor(),
              &buffer_size // output
              ));

          auto& allocator = *c10::cuda::CUDACachingAllocator::get();
          auto work_data = allocator.allocate(buffer_size);

          TORCH_CUDASPARSE_CHECK(cusparseSpSV_analysis(
              handle,
              opA,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSV_ALG_DEFAULT,
              desc_spsv.descriptor(),
              work_data.get()));

          TORCH_CUDASPARSE_CHECK(cusparseSpSV_solve(
              handle,
              opA,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSV_ALG_DEFAULT,
              desc_spsv.descriptor()));
        });
  } else {
#if !AT_USE_CUSPARSE_GENERIC_SPSM()
    TORCH_CHECK(
        false,
        "Calling triangular solve on a sparse GPU tensor requires compiling ",
        "PyTorch with at least CUDA 11.3.1. ",
        "Please use PyTorch built with newer CUDA version.");
#else
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        X.scalar_type(), "triangular_solve_out_sparse_csr_cuda_impl", [&] {
          scalar_t alpha = 1;
          cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
          auto handle = at::cuda::getCurrentCUDASparseHandle();
          size_t buffer_size;

          cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
          auto desc_spsm = at::cuda::sparse::CuSparseSpSMDescriptor();
          auto descB = at::cuda::sparse::CuSparseDnMatDescriptor(*B_);
          auto descX = at::cuda::sparse::CuSparseDnMatDescriptor(*X_);
          TORCH_CUDASPARSE_CHECK(cusparseSpSM_bufferSize(
              handle,
              opA,
              opB,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSM_ALG_DEFAULT,
              desc_spsm.descriptor(),
              &buffer_size // output
              ));

          auto& allocator = *c10::cuda::CUDACachingAllocator::get();
          auto work_data = allocator.allocate(buffer_size);

          TORCH_CUDASPARSE_CHECK(cusparseSpSM_analysis(
              handle,
              opA,
              opB,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSM_ALG_DEFAULT,
              desc_spsm.descriptor(),
              work_data.get()));

          TORCH_CUDASPARSE_CHECK(cusparseSpSM_solve(
              handle,
              opA,
              opB,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSM_ALG_DEFAULT,
              desc_spsm.descriptor()));
        });
#endif // !AT_USE_CUSPARSE_GENERIC_SPSM()
  }
  if (!X.is_same(*X_)) {
    X.copy_(*X_);
  }
#endif // !AT_USE_CUSPARSE_GENERIC_SPSV()
}

void sampled_addmm_out_sparse_csr(
    const Tensor& A,
    const Tensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const at::sparse_csr::SparseCsrTensor& C) {
#if !AT_USE_CUSPARSE_GENERIC_SDDMM()
  TORCH_CHECK(
      false,
      "Calling sampled_addmm with sparse GPU tensors requires compiling ",
      "PyTorch with CUDA 11.2.1+. ",
      "Please use PyTorch built with newer CUDA version.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.layout() == Layout::Strided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(B.layout() == Layout::Strided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(C.is_sparse_csr());

  auto descA = at::cuda::sparse::CuSparseDnMatDescriptor(A);
  auto descB = at::cuda::sparse::CuSparseDnMatDescriptor(B);
  auto descC = at::cuda::sparse::CuSparseSpMatCsrDescriptor(C);

  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      C.scalar_type(),
      "sampled_addmm_out_sparse_csr",
      [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        auto compute_type = at::cuda::getCudaDataType<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();
        size_t buffer_size = 0;
        TORCH_CUDASPARSE_CHECK(cusparseSDDMM_bufferSize(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            CUSPARSE_SDDMM_ALG_DEFAULT,
            &buffer_size // output
            ));

        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        auto buffer = allocator.allocate(buffer_size);

        TORCH_CUDASPARSE_CHECK(cusparseSDDMM_preprocess(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            CUSPARSE_SDDMM_ALG_DEFAULT,
            buffer.get()));

        TORCH_CUDASPARSE_CHECK(cusparseSDDMM(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            CUSPARSE_SDDMM_ALG_DEFAULT,
            buffer.get()));
      });
#endif
}

} // namespace cuda
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
