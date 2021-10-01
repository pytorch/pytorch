#include <ATen/Dispatch.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/mkl/Sparse.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mkl/SparseBlasImpl.h>

#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>

#if AT_USE_MKL_SPARSE()
#include <ATen/mkl/SparseBlas.h>
#include <ATen/mkl/SparseDescriptors.h>
#endif

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace mkl {

namespace {

c10::MaybeOwned<Tensor> prepare_dense_matrix_for_mkl(
    const Tensor& tensor) {
  if (tensor.is_non_overlapping_and_dense() ||
      is_blas_compatible_row_major_order(tensor) ||
      is_blas_compatible_column_major_order(tensor)) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

/*
  Get row-major or column-major matrix.

  Args:
  * `tensor` - 2D strided Tensor.
  * `row_major` - coltrol the memory layout.
*/
c10::MaybeOwned<Tensor> prepare_dense_matrix_for_mkl(
    const Tensor& tensor,
    bool row_major) {
  if (is_blas_compatible_row_major_order(tensor) && row_major) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    if (row_major) {
      return c10::MaybeOwned<Tensor>::owned(
          tensor.clone(at::MemoryFormat::Contiguous));
    } else {
      return c10::MaybeOwned<Tensor>::owned(cloneBatchedColumnMajor(tensor));
    }
  }
}

c10::MaybeOwned<Tensor> inline prepare_dense_vector_for_mkl(
    const Tensor& tensor) {
  if (tensor.is_non_overlapping_and_dense()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

void inline indices_to_mkl_compatible_inplace(const Tensor& input) {
#ifdef MKL_ILP64
  // ILP64 is a 64-bit API version of MKL
  // Indices tensor must have ScalarType::Long type
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
      ->set_member_tensors(
          input.crow_indices().to(kLong),
          input.col_indices().to(kLong),
          input.values(),
          input.sizes());
#else
  // LP64 is a 32-bit API version of MKL
  // Indices tensor must have ScalarType::Int type
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
      ->set_member_tensors(
          input.crow_indices().to(kInt),
          input.col_indices().to(kInt),
          input.values(),
          input.sizes());
#endif
}

void inline col_indices_and_values_resize_(const Tensor& input, int64_t nnz) {
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
      ->set_member_tensors(
          input.crow_indices(),
          input.col_indices().resize_({nnz}),
          input.values().resize_({nnz}),
          input.sizes());
}

/*
  Resizes `input` tensor and fills it with the data from MKL.
*/
template <typename scalar_t>
void mkl_result_copy_(const Tensor& input, sparse_matrix_t mkl_desc) {
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  MKL_INT rows, cols;
  MKL_INT *rows_start = nullptr, *rows_end = nullptr, *columns = nullptr;
  scalar_t* values = nullptr;
  at::mkl::sparse::export_csr(
      mkl_desc,
      &indexing,
      &rows,
      &cols,
      &rows_start,
      &rows_end,
      &columns,
      &values);

  // Resize input using nnz information from MKL
  MKL_INT nnz = rows_end[rows - 1];
  col_indices_and_values_resize_(input, nnz);

  auto crow_indices = input.crow_indices();
  auto col_indices = input.col_indices();
  auto input_values = input.values();

  // MKL Sparse Inspector-Executor doesn't have a way to provide external
  // buffers So we have to copy the memory allocated by MKL
  std::memcpy(
      input_values.data_ptr<scalar_t>(), values, nnz * sizeof(scalar_t));
  std::memcpy(col_indices.data_ptr<MKL_INT>(), columns, nnz * sizeof(MKL_INT));
  std::memcpy(
      crow_indices.data_ptr<MKL_INT>(), rows_start, rows * sizeof(MKL_INT));
  crow_indices.data_ptr<MKL_INT>()[rows] = nnz;
}

/*
  Computes a sparse matrix-dense matrix product defined as
  C <- alpha*(A*B) + beta*C

  Args:
  * `A` - Sparse Tensor storing m x k matrix.
  * `B` - Dense Tensor storing k x n matrix.
  * `C` - [in] Dense Tensor storing matrix of size m x n.
          [out] result of the operation.
*/
void addmm_dense_result(
    const Tensor& A,
    const Tensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& C) {
#if !AT_USE_MKL_SPARSE()
  TORCH_CHECK(
      false,
      "Calling addmm on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  c10::MaybeOwned<Tensor> C_ = prepare_dense_matrix_for_mkl(C);
  IntArrayRef C_strides = C_->strides();
  auto ndim = C_->dim();
  bool is_C_row_major = (C_strides[ndim - 1] == 1);

  // MKL requires same storage layout of matrices
  c10::MaybeOwned<Tensor> B_ = prepare_dense_matrix_for_mkl(B, is_C_row_major);
  IntArrayRef B_strides = B_->strides();
  bool is_B_row_major = (B_strides[ndim - 1] == 1);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!(is_C_row_major ^ is_B_row_major));

  auto order =
      is_C_row_major ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
  auto ldc = is_C_row_major ? C_strides[ndim - 2] : C_strides[ndim - 1];
  auto ldb = is_B_row_major ? B_strides[ndim - 2] : B_strides[ndim - 1];
  auto columns_C = mkl_int_cast(C.size(-1), "columns_C");

  matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      C.scalar_type(), "addmm_out_sparse_csr_impl_mkl", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();

        auto mkl_sparse_mat =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(A);
        at::mkl::sparse::mm<scalar_t>(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha_,
            mkl_sparse_mat.descriptor(),
            descrA,
            order,
            B_->data_ptr<scalar_t>(),
            columns_C,
            ldb,
            beta_,
            C_->data_ptr<scalar_t>(),
            ldc);
      });

  if (!C.is_same(*C_)) {
    C.copy_(*C_);
  }
#endif
}

/*
  Computes a sparse matrix-sparse matrix product defined as
  C <- alpha*(A*B) + beta*C

  Args:
  * `mat1` - Sparse CSR Tensor storing m x k matrix A.
  * `mat2` - Sparse CSR Tensor storing k x n matrix B.
  * `result` - [in] Sparse CSR Tensor storing matrix C of size m x n.
               [out] result of the operation.
*/
void addmm_sparse_result(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_MKL_SPARSE()
  TORCH_CHECK(
      false,
      "Calling add on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  // Compute beta*result because MKL doesn't do it
  // If beta is zero NaN and Inf should not be propagated to the result
  if (beta.toComplexDouble() == 0.) {
    result.values().zero_();
  } else {
    result.values().mul_(beta);
  }

  // MKL doesn't work with empty matrices
  if (mat1._nnz() == 0 || mat2._nnz() == 0) {
    return;
  }

  // MKL doesn't have an interface to compute alpha*(A*B) + beta*C at once
  Tensor mat1_mat2 = at::empty(result.sizes(), result.options());
  indices_to_mkl_compatible_inplace(mat1_mat2);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "addmm_out_sparse_csr_impl_mkl_sparse", [&] {
        auto mkl_sparse_mat1 =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat1);
        auto mkl_sparse_mat2 =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat2);
        auto mkl_result = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>();
        auto result_desc = mkl_result.descriptor();

        TORCH_MKLSPARSE_CHECK(mkl_sparse_spmm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            mkl_sparse_mat1.descriptor(),
            mkl_sparse_mat2.descriptor(),
            &result_desc));

        // copy the data from MKL, otherwise computed result will be destroyed
        // together with `mkl_result`
        mkl_result_copy_<scalar_t>(mat1_mat2, result_desc);
      });

  result.add_(mat1_mat2, alpha);
#endif
}

} // anonymous namespace

/*
  Computes a matrix-matrix product defined as
  C <- alpha*(A*B) + beta*C

  Args:
  * `mat1` - Tensor storing m x k matrix A.
  * `mat2` - Tensor storing k x n matrix B.
  * `result` - [in] Tensor storing matrix C of size m x n.
               [out] result of the operation.
*/
void addmm_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.dim() == 2 && mat2.dim() == 2 && result.dim() == 2);
  if (mat2.layout() == kStrided && result.layout() == kStrided) {
    return addmm_dense_result(mat1, mat2, beta, alpha, result);
  } else if (mat2.is_sparse_csr() && result.is_sparse_csr()) {
    return addmm_sparse_result(mat1, mat2, beta, alpha, result);
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "addmm: Received unexpected tensor layouts as input.");
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
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_MKL_SPARSE()
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  c10::MaybeOwned<Tensor> result_ = prepare_dense_vector_for_mkl(result);
  c10::MaybeOwned<Tensor> vec_ = prepare_dense_vector_for_mkl(vec);

  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;
  matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "addmv_out_sparse_csr_impl_mkl", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();

        auto mkl_sparse_mat =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat);

        at::mkl::sparse::mv<scalar_t>(
            opA,
            alpha_,
            mkl_sparse_mat.descriptor(),
            descrA,
            vec_->data_ptr<scalar_t>(),
            beta_,
            result_->data_ptr<scalar_t>());
      });

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif
}

void add_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_MKL_SPARSE()
  TORCH_CHECK(
      false,
      "Calling add on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else

  // MKL doesn't work with empty matrices
  if (mat2._nnz() == 0) {
    col_indices_and_values_resize_(result, mat1._nnz());
    result.copy_(mat1);
    return;
  } else if (mat1._nnz() == 0) {
    col_indices_and_values_resize_(result, mat2._nnz());
    result.copy_(mat2);
    result.values().mul_(alpha);
    return;
  }

  // Modify `result` tensor in-place to swap indices tensors with 32-bit (or
  // 64-bit) variants
  indices_to_mkl_compatible_inplace(result);
  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "add_out_sparse_csr_impl_mkl", [&] {
        auto alpha_ = alpha.to<scalar_t>();

        auto mkl_mat1 = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat1);
        auto mkl_mat2 = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat2);
        auto mkl_result = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>();

        // Note that the order the order of mat1 and mat2 arguments is swapped
        // because MKL computes alpha*mat1 + mat2 while PyTorch needs mat1 +
        // alpha*mat2
        auto result_desc = mkl_result.descriptor();
        at::mkl::sparse::add<scalar_t>(
            opA,
            mkl_mat2.descriptor(),
            alpha_,
            mkl_mat1.descriptor(),
            &result_desc);

        // now copy data from `result_desc` to `result`
        mkl_result_copy_<scalar_t>(result, result_desc);
      });
#endif
}

} // namespace mkl
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
