#include <ATen/native/eigen/SparseBlasImpl.h>

#include <ATen/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/SparseCsrTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <c10/core/ScalarType.h>

#include <Eigen/SparseCore>

namespace at::native::sparse::impl::eigen {

namespace {

void inline csr_indices_to_result_dtype_inplace(
    const c10::ScalarType& dtype,
    const at::Tensor& input) {
  if (input.layout() == kSparseCsr) {
    static_cast<at::SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
        ->set_member_tensors(
            input.crow_indices().to(dtype),
            input.col_indices().to(dtype),
            input.values(),
            input.sizes());
  }
}

void inline csr_col_indices_and_values_resize(
    const at::Tensor& input,
    int64_t nnz) {
  if (input.layout() == kSparseCsr) {
    static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
        ->set_member_tensors(
            input.crow_indices(),
            input.col_indices().resize_({nnz}),
            input.values().resize_({nnz}),
            input.sizes());
  }
}

template <typename scalar_t, typename index_t>
const Eigen::Map<Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, index_t>>
to_EigenCsc(const at::Tensor& tensor) {
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);
  int64_t nnz = tensor._nnz();

  index_t* ccol_indices_ptr = tensor.ccol_indices().data_ptr<index_t>();
  index_t* row_indices_ptr = tensor.row_indices().data_ptr<index_t>();
  scalar_t* values_ptr = tensor.values().data_ptr<scalar_t>();
  Eigen::Map<Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, index_t>> map(
      rows, cols, nnz, ccol_indices_ptr, row_indices_ptr, values_ptr);
  return map;
}

template <typename scalar_t, typename index_t>
const Eigen::Map<Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>>
to_EigenCsr(const at::Tensor& tensor) {
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);
  int64_t nnz = tensor._nnz();

  index_t* crow_indices_ptr = tensor.crow_indices().data_ptr<index_t>();
  index_t* col_indices_ptr = tensor.col_indices().data_ptr<index_t>();
  scalar_t* values_ptr = tensor.values().data_ptr<scalar_t>();
  Eigen::Map<Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>> map(
      rows, cols, nnz, crow_indices_ptr, col_indices_ptr, values_ptr);
  return map;
}

template <typename scalar_t, typename index_t>
void EigenCsr_to_tensor(
    const at::Tensor& tensor,
    const Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>& matrix) {
  TORCH_CHECK(
      tensor.layout() == kSparseCsr,
      "EigenCsr_to_tensor, expected tensor be kSparseCsr, but got",
      tensor.layout());

  int64_t nnz = matrix.nonZeros();
  int64_t rows = matrix.outerSize();
  csr_col_indices_and_values_resize(tensor, nnz);

  if (nnz > 0) {
    std::memcpy(
        tensor.values().data_ptr<scalar_t>(),
        matrix.valuePtr(),
        nnz * sizeof(scalar_t));
    std::memcpy(
        tensor.col_indices().data_ptr<index_t>(),
        matrix.innerIndexPtr(),
        nnz * sizeof(index_t));
  }
  if (rows > 0) {
    std::memcpy(
        tensor.crow_indices().data_ptr<index_t>(),
        matrix.outerIndexPtr(),
        rows * sizeof(index_t));
  }
  tensor.crow_indices().data_ptr<index_t>()[rows] = nnz;
}

template <typename scalar_t>
void add_out_sparse_csr_eigen(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha,
    const at::Tensor& result) {
  // empty matrices
  if (mat1._nnz() == 0 && mat2._nnz() == 0) {
    return;
  }

  if (mat2._nnz() == 0 || alpha.toComplexDouble() == 0.) {
    csr_col_indices_and_values_resize(result, mat1._nnz());
    result.copy_(mat1);
    return;
  } else if (mat1._nnz() == 0) {
    csr_col_indices_and_values_resize(result, mat2._nnz());
    result.copy_(mat2);
    result.values().mul_(alpha);
    return;
  }

  csr_indices_to_result_dtype_inplace(
      result.col_indices().scalar_type(), mat1);
  csr_indices_to_result_dtype_inplace(
      result.col_indices().scalar_type(), mat2);

  AT_DISPATCH_INDEX_TYPES(
      result.col_indices().scalar_type(), "eigen_sparse_add", [&]() {
        scalar_t _alpha = alpha.to<scalar_t>();
        typedef Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>
            EigenCsrMatrix;

        const Eigen::Map<EigenCsrMatrix> mat1_eigen =
            to_EigenCsr<scalar_t, index_t>(mat1);
        const Eigen::Map<EigenCsrMatrix> mat2_eigen =
            to_EigenCsr<scalar_t, index_t>(mat2);
        const EigenCsrMatrix mat1_mat2_eigen =
            (mat1_eigen + _alpha * mat2_eigen);

        EigenCsr_to_tensor<scalar_t, index_t>(result, mat1_mat2_eigen);
      });
}

template <typename scalar_t>
void addmm_out_sparse_csr_eigen(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& result,
    const at::Scalar& alpha,
    const at::Scalar& beta) {
  // empty matrices
  if (mat1._nnz() == 0 || mat2._nnz() == 0) {
    return;
  }

  // If beta is zero NaN and Inf should not be propagated to the result
  if (beta.toComplexDouble() == 0.) {
    result.values().zero_();
  } else {
    result.values().mul_(beta);
  }

  csr_indices_to_result_dtype_inplace(
      result.col_indices().scalar_type(), mat1);
  csr_indices_to_result_dtype_inplace(
      result.col_indices().scalar_type(), mat2);

  AT_DISPATCH_INDEX_TYPES(
      result.col_indices().scalar_type(), "eigen_sparse_mm", [&]() {
        typedef Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, index_t>
            EigenCscMatrix;
        typedef Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>
            EigenCsrMatrix;

        at::Tensor mat1_mat2 = at::empty(result.sizes(), result.options());

        if (mat1.layout() == kSparseCsr) {
          const Eigen::Map<EigenCsrMatrix> mat1_eigen =
              to_EigenCsr<scalar_t, index_t>(mat1);
          if (mat2.layout() == kSparseCsr) {
            const Eigen::Map<EigenCsrMatrix> mat2_eigen =
                to_EigenCsr<scalar_t, index_t>(mat2);
            const EigenCsrMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
            EigenCsr_to_tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
          } else if (mat2.layout() == kSparseCsc) {
            const Eigen::Map<EigenCscMatrix> mat2_eigen =
                to_EigenCsc<scalar_t, index_t>(mat2);
            const EigenCsrMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
            EigenCsr_to_tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
          }
        } else if (mat1.layout() == kSparseCsc) {
          const Eigen::Map<EigenCscMatrix> mat1_eigen =
              to_EigenCsc<scalar_t, index_t>(mat1);
          if (mat2.layout() == kSparseCsc) {
            const Eigen::Map<EigenCscMatrix> mat2_eigen =
                to_EigenCsc<scalar_t, index_t>(mat2);
            const EigenCsrMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
            EigenCsr_to_tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
          } else if (mat2.layout() == kSparseCsr) {
            const Eigen::Map<EigenCsrMatrix> mat2_eigen =
                to_EigenCsr<scalar_t, index_t>(mat2);
            const EigenCsrMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
            EigenCsr_to_tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
          }
        }

        result.add_(mat1_mat2, alpha.to<scalar_t>());
      });
}

} // anonymus namespace

void addmm_out_sparse(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& result,
    const at::Scalar& alpha,
    const at::Scalar& beta) {
  TORCH_CHECK(
      result.layout() == kSparseCsr && mat1.layout() != kStrided && mat2.layout() != kStrided,
      "eigen::addmm_out_sparse: computation on CPU is not implemented for ",
      result.layout(),
      " + ",
      mat1.layout(),
      " @ ",
      mat2.layout());

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    result.scalar_type(), "addmm_out_sparse_csr_eigen", [&] {
      addmm_out_sparse_csr_eigen<scalar_t>(mat1, mat2, result, alpha, beta);
  });
}

void add_out_sparse(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha,
    const at::Tensor& result) {
  TORCH_CHECK(
      result.layout() == kSparseCsr && mat1.layout() == kSparseCsr && mat2.layout() == kSparseCsr,
      "eigen::add_out_sparse: computation on CPU is not implemented for ",
      mat1.layout(),
      " + ",
      mat2.layout(),
      " -> ",
      result.layout());

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    result.scalar_type(), "addmm_out_sparse_csr_eigen", [&] {
      add_out_sparse_csr_eigen<scalar_t>(mat1, mat2, alpha, result);
  });
}

} // namespace at::native::eigen::sparse
