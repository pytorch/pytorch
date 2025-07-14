#include <ATen/native/sparse/eigen/SparseBlasImpl.h>

#if AT_USE_EIGEN_SPARSE()

#include <ATen/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/SparseCsrTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <c10/core/ScalarType.h>

#include <Eigen/SparseCore>

namespace at::native::sparse::impl::eigen {

namespace {

void inline sparse_indices_to_result_dtype_inplace(
    const c10::ScalarType& dtype,
    const at::Tensor& input) {
  if (input.layout() == kSparseCsr) {
    static_cast<at::SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
        ->set_member_tensors(
            input.crow_indices().to(dtype),
            input.col_indices().to(dtype),
            input.values(),
            input.sizes());
  } else {
    static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
        ->set_member_tensors(
            input.ccol_indices().to(dtype),
            input.row_indices().to(dtype),
            input.values(),
            input.sizes());
  }
}

void inline sparse_indices_and_values_resize(
    const at::Tensor& input,
    int64_t nnz) {
  if (input.layout() == kSparseCsr) {
    static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
        ->set_member_tensors(
            input.crow_indices(),
            input.col_indices().resize_({nnz}),
            input.values().resize_({nnz}),
            input.sizes());
  } else {
    static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
        ->set_member_tensors(
            input.ccol_indices(),
            input.row_indices().resize_({nnz}),
            input.values().resize_({nnz}),
            input.sizes());
  }
}

template <typename scalar_t, typename index_t>
const Eigen::Map<Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, index_t>>
Tensor_to_EigenCsc(const at::Tensor& tensor) {
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);
  int64_t nnz = tensor._nnz();

  TORCH_CHECK(tensor.values().is_contiguous(), "eigen accepts only contiguous tensors");

  index_t* ccol_indices_ptr = tensor.ccol_indices().data_ptr<index_t>();
  index_t* row_indices_ptr = tensor.row_indices().data_ptr<index_t>();
  scalar_t* values_ptr = tensor.values().data_ptr<scalar_t>();
  Eigen::Map<Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, index_t>> map(
      rows, cols, nnz, ccol_indices_ptr, row_indices_ptr, values_ptr);
  return map;
}

template <typename scalar_t, typename index_t>
const Eigen::Map<Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>>
Tensor_to_EigenCsr(const at::Tensor& tensor) {
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);
  int64_t nnz = tensor._nnz();

  TORCH_CHECK(tensor.values().is_contiguous(), "eigen accepts only contiguous tensors");

  index_t* crow_indices_ptr = tensor.crow_indices().data_ptr<index_t>();
  index_t* col_indices_ptr = tensor.col_indices().data_ptr<index_t>();
  scalar_t* values_ptr = tensor.values().data_ptr<scalar_t>();
  Eigen::Map<Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>> map(
      rows, cols, nnz, crow_indices_ptr, col_indices_ptr, values_ptr);
  return map;
}

template <typename scalar_t, typename index_t>
void EigenCsr_to_Tensor(
    const at::Tensor& tensor,
    const Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>& matrix) {
  TORCH_CHECK(
      tensor.layout() == kSparseCsr,
      "EigenCsr_to_Tensor, expected tensor be kSparseCsr, but got",
      tensor.layout());

  int64_t nnz = matrix.nonZeros();
  int64_t rows = matrix.outerSize();
  sparse_indices_and_values_resize(tensor, nnz);

  if (nnz > 0) {
    std::memcpy(
        tensor.values().mutable_data_ptr<scalar_t>(),
        matrix.valuePtr(),
        nnz * sizeof(scalar_t));
    std::memcpy(
        tensor.col_indices().mutable_data_ptr<index_t>(),
        matrix.innerIndexPtr(),
        nnz * sizeof(index_t));
  }
  if (rows > 0) {
    std::memcpy(
        tensor.crow_indices().mutable_data_ptr<index_t>(),
        matrix.outerIndexPtr(),
        rows * sizeof(index_t));
  }
  tensor.crow_indices().mutable_data_ptr<index_t>()[rows] = nnz;
}

template <typename scalar_t, typename index_t>
void EigenCsc_to_Tensor(
    const at::Tensor& tensor,
    const Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, index_t>& matrix) {
  TORCH_CHECK(
      tensor.layout() == kSparseCsc,
      "EigenCsr_to_Tensor, expected tensor be kSparseCsc, but got",
      tensor.layout());

  int64_t nnz = matrix.nonZeros();
  int64_t cols = matrix.outerSize();
  sparse_indices_and_values_resize(tensor, nnz);

  if (nnz > 0) {
    std::memcpy(
        tensor.values().mutable_data_ptr<scalar_t>(),
        matrix.valuePtr(),
        nnz * sizeof(scalar_t));
    std::memcpy(
        tensor.row_indices().mutable_data_ptr<index_t>(),
        matrix.innerIndexPtr(),
        nnz * sizeof(index_t));
  }
  if (cols > 0) {
    std::memcpy(
        tensor.ccol_indices().mutable_data_ptr<index_t>(),
        matrix.outerIndexPtr(),
        cols * sizeof(index_t));
  }
  tensor.ccol_indices().mutable_data_ptr<index_t>()[cols] = nnz;
}

template <typename scalar_t>
void add_out_sparse_eigen(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha,
    const at::Tensor& result) {
  // empty matrices
  if (mat1._nnz() == 0 && mat2._nnz() == 0) {
    return;
  }

  if (mat2._nnz() == 0 || alpha.toComplexDouble() == 0.) {
    sparse_indices_and_values_resize(result, mat1._nnz());
    result.copy_(mat1);
    return;
  } else if (mat1._nnz() == 0) {
    sparse_indices_and_values_resize(result, mat2._nnz());
    result.copy_(mat2);
    result.values().mul_(alpha);
    return;
  }

  c10::ScalarType result_index_dtype = at::sparse_csr::getIndexDtype(result);

  sparse_indices_to_result_dtype_inplace(result_index_dtype, mat1);
  sparse_indices_to_result_dtype_inplace(result_index_dtype, mat2);

  AT_DISPATCH_INDEX_TYPES(
      result_index_dtype, "eigen_sparse_add", [&]() {
        scalar_t _alpha = alpha.to<scalar_t>();
        typedef Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, index_t>
            EigenCscMatrix;
        typedef Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>
            EigenCsrMatrix;

        if (result.layout() == kSparseCsr) {
          const Eigen::Map<EigenCsrMatrix> mat1_eigen =
              Tensor_to_EigenCsr<scalar_t, index_t>(mat1);
          const Eigen::Map<EigenCsrMatrix> mat2_eigen =
              Tensor_to_EigenCsr<scalar_t, index_t>(mat2);
          const EigenCsrMatrix mat1_mat2_eigen =
              (mat1_eigen + _alpha * mat2_eigen);

          EigenCsr_to_Tensor<scalar_t, index_t>(result, mat1_mat2_eigen);
        } else {
          const Eigen::Map<EigenCscMatrix> mat1_eigen =
              Tensor_to_EigenCsc<scalar_t, index_t>(mat1);
          const Eigen::Map<EigenCscMatrix> mat2_eigen =
              Tensor_to_EigenCsc<scalar_t, index_t>(mat2);
          const EigenCscMatrix mat1_mat2_eigen =
              (mat1_eigen + _alpha * mat2_eigen);

          EigenCsc_to_Tensor<scalar_t, index_t>(result, mat1_mat2_eigen);
        }
      });
}

template <typename scalar_t>
void addmm_out_sparse_eigen(
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
  // In addition, beta = 0 lets us enable a fast-path for result = alpha * A @ B
  bool is_beta_zero = false;
  if (beta.toComplexDouble() == 0.) {
    is_beta_zero = true;
    result.values().zero_();
  } else {
    result.values().mul_(beta);
  }

  c10::ScalarType result_index_dtype = at::sparse_csr::getIndexDtype(result);

  sparse_indices_to_result_dtype_inplace(result_index_dtype, mat1);
  sparse_indices_to_result_dtype_inplace(result_index_dtype, mat2);

  AT_DISPATCH_INDEX_TYPES(
      result_index_dtype, "eigen_sparse_mm", [&]() {
        typedef Eigen::SparseMatrix<scalar_t, Eigen::ColMajor, index_t>
            EigenCscMatrix;
        typedef Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, index_t>
            EigenCsrMatrix;

        at::Tensor mat1_mat2;
        if (is_beta_zero) {
          mat1_mat2 = result;
        } else {
          mat1_mat2 = at::empty_like(result, result.options());
        }

        if (mat1_mat2.layout() == kSparseCsr) {
          if (mat1.layout() == kSparseCsr) {
            const Eigen::Map<EigenCsrMatrix> mat1_eigen =
                Tensor_to_EigenCsr<scalar_t, index_t>(mat1);
            if (mat2.layout() == kSparseCsr) {
              const Eigen::Map<EigenCsrMatrix> mat2_eigen =
                  Tensor_to_EigenCsr<scalar_t, index_t>(mat2);
              const EigenCsrMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
              EigenCsr_to_Tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
            } else if (mat2.layout() == kSparseCsc) {
              const Eigen::Map<EigenCscMatrix> mat2_eigen =
                  Tensor_to_EigenCsc<scalar_t, index_t>(mat2);
              const EigenCsrMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
              EigenCsr_to_Tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
            }
          } else {
            const Eigen::Map<EigenCscMatrix> mat1_eigen =
                Tensor_to_EigenCsc<scalar_t, index_t>(mat1);
            if (mat2.layout() == kSparseCsc) {
              const Eigen::Map<EigenCscMatrix> mat2_eigen =
                  Tensor_to_EigenCsc<scalar_t, index_t>(mat2);
              const EigenCsrMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
              EigenCsr_to_Tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
            } else if (mat2.layout() == kSparseCsr) {
              const Eigen::Map<EigenCsrMatrix> mat2_eigen =
                  Tensor_to_EigenCsr<scalar_t, index_t>(mat2);
              const EigenCsrMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
              EigenCsr_to_Tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
            }
          }
        } else {
          if (mat1.layout() == kSparseCsr) {
            const Eigen::Map<EigenCsrMatrix> mat1_eigen =
                Tensor_to_EigenCsr<scalar_t, index_t>(mat1);
            if (mat2.layout() == kSparseCsr) {
              const Eigen::Map<EigenCsrMatrix> mat2_eigen =
                  Tensor_to_EigenCsr<scalar_t, index_t>(mat2);
              const EigenCscMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
              EigenCsc_to_Tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
            } else if (mat2.layout() == kSparseCsc) {
              const Eigen::Map<EigenCscMatrix> mat2_eigen =
                  Tensor_to_EigenCsc<scalar_t, index_t>(mat2);
              const EigenCscMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
              EigenCsc_to_Tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
            }
          } else {
            const Eigen::Map<EigenCscMatrix> mat1_eigen =
                Tensor_to_EigenCsc<scalar_t, index_t>(mat1);
            if (mat2.layout() == kSparseCsc) {
              const Eigen::Map<EigenCscMatrix> mat2_eigen =
                  Tensor_to_EigenCsc<scalar_t, index_t>(mat2);
              const EigenCscMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
              EigenCsc_to_Tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
            } else if (mat2.layout() == kSparseCsr) {
              const Eigen::Map<EigenCsrMatrix> mat2_eigen =
                  Tensor_to_EigenCsr<scalar_t, index_t>(mat2);
              const EigenCscMatrix mat1_mat2_eigen = (mat1_eigen * mat2_eigen);
              EigenCsc_to_Tensor<scalar_t, index_t>(mat1_mat2, mat1_mat2_eigen);
            }
          }
        }

        if (is_beta_zero) {
          result.mul_(alpha.to<scalar_t>());
        } else {
          result.add_(mat1_mat2, alpha.to<scalar_t>());
        }
      });
}

} // anonymus namespace

void addmm_out_sparse(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& result,
    const at::Scalar& alpha,
    const at::Scalar& beta) {
  AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(mat1.layout(), "eigen::addmm_out_sparse:mat1", [&]{});
  AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(mat2.layout(), "eigen::addmm_out_sparse:mat2", [&]{});
  AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(result.layout(), "eigen::addmm_out_sparse:result", [&]{});

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    result.scalar_type(), "addmm_out_sparse_eigen", [&] {
      addmm_out_sparse_eigen<scalar_t>(mat1, mat2, result, alpha, beta);
  });
}

void add_out_sparse(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha,
    const at::Tensor& result) {
  TORCH_CHECK(
      (result.layout() == kSparseCsr && mat1.layout() == kSparseCsr && mat2.layout() == kSparseCsr) ||
      (result.layout() == kSparseCsc && mat1.layout() == kSparseCsc && mat2.layout() == kSparseCsc),
      "eigen::add_out_sparse: expected the same layout for all operands but got ",
      mat1.layout(),
      " + ",
      mat2.layout(),
      " -> ",
      result.layout());

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    result.scalar_type(), "add_out_sparse_eigen", [&] {
      add_out_sparse_eigen<scalar_t>(mat1, mat2, alpha, result);
  });
}

} // namespace at::native::sparse::impl::eigen

#else

namespace at::native::sparse::impl::eigen {

void addmm_out_sparse(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& result,
    const at::Scalar& alpha,
    const at::Scalar& beta) {
    TORCH_CHECK(
      false,
      "eigen::addmm_out_sparse: Eigen was not enabled for ",
      result.layout(),
      " + ",
      mat1.layout(),
      " @ ",
      mat2.layout());
}

void add_out_sparse(
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha,
    const at::Tensor& result) {
    TORCH_CHECK(
      false,
      "eigen::add_out_sparse: Eigen was not enabled for ",
      mat1.layout(),
      " + ",
      mat2.layout(),
      " -> ",
      result.layout());
}

} // namespace at::native::sparse::impl::eigen

#endif // AT_USE_EIGEN_SPARSE()
