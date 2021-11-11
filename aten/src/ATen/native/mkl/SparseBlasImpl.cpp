#include <ATen/Dispatch.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/mkl/Sparse.h>
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

} // anonymous namespace

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
        sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
        MKL_INT rows_C, cols_C;
        MKL_INT *rows_start_C = nullptr, *rows_end_C = nullptr,
                *columns_C = nullptr;
        scalar_t* values_C = nullptr;
        at::mkl::sparse::export_csr(
            result_desc,
            &indexing,
            &rows_C,
            &cols_C,
            &rows_start_C,
            &rows_end_C,
            &columns_C,
            &values_C);

        // Resize result using nnz information from MKL
        MKL_INT nnz_C = rows_end_C[rows_C - 1];
        col_indices_and_values_resize_(result, nnz_C);

        auto result_crow_indices = result.crow_indices();
        auto result_col_indices = result.col_indices();
        auto result_values = result.values();

        // MKL Sparse Inspector-Executor doesn't have a way to provide external
        // buffers So we have to copy the memory allocated by MKL
        std::memcpy(
            result_values.data_ptr<scalar_t>(),
            values_C,
            nnz_C * sizeof(scalar_t));
        std::memcpy(
            result_col_indices.data_ptr<MKL_INT>(),
            columns_C,
            nnz_C * sizeof(MKL_INT));
        std::memcpy(
            result_crow_indices.data_ptr<MKL_INT>(),
            rows_start_C,
            rows_C * sizeof(MKL_INT));
        result_crow_indices.data_ptr<MKL_INT>()[rows_C] = nnz_C;
      });
#endif
}

} // namespace mkl
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
