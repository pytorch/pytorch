#pragma once

/*
  Provides templated descriptor wrappers of MKL Sparse BLAS sparse matrices:

    MklSparseCsrDescriptor<scalar_t>(sparse_csr_tensor)

  where scalar_t is double, float, c10::complex<double> or c10::complex<float>.
  The descriptors are available in at::mkl::sparse namespace.
*/

#include <ATen/Tensor.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Utils.h>

#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>

#include <mkl_spblas.h>

namespace at::mkl::sparse {

template <typename T, sparse_status_t (*destructor)(T*)>
struct MklSparseDescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_MKLSPARSE_CHECK(destructor(x));
    }
  }
};

template <typename T, sparse_status_t (*destructor)(T*)>
class MklSparseDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, MklSparseDescriptorDeleter<T, destructor>> descriptor_;
};

namespace {

c10::MaybeOwned<Tensor> inline prepare_indices_for_mkl(const Tensor& indices) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      isIntegralType(indices.scalar_type(), /*includeBool=*/false));
#ifdef MKL_ILP64
  // ILP64 is a 64-bit API version of MKL
  // Indices tensor must have ScalarType::Long type
  if (indices.scalar_type() == ScalarType::Long) {
    return c10::MaybeOwned<Tensor>::borrowed(indices);
  } else {
    return c10::MaybeOwned<Tensor>::owned(indices.to(ScalarType::Long));
  }
#else
  // LP64 is a 32-bit API version of MKL
  // Indices tensor must have ScalarType::Int type
  if (indices.scalar_type() == ScalarType::Int) {
    return c10::MaybeOwned<Tensor>::borrowed(indices);
  } else {
    return c10::MaybeOwned<Tensor>::owned(indices.to(ScalarType::Int));
  }
#endif
}

} // anonymous namespace

template <typename scalar_t>
class MklSparseCsrDescriptor
    : public MklSparseDescriptor<sparse_matrix, &mkl_sparse_destroy> {
 public:
  MklSparseCsrDescriptor(const Tensor& input) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY((input.layout() == kSparseCsr || input.layout() == kSparseBsr));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() == 2);

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        input._nnz() > 0, "MKL doesn't work with empty CSR matrices");

    IntArrayRef input_sizes = input.sizes();
    auto rows = mkl_int_cast(input_sizes[0], "rows");
    auto cols = mkl_int_cast(input_sizes[1], "cols");

    auto crow_indices = input.crow_indices();
    auto col_indices = input.col_indices();
    auto values = input.values();

    crow_indices_ = prepare_indices_for_mkl(crow_indices);
    col_indices_ = prepare_indices_for_mkl(col_indices);
    values_ = values.expect_contiguous();

    auto values_ptr = values_->data_ptr<scalar_t>();
    auto crow_indices_ptr = crow_indices_->data_ptr<MKL_INT>();
    auto col_indices_ptr = col_indices_->data_ptr<MKL_INT>();

    sparse_matrix_t raw_descriptor;

    if (input.layout() == kSparseBsr) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          values.dim() == 3 && crow_indices.dim() == 1 &&
          col_indices.dim() == 1);
      TORCH_CHECK(
          values.size(-1) == values.size(-2),
          "MKL Sparse doesn't support matrices with non-square blocks.");
      auto block_size = mkl_int_cast(values.size(-1), "block_size");
      create_bsr<scalar_t>(
          &raw_descriptor,
          SPARSE_INDEX_BASE_ZERO,
          SPARSE_LAYOUT_ROW_MAJOR,
          rows / block_size,
          cols / block_size,
          block_size,
          crow_indices_ptr,
          crow_indices_ptr + 1,
          col_indices_ptr,
          values_ptr);
    } else {
      create_csr<scalar_t>(
          &raw_descriptor,
          SPARSE_INDEX_BASE_ZERO,
          rows,
          cols,
          crow_indices_ptr,
          crow_indices_ptr + 1,
          col_indices_ptr,
          values_ptr);
    }

    descriptor_.reset(raw_descriptor);
  }

  MklSparseCsrDescriptor() {
    sparse_matrix_t raw_descriptor = nullptr;
    descriptor_.reset(raw_descriptor);
  }

 private:
  c10::MaybeOwned<Tensor> crow_indices_;
  c10::MaybeOwned<Tensor> col_indices_;
  c10::MaybeOwned<Tensor> values_;
};

} // namespace at::mkl::sparse
