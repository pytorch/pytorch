#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/CompositeRandomAccessor.h>
#include <c10/util/irange.h>
#include <unordered_map>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_sparse_matmul_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

using namespace at::sparse;

/*
    This is an implementation of the SMMP algorithm:
     "Sparse Matrix Multiplication Package (SMMP)"

      Randolph E. Bank and Craig C. Douglas
      https://doi.org/10.1007/BF02070824
*/
namespace {
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
void csr_to_coo(const int64_t n_row, const int64_t Ap[], int64_t Bi[]) {
  /*
    Expands a compressed row pointer into a row indices array
    Inputs:
      `n_row` is the number of rows in `Ap`
      `Ap` is the row pointer

    Output:
      `Bi` is the row indices
  */
  for (const auto i : c10::irange(n_row)) {
    for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++) {
      Bi[jj] = i;
    }
  }
}

template<typename index_t_ptr = int64_t*>
int64_t _csr_matmult_maxnnz(
    const int64_t n_row,
    const int64_t n_col,
    const index_t_ptr Ap,
    const index_t_ptr Aj,
    const index_t_ptr Bp,
    const index_t_ptr Bj) {
  /*
    Compute needed buffer size for matrix `C` in `C = A@B` operation.

    The matrices should be in proper CSR structure, and their dimensions
    should be compatible.
  */
  std::vector<int64_t> mask(n_col, -1);
  int64_t nnz = 0;
  for (const auto i : c10::irange(n_row)) {
    int64_t row_nnz = 0;

    for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++) {
      int64_t j = Aj[jj];
      for (int64_t kk = Bp[j]; kk < Bp[j + 1]; kk++) {
        int64_t k = Bj[kk];
        if (mask[k] != i) {
          mask[k] = i;
          row_nnz++;
        }
      }
    }
    int64_t next_nnz = nnz + row_nnz;
    nnz = next_nnz;
  }
  return nnz;
}

template<typename index_t_ptr, typename scalar_t_ptr>
void _csr_matmult(
    const int64_t n_row,
    const int64_t n_col,
    const index_t_ptr Ap,
    const index_t_ptr Aj,
    const scalar_t_ptr Ax,
    const index_t_ptr Bp,
    const index_t_ptr Bj,
    const scalar_t_ptr Bx,
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename index_t_ptr::value_type Cp[],
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename index_t_ptr::value_type Cj[],
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    typename scalar_t_ptr::value_type Cx[]) {
  /*
    Compute CSR entries for matrix C = A@B.

    The matrices `A` and 'B' should be in proper CSR structure, and their dimensions
    should be compatible.

    Inputs:
      `n_row`         - number of row in A
      `n_col`         - number of columns in B
      `Ap[n_row+1]`   - row pointer
      `Aj[nnz(A)]`    - column indices
      `Ax[nnz(A)]     - nonzeros
      `Bp[?]`         - row pointer
      `Bj[nnz(B)]`    - column indices
      `Bx[nnz(B)]`    - nonzeros
    Outputs:
      `Cp[n_row+1]` - row pointer
      `Cj[nnz(C)]`  - column indices
      `Cx[nnz(C)]`  - nonzeros

    Note:
      Output arrays Cp, Cj, and Cx must be preallocated
  */
  using index_t = typename index_t_ptr::value_type;
  using scalar_t = typename scalar_t_ptr::value_type;

  std::vector<index_t> next(n_col, -1);
  std::vector<scalar_t> sums(n_col, 0);

  int64_t nnz = 0;

  Cp[0] = 0;

  for (const auto i : c10::irange(n_row)) {
    index_t head = -2;
    index_t length = 0;

    index_t jj_start = Ap[i];
    index_t jj_end = Ap[i + 1];
    for (const auto jj : c10::irange(jj_start, jj_end)) {
      index_t j = Aj[jj];
      scalar_t v = Ax[jj];

      index_t kk_start = Bp[j];
      index_t kk_end = Bp[j + 1];
      for (const auto kk : c10::irange(kk_start, kk_end)) {
        index_t k = Bj[kk];

        sums[k] += v * Bx[kk];

        if (next[k] == -1) {
          next[k] = head;
          head = k;
          length++;
        }
      }
    }

    for (C10_UNUSED const auto jj : c10::irange(length)) {

      // NOTE: the linked list that encodes col indices
      // is not guaranteed to be sorted.
      Cj[nnz] = head;
      Cx[nnz] = sums[head];
      nnz++;

      index_t temp = head;
      head = next[head];

      next[temp] = -1; // clear arrays
      sums[temp] = 0;
    }

    // Make sure that col indices are sorted.
    // TODO: a better approach is to implement a CSR @ CSC kernel.
    // NOTE: Cx arrays are expected to be contiguous!
    auto col_indices_accessor = StridedRandomAccessor<int64_t>(Cj + nnz - length, 1);
    auto val_accessor = StridedRandomAccessor<scalar_t>(Cx + nnz - length, 1);
    auto kv_accessor = CompositeRandomAccessorCPU<
      decltype(col_indices_accessor), decltype(val_accessor)
    >(col_indices_accessor, val_accessor);
    std::sort(kv_accessor, kv_accessor + length, [](const auto& lhs, const auto& rhs) -> bool {
        return get<0>(lhs) < get<0>(rhs);
    });

    Cp[i + 1] = nnz;
  }
}


template <typename scalar_t>
void sparse_matmul_kernel(
    Tensor& output,
    const Tensor& mat1,
    const Tensor& mat2) {
  /*
    Computes  the sparse-sparse matrix multiplication between `mat1` and `mat2`, which are sparse tensors in COO format.
  */

  auto M = mat1.size(0);
  auto N = mat2.size(1);

  const auto mat1_csr = mat1.to_sparse_csr();
  const auto mat2_csr = mat2.to_sparse_csr();

  auto mat1_crow_indices_ptr = StridedRandomAccessor<int64_t>(
      mat1_csr.crow_indices().data_ptr<int64_t>(),
      mat1_csr.crow_indices().stride(-1));
  auto mat1_col_indices_ptr = StridedRandomAccessor<int64_t>(
      mat1_csr.col_indices().data_ptr<int64_t>(),
      mat1_csr.col_indices().stride(-1));
  auto mat1_values_ptr = StridedRandomAccessor<scalar_t>(
      mat1_csr.values().data_ptr<scalar_t>(),
      mat1_csr.values().stride(-1));
  auto mat2_crow_indices_ptr = StridedRandomAccessor<int64_t>(
      mat2_csr.crow_indices().data_ptr<int64_t>(),
      mat2_csr.crow_indices().stride(-1));
  auto mat2_col_indices_ptr = StridedRandomAccessor<int64_t>(
      mat2_csr.col_indices().data_ptr<int64_t>(),
      mat2_csr.col_indices().stride(-1));
  auto mat2_values_ptr = StridedRandomAccessor<scalar_t>(
      mat2_csr.values().data_ptr<scalar_t>(),
      mat2_csr.values().stride(-1));

  const auto nnz = _csr_matmult_maxnnz(
      M,
      N,
      mat1_crow_indices_ptr,
      mat1_col_indices_ptr,
      mat2_crow_indices_ptr,
      mat2_col_indices_ptr);

  auto output_indices = output._indices();
  auto output_values = output._values();

  Tensor output_indptr = at::empty({M + 1}, kLong);
  at::native::resize_output(output_indices, {2, nnz});
  at::native::resize_output(output_values, nnz);

  Tensor output_row_indices = output_indices.select(0, 0);
  Tensor output_col_indices = output_indices.select(0, 1);

  // TODO: replace with a CSR @ CSC kernel for better performance.
  _csr_matmult(
      M,
      N,
      mat1_crow_indices_ptr,
      mat1_col_indices_ptr,
      mat1_values_ptr,
      mat2_crow_indices_ptr,
      mat2_col_indices_ptr,
      mat2_values_ptr,
      output_indptr.data_ptr<int64_t>(),
      output_col_indices.data_ptr<int64_t>(),
      output_values.data_ptr<scalar_t>());

  csr_to_coo(M, output_indptr.data_ptr<int64_t>(), output_row_indices.data_ptr<int64_t>());
  output._coalesced_(true);
}

} // end anonymous namespace

Tensor sparse_sparse_matmul_cpu(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
  TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
  TORCH_CHECK(mat1_.dim() == 2);
  TORCH_CHECK(mat2_.dim() == 2);
  TORCH_CHECK(mat1_.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", mat1_.dense_dim(), "D values");
  TORCH_CHECK(mat2_.dense_dim() == 0, "sparse_sparse_matmul_cpu: scalar values expected, got ", mat2_.dense_dim(), "D values");

  TORCH_CHECK(
      mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
           "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

  auto output = at::native::empty_like(mat1_);
  output.sparse_resize_and_clear_({mat1_.size(0), mat2_.size(1)}, mat1_.sparse_dim(), 0);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(mat1_.scalar_type(), "sparse_matmul", [&] {
    sparse_matmul_kernel<scalar_t>(output, mat1_.coalesce(), mat2_.coalesce());
  });
  return output;
}


} // namespace at::native
