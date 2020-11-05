#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at { namespace native {

using namespace at::sparse;

/*
    This is an implementation of the SMMP algorithm:
     "Sparse Matrix Multiplication Package (SMMP)"

      Randolph E. Bank and Craig C. Douglas
      https://doi.org/10.1007/BF02070824
*/
namespace {
void csr_to_coo(const int64_t n_row, const int64_t Ap[], int64_t Bi[]) {
  /* 
    Expands a compressed row pointer into a row indices array
    Inputs:
      `n_row` is the number of rows in `Ap`
      `Ap` is the row pointer
    
    Output: 
      `Bi` is the row indices 
  */
  for (int64_t i = 0; i < n_row; i++) {
    for (int64_t jj = Ap[i]; jj < Ap[i + 1]; jj++) {
      Bi[jj] = i;
    }
  }
}

int64_t _csr_matmult_maxnnz(
    const int64_t n_row,
    const int64_t n_col,
    const int64_t Ap[],
    const int64_t Aj[],
    const int64_t Bp[],
    const int64_t Bj[]) {
  /*
    Compute needed buffer size for matrix `C` in `C = A*B` operation.

    The matrices should be in proper CSR structure, and their dimensions
    should be compatible.
  */
  std::vector<int64_t> mask(n_col, -1);
  int64_t nnz = 0;
  for (int64_t i = 0; i < n_row; i++) {
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

template<class scalar_t>
void _csr_matmult(
    const int64_t n_row,
    const int64_t n_col,
    const int64_t Ap[],
    const int64_t Aj[],
    const scalar_t Ax[],
    const int64_t Bp[],
    const int64_t Bj[],
    const scalar_t Bx[],
    int64_t Cp[],
    int64_t Cj[],
    scalar_t Cx[]) {
  /* 
    Compute CSR entries for matrix C = A*B.

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
  std::vector<int64_t> next(n_col, -1);
  std::vector<scalar_t> sums(n_col, 0);

  int64_t nnz = 0;

  Cp[0] = 0;

  for (int64_t i = 0; i < n_row; i++) {
    int64_t head = -2;
    int64_t length = 0;

    int64_t jj_start = Ap[i];
    int64_t jj_end = Ap[i + 1];
    for (int64_t jj = jj_start; jj < jj_end; jj++) {
      int64_t j = Aj[jj];
      scalar_t v = Ax[jj];

      int64_t kk_start = Bp[j];
      int64_t kk_end = Bp[j + 1];
      for (int64_t kk = kk_start; kk < kk_end; kk++) {
        int64_t k = Bj[kk];

        sums[k] += v * Bx[kk];

        if (next[k] == -1) {
          next[k] = head;
          head = k;
          length++;
        }
      }
    }

    for (int64_t jj = 0; jj < length; jj++) {
      Cj[nnz] = head;
      Cx[nnz] = sums[head];
      nnz++;

      int64_t temp = head;
      head = next[head];

      next[temp] = -1; // clear arrays
      sums[temp] = 0;
    }

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
  auto K = mat1.size(1);
  auto N = mat2.size(1);

  auto mat1_indices_ = mat1._indices().contiguous();
  auto mat1_values = mat1._values().contiguous();
  LongTensor mat1_row_indices = mat1_indices_.select(0, 0);
  LongTensor mat1_col_indices = mat1_indices_.select(0, 1);

  Tensor mat1_indptr = coo_to_csr(mat1_row_indices.data_ptr<int64_t>(), M, mat1._nnz());

  auto mat2_indices_ = mat2._indices().contiguous();
  auto mat2_values = mat2._values().contiguous();
  LongTensor mat2_row_indices = mat2_indices_.select(0, 0);
  LongTensor mat2_col_indices = mat2_indices_.select(0, 1);

  Tensor mat2_indptr = coo_to_csr(mat2_row_indices.data_ptr<int64_t>(), K, mat2._nnz()); 

  auto nnz = _csr_matmult_maxnnz(M, N, mat1_indptr.data_ptr<int64_t>(), mat1_col_indices.data_ptr<int64_t>(), 
      mat2_indptr.data_ptr<int64_t>(), mat2_col_indices.data_ptr<int64_t>());

  auto output_indices = output._indices();
  auto output_values = output._values();

  Tensor output_indptr = at::empty({M + 1}, kLong);

  output_indices.resize_({2, nnz});
  output_values.resize_(nnz);

  LongTensor output_row_indices = output_indices.select(0, 0);
  LongTensor output_col_indices = output_indices.select(0, 1);

  _csr_matmult(M, N, mat1_indptr.data_ptr<int64_t>(), mat1_col_indices.data_ptr<int64_t>(), mat1_values.data_ptr<scalar_t>(), 
  mat2_indptr.data_ptr<int64_t>(), mat2_col_indices.data_ptr<int64_t>(), mat2_values.data_ptr<scalar_t>(), 
  output_indptr.data_ptr<int64_t>(), output_col_indices.data_ptr<int64_t>(), output_values.data_ptr<scalar_t>());
  
  csr_to_coo(M, output_indptr.data_ptr<int64_t>(), output_row_indices.data_ptr<int64_t>());
}

template <typename scalar_t, short fill_with_ones_order>
void sparse_matmul_with_ones_kernel(Tensor& result, const Tensor& x) {
  auto x_indices = x._indices().contiguous();
  auto x_values = x._values().contiguous();

  auto nnz_a = x_values.size(0);

  auto indices_accessor = x_indices.accessor<int64_t, 2>();
  auto rows = indices_accessor[0];
  auto cols = indices_accessor[1];
  auto values_accessor = x_values.accessor<scalar_t, 1>();

  auto result_indices = result._indices();
  auto result_values = result._values();

  auto n = result.size(0);
  auto m = result.size(1);
  auto size = fill_with_ones_order ? n : m;
  auto inner_size = fill_with_ones_order ? m : n;

  auto indices = (fill_with_ones_order ? rows : cols);

  std::vector<scalar_t> scalar_values(size, static_cast<scalar_t>(0));
  for (int64_t i = 0; i < nnz_a; i++) {
    for (int64_t index = 0; index < size; index++) {
      if (indices[i] == index) {
        scalar_values[index] += values_accessor[i];
      }
    }
  }

  int64_t index = 0;
  std::map<std::pair<int64_t, int64_t>, scalar_t> d;

  for (int64_t curr_index = 0; curr_index < scalar_values.size();
       curr_index++) {
    if (scalar_values[curr_index] != static_cast<scalar_t>(0)) {
      for (int64_t mat2_index = 0; mat2_index < inner_size; mat2_index++) {
        std::pair<int64_t, int64_t> current_index;
        if (fill_with_ones_order) {
          current_index = std::make_pair(curr_index, mat2_index);
        } else {
          current_index = std::make_pair(mat2_index, curr_index);
        }
        d[current_index] += scalar_values[curr_index];
        index++;
      }
    }
  }
  int64_t nnz_result = d.size();

  std::vector<int64_t> values_size = {nnz_result};
  result_indices.resize_({2, nnz_result});
  result_values.resize_(values_size);

  auto result_indices_accessor = result_indices.accessor<int64_t, 2>();
  auto result_values_accessor = result_values.accessor<scalar_t, 1>();

  index = 0;
  for (auto kv : d) {
    auto idx = kv.first;
    result_indices_accessor[0][index] = idx.first;
    result_indices_accessor[1][index] = idx.second;
    result_values[index] = kv.second;
    index++;
  } 
}

template <typename scalar_t, short grad_order>
void sparse_matmul_kernel_grad(Tensor& output, const Tensor& grad_, const Tensor& x) {
  /* 
    Computes  the backward output  for matrix C = A@B.

    C = A@B 
      then 
    A_grad = C_grad @ B^T
    B_grad = A^T @ C_grad

    A matrix multiplication of two sparse tensors A and B with fill values a and b,
    respectively, then the matmul operation can be expanded as follows:
    matmul(A, B) = matmul(A - a + a, B - b + b)
                 = matmul(A - a, B - b) + a * matmul(ones_like(A), B) + b * matmul(A, ones_like(B))

    if grad_order == 0:
      A_grad = C_grad @ x^T, where x is B 
  
      A_grad = matmul(C_grad - 1, x^T) + matmul(C_grad, ones_like(x^T)) 

    else:
      B_grad = x^T @ C_grad, where x is A 

      B_grad = matmul(x^T, C_grad - 1) + matmul(ones_like(x^T), C_grad) 
  */

  auto grad = grad_.coalesce();
  auto xt = x.transpose(0, 1).coalesce();

  Tensor output_ones = at::native::empty_sparse(output.sizes(), output.options());
  sparse_matmul_with_ones_kernel<scalar_t, grad_order>(output_ones, xt);

  auto grad_values = grad._values().contiguous().clone();
  auto grad_values_accessor = grad_values.accessor<scalar_t, 1>();
  at::parallel_for(0, grad_values.size(0), 0, [&](int64_t start, int64_t end) {
    for (auto index = start; index < end; index++) {
      grad_values_accessor[index] -= static_cast<scalar_t>(1);
    }
  });
  Tensor grad_updated = at::native::empty_like(grad);
  alias_into_sparse(grad_updated, grad._indices(), grad_values);

  if (grad_order == 1) {
    sparse_matmul_kernel<scalar_t>(output, xt, grad_updated);
  } else if (grad_order == 0) {
    sparse_matmul_kernel<scalar_t>(output, grad_updated, xt);
  }
  native::add_sparse_(output, output_ones);
}

} // end anonymous namespace

Tensor sparse_sparse_matmul_cpu(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
  TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
  TORCH_CHECK(mat1_.dim() == 2);
  TORCH_CHECK(mat2_.dim() == 2);

  TORCH_CHECK(
      mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
           "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

  Tensor output = at::native::empty_sparse({mat1_.size(0), mat2_.size(1)}, mat1_.options());

  AT_DISPATCH_FLOATING_TYPES(mat1_.scalar_type(), "sparse_matmul", [&] {
    sparse_matmul_kernel<scalar_t>(output, mat1_.coalesce(), mat2_.coalesce());
  });
  return output;
}

Tensor sparse_sparse_matmul_backward_cpu(
    const Tensor& grad,
    const Tensor& var,
    int64_t grad_order) {
  TORCH_CHECK(
      grad_order == 0 || grad_order == 1,
      ": grad_order not in [0, 1] at sparse_sparse_matmul_backward_cpu function");
  Tensor output = at::native::empty_like(var);
  if (grad_order == 0) {
    std::vector<int64_t> size = {var.size(1), grad.size(1)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(size.size(), 0, size);
    
    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "sparse_matmul_kernel_grad", [&] {
          sparse_matmul_kernel_grad<scalar_t, 1>(output,  grad, var);
        });
  } else if (grad_order == 1) {
    std::vector<int64_t> size = {grad.size(0), var.size(0)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(size.size(), 0, size);

    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "sparse_matmul_kernel_grad", [&] {
          sparse_matmul_kernel_grad<scalar_t, 0>(output, grad, var);
        });
  }
  return output;
}

} // namespace native
} // namespace at
