#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at { namespace native {

using namespace at::sparse;

namespace {

LongTensor _to_csr(const int64_t* indices, int64_t dim, int64_t nnz) {
  /* Return the CSR indices from COO indices 

    `indices` is the COO indices array
    `dim` is the number of rows
    `nnz` is the number of non zeros in the original sparse tensor
  */
  LongTensor csr = native::zeros({dim + 1}, kLong);

  // TODO: eliminate this conditional when zero-size dims supported correctly
  if (nnz > 0) {
    auto csr_accessor = csr.accessor<int64_t, 1>();
    // Convert the sparse matrix to CSR format
    at::parallel_for(0, nnz, 10000, [&](int64_t start, int64_t end) {
      int64_t h, hp0, hp1;
      for (auto i = start; i < end; i++) {
        hp0 = indices[i];
        hp1 = (i+1 == nnz) ?  dim : indices[i+1];
        if (hp0 != hp1) for (h = hp0; h < hp1; h++) {
          csr_accessor[h+1] = i+1;
        }
      }
    });
  }
  return csr;
}


void expandptr(const int64_t n_row, const int64_t Ap[], int64_t Bi[]) {
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

int64_t __csr_matmult_maxnnz(
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

    This is an implementation of the SMMP algorithm:
     "Sparse Matrix Multiplication Package (SMMP)"

       Randolph E. Bank and Craig C. Douglas

     http://citeseer.ist.psu.edu/445062.html
     http://www.mgnet.org/~douglas/ccd-codes.html
  */
  std::vector<int64_t> mask(n_col, -1);
  const auto long_max = std::numeric_limits<int64_t>::max();

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
    TORCH_CHECK(row_nnz <= long_max - nnz, "nnz of the output is too large");
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

    This is an implementation of the SMMP algorithm:
     "Sparse Matrix Multiplication Package (SMMP)"

       Randolph E. Bank and Craig C. Douglas

     http://citeseer.ist.psu.edu/445062.html
     http://www.mgnet.org/~douglas/ccd-codes.html

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
      if (sums[head] != 0) {
        Cj[nnz] = head;
        Cx[nnz] = sums[head];
        nnz++;
      }

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

    The implementation is based on the paper  
    Bank and Douglas, 2001, Sparse Matrix Multiplication Package (SMPP)
  */

  auto mat1_indices_ = mat1._indices().contiguous();
  auto mat1_values = mat1._values().contiguous();
  Tensor mat1_indptr = _to_csr(mat1_indices_.data_ptr<int64_t>(), mat1.size(0), mat1._nnz());
  auto mat1_indices = mat1_indices_[1];

  auto mat2_indices_ = mat2._indices().contiguous();
  auto mat2_values = mat2._values().contiguous();
  Tensor mat2_indptr = _to_csr(mat2_indices_.data_ptr<int64_t>(), mat2.size(0), mat2._nnz());
  auto mat2_indices = mat2_indices_[1];
  
  auto M = mat1.size(0);
  auto K1 = mat1.size(1);

  auto K2 = mat2.size(0);
  auto N = mat2.size(1);

  auto major_axis = M;

  auto nnz = __csr_matmult_maxnnz(M, N, mat1_indptr.data_ptr<int64_t>(), mat1_indices.data_ptr<int64_t>(), 
      mat2_indptr.data_ptr<int64_t>(), mat2_indices.data_ptr<int64_t>());

  auto output_indices = output._indices();
  auto output_values = output._values();

  Tensor output_indptr = at::empty({major_axis + 1}, kLong);

  output_indices.resize_({2 * nnz});
  output_values.resize_(nnz);

  _csr_matmult(M, N, mat1_indptr.data_ptr<int64_t>(), mat1_indices.data_ptr<int64_t>(), mat1_values.data_ptr<scalar_t>(), 
  mat2_indptr.data_ptr<int64_t>(), mat2_indices.data_ptr<int64_t>(), mat2_values.data_ptr<scalar_t>(), 
  output_indptr.data_ptr<int64_t>(), output_indices.data_ptr<int64_t>() + nnz, output_values.data_ptr<scalar_t>());
  
  auto major_dim = output.size(0);

  Tensor major_indices = at::empty( {nnz}, kLong );
  expandptr(major_dim, output_indptr.data_ptr<int64_t>(), major_indices.data_ptr<int64_t>());

  std::memcpy(output_indices.data_ptr<int64_t>(), major_indices.data_ptr<int64_t>(), sizeof(int64_t) * nnz);
  output._indices().set_(output_indices.view({2, nnz}));

}

template <typename scalar_t>
Tensor fill_with(const Tensor& input, scalar_t fill_value){
  auto input_indices = input._indices();
  auto input_values = input._values();

  if (input.size(0) * input.size(1)  ==  input_values.numel()) {
    return input;
  }
  Tensor output = at::empty_like(input);

  auto output_indices = output._indices();
  auto output_values = output._values();
  auto n_rows = input.size(0);
  auto n_cols = input.size(1);
  auto matrix_size = n_rows * n_cols;
  
  auto new_nnz = input.size(0) * input.size(1) + input_values.numel(); 
  
  output_indices.resize_({2, new_nnz});
  output_values.resize_(new_nnz);

  auto input_indices_accessor = input_indices.accessor<int64_t, 2>();
  auto input_values_accessor = input_values.accessor<scalar_t, 1>();

  auto output_indices_accessor = output_indices.accessor<int64_t, 2>();
  auto output_values_accessor = output_values.accessor<scalar_t, 1>();

  at::parallel_for(0, matrix_size, 0, [&](int64_t start, int64_t end) {
    for (auto index = start; index < end; index++) {
      output_indices_accessor[0][index] = index / n_cols;
      output_indices_accessor[1][index] = index % n_cols;
      output_values_accessor[index] = fill_value;
    }
  });
  at::parallel_for(matrix_size, new_nnz, 0, [&](int64_t start, int64_t end) {
    for (auto index = start; index < end; index++) {
      int64_t j = index - matrix_size;
      output_indices_accessor[0][index] = input_indices_accessor[0][j];
      output_indices_accessor[1][index] = input_indices_accessor[1][j];
      output_values_accessor[index] = input_values_accessor[j] - fill_value;
    }
  });
  return output;
}

template <typename scalar_t, short grad_order>
void sparse_matmul_kernel_grad(Tensor& output, const Tensor& grad, const Tensor& x) {
  /* 
    Computes  the backward output  for matrix C = A*B.

    C = A@B 
      then 
    A_grad = C_grad @ B^T
    B_grad = A^T @ C_grad

    if grad_order == 1:
      output = x^T @ C_grad 
    else:
      output = C_grad @ x^T 
  */
  Tensor grad_updated = fill_with(grad, /*fill_value = */ scalar_t(1.0));
  if (grad_order == 1) {
    sparse_matmul_kernel<scalar_t>(output, x.transpose(0, 1).coalesce(), grad_updated.coalesce());
  } else if (grad_order == 0) {
    sparse_matmul_kernel<scalar_t>(output, grad_updated.coalesce(), x.transpose(0, 1).coalesce());
  }
}

} // end anonymous namespace

Tensor sparse_sparse_matmul_cpu(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
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
        output.scalar_type(), "sparse_matmul_kernel_grad_order", [&] {
          sparse_matmul_kernel_grad<scalar_t, 1>(output,  grad, var);
        });
  } else if (grad_order == 1) {
    std::vector<int64_t> size = {grad.size(0), var.size(0)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(size.size(), 0, size);

    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "sparse_matmul_kernel_grad_by_col", [&] {
          sparse_matmul_kernel_grad<scalar_t, 0>(output, grad, var);
        });
  }
  return output;
}

} // namespace native
} // namespace at
