#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at {
namespace native {

// helper
namespace {

std::vector<int64_t> indices2csr(
    const TensorAccessor<int64_t, 1>& indices,
    int64_t dim) {
  /* Return the new indices using csr sparse format.

    `indices` original indices tensor using coo sparse format.
    `dim` is the dimension of the sparse part of a sparse tensor.
  */
  auto nnz = indices.size(0);
  std::vector<int64_t> csr(dim + 1, int64_t(0));
  int64_t last_i = 0;
  for (int64_t index = 0; index < nnz; index++) {
    int64_t i = indices[index];
    if (i != last_i) {
      for (int64_t j = last_i; j < i + 1; j++) {
        csr[j + 1] = csr[last_i + 1];
      }
      last_i = i;
    }
    csr[last_i + 1] += 1;
  }
  for (int64_t j = last_i; j < dim; j++) {
    csr[j + 1] = csr[last_i + 1];
  }
  return csr;
}

template <typename scalar_t, bool is_coalesced>
void sparse_matmul_kernel(
    Tensor& result,
    const Tensor& mat1,
    const Tensor& mat2) {
  /*
    See test/test_sparse.py:test_sparse_matmul:sparse_mm for the Python
    prototype of the sparse matmult algorithm that this implementation
    is based on.

    This kernel implements a matrix multiplication of the sparse matrix :attr:`mat1`
    and sparse matrix :attr:`mat2`. This implementation uses the right operand in CSR format 
    and collects the elements of the result into a dictionary.

    
  */

  auto indices_a = mat1._indices().contiguous();
  auto values_a = mat1._values().contiguous();

  auto indices_b = mat2._indices().contiguous();
  auto values_b = mat2._values().contiguous();

  auto nnz_a = values_a.size(0);
  auto nnz_b = values_b.size(0);
  auto indices_accessor_a = indices_a.accessor<int64_t, 2>();
  auto indices_accessor_b = indices_b.accessor<int64_t, 2>();

  auto values_accessor_a = values_a.accessor<scalar_t, 1>();
  auto values_accessor_b = values_b.accessor<scalar_t, 1>();

  auto result_indices = result._indices();
  auto result_values = result._values();

  std::map<std::pair<int64_t, int64_t>, scalar_t> d;
  for (int64_t n1 = 0; n1 < nnz_a; n1++) {
    if constexpr (is_coalesced) {
      auto r2 = indices2csr(indices_accessor_b[0], mat2.size(0));
      for (int64_t n2 = r2[indices_accessor_a[1][n1]]; 
          n2 < r2[indices_accessor_a[1][n1] + 1];
          n2++) {
        auto current_index = std::make_pair(
            indices_accessor_a[0][n1], indices_accessor_b[1][n2]);
        d[current_index] += values_accessor_a[n1] * values_accessor_b[n2];
      }
    } else {
      for (int64_t n2 = 0; n2 < nnz_b; n2++) {
        if (indices_accessor_b[0][n2] == indices_accessor_a[1][n1]) {
          auto current_index = std::make_pair(
              indices_accessor_a[0][n1], indices_accessor_b[1][n2]);
          d[current_index] += values_accessor_a[n1] * values_accessor_b[n2];
        }
      }
    }
  }
  int64_t nnz_result = d.size();
  std::vector<int64_t> values_size = {nnz_result};
  result_indices.resize_({2, nnz_result});
  result_values.resize_(values_size);

  auto result_indices_accessor = result_indices.accessor<int64_t, 2>();
  auto result_values_accessor = result_values.accessor<scalar_t, 1>();

  int64_t index = 0;
  for (auto kv : d) {
    auto idx = kv.first;
    result_indices_accessor[0][index] = idx.first;
    result_indices_accessor[1][index] = idx.second;
    result_values[index] = kv.second;
    index++;
  } 
}

template <typename scalar_t, bool grad_by_row>
void sparse_matmul_kernel_grad(Tensor& result, const Tensor& x) {
  auto x_indices = x._indices().contiguous();
  auto x_values = x._values().contiguous();

  auto nnz_a = x_values.size(0);

  auto indices_accessor = x_indices.accessor<int64_t, 2>();
  auto rows = indices_accessor[1];
  auto cols = indices_accessor[0];
  auto values_accessor = x_values.accessor<scalar_t, 1>();

  auto result_indices = result._indices();
  auto result_values = result._values();

  auto n = result.size(0);
  auto m = result.size(1);
  auto size = grad_by_row ? n : m;
  auto inner_size = grad_by_row ? m : n;

  auto indices = (grad_by_row ? rows : cols);

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
        if (grad_by_row) {
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

} // end anonymous namespace

Tensor sparse_matmul(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
  TORCH_CHECK(mat1_.dim() == 2);
  TORCH_CHECK(mat2_.dim() == 2);
  TORCH_CHECK(mat1_.size(1) == mat2_.size(0), "Incompatible matrices");
  TORCH_CHECK(
      mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
           "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

  auto mat1 = mat1_.coalesce();
  auto mat2 = mat2_.coalesce();
  Tensor output = at::native::empty_sparse(
      {mat1_.size(0), mat2_.size(1)}, mat1_.options());

  AT_DISPATCH_FLOATING_TYPES(mat1_.scalar_type(), "sparse_matmul", [&] {
    if (mat1.is_coalesced() && mat2.is_coalesced())
      sparse_matmul_kernel<scalar_t, true>(output, mat1, mat2);
    else
      sparse_matmul_kernel<scalar_t, false>(output, mat1, mat2);
  });
  return output;
}

Tensor sparse_matmul_backward_cpu(
    const Tensor& grad,
    const Tensor& var,
    int64_t mult_order) {
  TORCH_CHECK(
      mult_order == 0 || mult_order == 1,
      ": mult_order not in [0, 1] at sparse_matmul_backward_cpu function");
  Tensor output = at::native::empty_like(grad);
  if (mult_order == 0) {
    std::vector<int64_t> size = {var.size(1), grad.size(1)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(
        size.size(), 0, size);
    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "sparse_matmul_kernel_grad_by_row", [&] {
          sparse_matmul_kernel_grad<scalar_t, true>(output, var);
        });
  } else if (mult_order == 1) {
    std::vector<int64_t> size = {grad.size(0), var.size(0)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(
        size.size(), 0, size);

    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "sparse_matmul_kernel_grad_by_col", [&] {
          sparse_matmul_kernel_grad<scalar_t, false>(output, var);
        });
  }
  return output;
}

} // namespace native
} // namespace at
