#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

// helper
namespace {

template<typename scalar_t>
std::vector<int64_t> sort_indices(const TensorAccessor<scalar_t, 1> &v) {
  std::vector<int64_t> indices(v.size(0));
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(), [&](const scalar_t& i1, const scalar_t& i2) {
    return v[i1] < v[i2];
  });
  return indices;
}

template<size_t N>
std::vector<int64_t> indices2csr(const TensorAccessor<int64_t,N>& indices, int64_t dim) {
  auto nnz = indices.size(0);
  std::vector<int64_t> csr(dim + 1, int64_t(0));
  int64_t last_i = 0;
  for (int64_t index = 0; index < nnz; index++) {
    int64_t i = indices[index]; 
    if (i == last_i) {
      csr[last_i + 1] += 1;
    } else {
      for (int64_t j = last_i; j < i + 1; j++) {
        csr[j + 1] = csr[last_i + 1];
      }
      last_i = i;
      csr[last_i + 1] += 1;
    }
  }
  for (int64_t j = last_i; j < dim; j++) {
    csr[j + 1] = csr[last_i + 1];
  } 
  return csr;
}

template<typename scalar_t>
void sparse_coalesce_matmul_kernel(Tensor& result, const Tensor& self, const Tensor& other) {
  auto n = self.size(0);
  auto p = self.size(1);
  auto indices_a = self._indices().contiguous();
  auto values_a = self._values().contiguous();

  auto m = other.size(1);
  auto indices_b = other._indices().contiguous();
  auto values_b = other._values().contiguous();

  auto nnz_a = values_a.size(0);
  auto nnz_b = values_b.size(0);
  auto indices_accessor_a = indices_a.accessor<int64_t, 2>();
  auto indices_accessor_b = indices_b.accessor<int64_t, 2>();

  auto values_accessor_a = values_a.accessor<scalar_t, 1>();
  auto values_accessor_b = values_b.accessor<scalar_t, 1>();

  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();
 
  std::map<std::pair<int64_t, int64_t>, scalar_t> d;
  if (self.is_coalesced() && other.is_coalesced()) {
    for (int64_t n1 = 0; n1 < nnz_a; n1++) {
      auto r2 = indices2csr(indices_accessor_b[0], other.size(0));
      for (int n2 = r2[indices_accessor_a[1][n1]]; n2 < r2[indices_accessor_a[1][n1] + 1]; n2++) {
        d[std::make_pair(indices_accessor_a[0][n1], indices_accessor_b[1][n2]) ] += values_accessor_a[n1] * values_accessor_b[n2];
      }    
    }
  } else {
    for (int64_t n1 = 0; n1 < nnz_a; n1++) {
      for (int n2 = 0; n2 < nnz_b; n2++) {
        if (indices_accessor_b[0][n2] == indices_accessor_a[1][n1]) {
            d[std::make_pair(indices_accessor_a[0][n1], indices_accessor_b[1][n2]) ] += values_accessor_a[n1] * values_accessor_b[n2];
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

template<typename scalar_t>
void sparse_matmul_kernel(Tensor& result, const Tensor& self, const Tensor& other) {
  auto n = self.size(0);
  auto p = self.size(1);
  auto indices_a = self._indices().contiguous();
  auto values_a = self._values().contiguous();

  auto m = other.size(1);
  auto indices_b = other._indices().contiguous();
  auto values_b = other._values().contiguous();

  auto nnz_a = values_a.size(0);
  auto nnz_b = values_b.size(0);
  auto indices_accessor_a = indices_a.accessor<int64_t, 2>();
  auto indices_accessor_b = indices_b.accessor<int64_t, 2>();

  auto values_accessor_a = values_a.accessor<scalar_t, 1>();
  auto values_accessor_b = values_b.accessor<scalar_t, 1>();

  auto map_a = sort_indices(indices_accessor_a[1]);
  auto map_b = sort_indices(indices_accessor_b[0]);

  std::vector<int64_t> b1 = std::initializer_list<int64_t>{0};
  for (int64_t i = 0; i < nnz_a - 1; i++) {
    auto idx = map_a[i];
    auto idx_1 = map_a[i+1];
    if (indices_accessor_a[1][idx] != indices_accessor_a[1][idx_1]) 
      b1.push_back(i+1);
  }
  b1.push_back(nnz_a);

  std::vector<int64_t> b2 = std::initializer_list<int64_t>{0};
  for (int64_t i = 0; i < nnz_b - 1; i++) {
    auto idx = map_b[i];
    auto idx_1 = map_b[i+1];
    if (indices_accessor_b[0][idx] != indices_accessor_b[0][idx_1]) 
      b2.push_back(i+1);
  }
  b2.push_back(nnz_b);

  int64_t nnz_result = 0;
  int64_t i = 0;
  int64_t j = 0;
  
  while (i < b1.size()-1 && j < b2.size()-1){
    auto b1i = map_a[b1[i]];
    auto b2j = map_b[b2[j]];
    if (indices_accessor_a[1][b1i] == indices_accessor_b[0][b2j]) {
      nnz_result += (b1[i + 1] - b1[i]) * (b2[j + 1] - b2[j]);
      i++;
      j++;
    } else {
      if (indices_accessor_a[1][b1i] < indices_accessor_b[0][b2j]) {
        i += 1;
      } else {
        j += 1;
      }
    }
  } 
  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();

  std::vector<int64_t> values_size = {nnz_result};

  result_indices.resize_({2, nnz_result});
  result_values.resize_(values_size);

  auto result_indices_accessor = result_indices.accessor<int64_t, 2>();
  auto result_values_accessor = result_values.accessor<scalar_t, 1>();
  i = 0;
  j = 0;
  int64_t index = 0;

  while (i < b1.size()-1 && j < b2.size()-1){
    auto b1i = map_a[b1[i]];
    auto b2j = map_b[b2[j]];
    if (indices_accessor_a[1][b1i] == indices_accessor_b[0][b2j]) {
      for (int64_t c1 = b1[i]; c1 < b1[i+1]; c1++){
        auto mc1 = map_a[c1];
        for (int64_t c2 = b2[j]; c2 < b2[j+1]; c2++){
          auto mc2 = map_b[c2];
          result_indices_accessor[0][index] = indices_accessor_a[0][mc1];
          result_indices_accessor[1][index] = indices_accessor_b[1][mc2];
          result_values[index] = values_accessor_a[mc1] * values_accessor_b[mc2];
          index++;
        }
      }
      i++;
      j++;
    } else if (indices_accessor_a[1][b1i] < indices_accessor_b[0][b2j]) {
      i += 1;
    } else {
      j += 1;
    }
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

  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();

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
  int64_t nnz_result = 0;
  for (int64_t index = 0; index < scalar_values.size(); index++) {
    // skip zeros in result
    if (scalar_values[index] != static_cast<scalar_t>(0)) {
      nnz_result += inner_size;
    }
  }
  std::vector<int64_t> values_size = {nnz_result};
  result_indices.resize_({2, nnz_result});
  result_values.resize_(values_size);

  auto result_indices_accessor = result_indices.accessor<int64_t, 2>();
  auto result_values_accessor = result_values.accessor<scalar_t, 1>();

  int64_t index = 0;
  for (int64_t curr_index = 0; curr_index < scalar_values.size();
       curr_index++) {
    if (scalar_values[curr_index] != static_cast<scalar_t>(0)) {
      for (int64_t other_index = 0; other_index < inner_size; other_index++) {
        if (grad_by_row) {
          result_indices_accessor[0][index] = curr_index;
          result_indices_accessor[1][index] = other_index;
        } else {
          result_indices_accessor[0][index] = other_index;
          result_indices_accessor[1][index] = curr_index;
        }
        result_values[index] = scalar_values[curr_index];
        index++;
      }
    }
  }
}



} // end anonymous namespace

Tensor sparse_matmul(const Tensor& self_, const Tensor& other_) {
  TORCH_INTERNAL_ASSERT(self_.is_sparse());
  TORCH_CHECK(self_.dim() == 2);
  TORCH_CHECK(other_.dim() == 2);
  TORCH_CHECK(self_.size(1) == other_.size(0), "Incompatible matrices");

  auto self = self_.coalesce();
  auto other = other_.coalesce();
  Tensor output = at::native::empty_sparse({self_.size(0), other_.size(1)}, self_.options());

  AT_DISPATCH_FLOATING_TYPES(
    self_.scalar_type(), "sparse_matmul", [&] {
      sparse_coalesce_matmul_kernel<scalar_t>(output, self, other);
    });
  return output;
}

Tensor sparse_matmul_backward_cpu(const Tensor& grad, const Tensor& var, int64_t mult_order){
  TORCH_CHECK(
      mult_order == 0 || mult_order == 1,
      "mult_order not in [0, 1] at sparse_matmul_backward_cpu function");
  Tensor output = at::native::empty_like(grad);
  if (mult_order == 0) {
    std::vector<int64_t> size = {var.size(1), grad.size(1)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(size.size(), 0, size);
    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "sparse_matmul_kernel_grad_by_row", [&] {
          sparse_matmul_kernel_grad<scalar_t, true>(output, var);
        });
  }
  else if (mult_order == 1) {
    std::vector<int64_t> size = {grad.size(0), var.size(0)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(size.size(), 0, size);

    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "sparse_matmul_kernel_grad_by_col", [&] {
          sparse_matmul_kernel_grad<scalar_t, false>(output, var);
        });
  }
  return output;
}

}} // namespace at::native