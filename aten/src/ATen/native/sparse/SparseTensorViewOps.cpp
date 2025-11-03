#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::native {

namespace {

Tensor view_as_complex_sparse_cpu(const Tensor& self) {
  auto new_sizes = self.sizes().vec();
  TORCH_CHECK(!new_sizes.empty(), "Input tensor must have one or more dimensions");
  TORCH_CHECK(new_sizes[new_sizes.size() - 1] == 2, "Tensor must have a last dimension of size 2");
  new_sizes.pop_back();

  auto values = self._values();
  auto indices = self._indices();
  auto ndim = indices.size(0);
  auto nnz = indices.size(1);

  auto new_indices = indices.slice(/*dim=*/0, /*start=*/0, /*end=*/ndim-1);
  auto last_dim_indices = indices.select(0, ndim-1);
  const auto complex_type = c10::toComplexType(self.scalar_type());
  auto complex_values = at::empty(nnz, values.options().dtype(complex_type));

  // Create a flattened hash for sorting
  Tensor flatten_indices = at::sparse::flatten_indices(new_indices, new_sizes);

  // Sort columns by new_indices and reorder indices/values
  auto sort_result = flatten_indices.sort();
  Tensor sorted_flatten_indices = std::get<0>(sort_result);
  Tensor sort_perm = std::get<1>(sort_result);
  new_indices = new_indices.index_select(1, sort_perm);
  last_dim_indices = last_dim_indices.index_select(0, sort_perm);
  values = values.index_select(0, sort_perm);

  auto flatten_indices_accessor = sorted_flatten_indices.accessor<int64_t, 1>();
  auto last_dim_indices_accessor = last_dim_indices.accessor<int64_t, 1>();

  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, values.scalar_type(), "view_as_complex_sparse_cpu", [&] {
    using complex_t = c10::complex<scalar_t>;
    std::unordered_set<int64_t> skip_cols;
    std::vector<int64_t> keep_cols;

    auto values_accessor = values.accessor<scalar_t, 1>();
    auto complex_accessor = complex_values.accessor<complex_t, 1>();

    for (int64_t i=0; i<nnz; i++) {
      if (skip_cols.find(i) != skip_cols.end()) {
        continue;
      }
      keep_cols.push_back(i);
      scalar_t val = values_accessor[i];
      complex_t result = last_dim_indices_accessor[i] == 0 ? complex_t(val, 0) : complex_t(0, val);

      // check if any indices are the same and sum them together.
      int64_t j = i+1;
      while (j < nnz && flatten_indices_accessor[i] == flatten_indices_accessor[j]) {
        val = values_accessor[j];
        result += last_dim_indices_accessor[j] == 0 ? complex_t(val, 0) : complex_t(0, val);
        skip_cols.insert(j);
        j++;
      }

      complex_accessor[i] = result;
    }

    if (keep_cols.size() != static_cast<size_t>(nnz)) {
      at::Tensor keep_col_indices = at::tensor(keep_cols, indices.options());
      new_indices = new_indices.index_select(1, keep_col_indices);
      complex_values = complex_values.index_select(0, keep_col_indices);
    }
  });

  return at::_sparse_coo_tensor_with_dims_and_tensors(
      self.sparse_dim() - 1,
      self.dense_dim(),
      new_sizes,
      new_indices,
      complex_values,
      self.options().dtype(complex_type),
      true
  );
}

} // anonymous namespace

// Register CPU implementation for all CPU architectures
REGISTER_ARCH_DISPATCH(view_as_complex_sparse_stub, DEFAULT, &view_as_complex_sparse_cpu)
REGISTER_AVX512_DISPATCH(view_as_complex_sparse_stub, &view_as_complex_sparse_cpu)
REGISTER_AVX2_DISPATCH(view_as_complex_sparse_stub, &view_as_complex_sparse_cpu)
REGISTER_VSX_DISPATCH(view_as_complex_sparse_stub, &view_as_complex_sparse_cpu)
REGISTER_ZVECTOR_DISPATCH(view_as_complex_sparse_stub, &view_as_complex_sparse_cpu)
REGISTER_SVE256_DISPATCH(view_as_complex_sparse_stub, &view_as_complex_sparse_cpu)

} // namespace at::native
