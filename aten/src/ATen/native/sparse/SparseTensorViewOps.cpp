#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::native {

namespace {

Tensor view_as_complex_sparse_cpu(const Tensor& self) {
//    return self.clone();
  auto new_sizes = self.sym_sizes().vec();
  TORCH_CHECK(!new_sizes.empty(), "Input tensor must have one or more dimensions");
  TORCH_CHECK(new_sizes[new_sizes.size() - 1] == 2, "Tensor must have a last dimension of size 2");
  new_sizes.pop_back();

  auto values = self._values();
  auto indices = self._indices();
  auto ndim = indices.size(0);
  auto nnz = indices.size(1);

  auto new_indices = indices.slice(/*dim=*/0, /*start=*/0, /*end=*/ndim-1);
  const auto complex_type = c10::toComplexType(self.scalar_type());
  auto complex_values = at::zeros(nnz, values.options().dtype(complex_type));
  auto last_dim = ndim - 1;

  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "view_as_complex_sparse_cpu", [&] {
    using complex_t = c10::complex<scalar_t>;
    const complex_t I(scalar_t(0), scalar_t(1));
    std::unordered_set<int64_t> skip_cols;
    std::vector<int64_t> keep_cols;

    for (int64_t i=0; i<nnz; i++) {
      if (skip_cols.find(i) != skip_cols.end()) {
        continue;
      }
      keep_cols.push_back(i);
      complex_values[i] = indices[last_dim][i].item<int64_t>() == 0 ? values[i] : values[i].mul(I);
      auto coli = new_indices.select(/*dim=*/1, /*index=*/i);

      for (int64_t j=i+1; j<nnz; j++) {
        auto colj = new_indices.select(/*dim=*/1, /*index=*/j);
        if (at::equal(coli, colj)) {
          complex_values[i] += indices[last_dim][j].item<int64_t>() == 0 ? values[j] : values[j].mul(I);
          skip_cols.insert(j);
        }
      }
    }

    if (keep_cols.size() != static_cast<size_t>(nnz)) {
      at::Tensor keep_col_indices = at::tensor(keep_cols, indices.options());
      new_indices = new_indices.index_select(1, keep_col_indices);
      complex_values = complex_values.index_select(0, keep_col_indices);
    }
  });

  return at::_sparse_coo_tensor_with_dims_and_tensors_symint(
      self.sparse_dim() - 1,
      self.dense_dim(),
      new_sizes,
      new_indices,
      complex_values,
      self.options().dtype(complex_type),
      self.is_coalesced()
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