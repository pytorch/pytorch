#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/SparseTensorUtils.h>

namespace at { namespace native {

using namespace at::sparse;

SparseTensor& sparse_mask_out_cuda(SparseTensor& r, const Tensor& t, const SparseTensor& mask) {
  AT_CHECK(mask.is_coalesced(), "sparse_mask: mask is uncoalesced");
  AT_CHECK(mask.sizes().equals(t.sizes()), "sparse_mask: operands have incompatible sizes; self has size ",
      t.sizes(), " but mask has size ", mask.sizes());
  AT_ASSERT(t.is_cuda());  // dispatch argument
  AT_CHECK(mask.is_cuda(), "sparse_mask: expected 'mask' to be CUDA, but got CPU");
  AT_CHECK(r.is_cuda(), "sparse_mask: expected 'out' to be CUDA, but got CPU");
  AT_CHECK(check_device({r, t, mask}),
      "sparse_mask: arguments are located on different devices; self is on device ", t.get_device(),
      ", mask is on device ", mask.get_device(), ", out is on device ", r.get_device());
  resize_as_sparse_(r, mask);
  if (mask._nnz() == 0) {
    return r.zero_();
  }
  LongTensor mask_indices = mask._indices();
  Tensor mask_values = mask._values();
  Tensor r_values = at::empty(mask_values.sizes(), r._values().options());
  alias_into_sparse(r, mask_indices.clone(), r_values);
  r._coalesced_(mask.is_coalesced());
  if (t.numel() == 0) {  // if t is an empty tensor, there is no need to mask its elements
    return r;
  }

  // Get a flattened sparse indices, similar to NOTE [ Flatten Sparse Indices ].
  // Keeping this implementation because it is faster than flatten_indices()
  LongTensor indices = at::zeros({mask._nnz()}, mask_indices.options());
  for (int64_t d = 0; d < mask.sparse_dim(); d++) {
    indices.mul_(mask.size(d));
    // This used to use a buffer but I deoptimized it
    indices.add_(mask_indices.select(0, d));
  }

  std::vector<int64_t> view_size(1 + mask.dense_dim());
  view_size[0] = -1;
  for (int64_t d = 0; d < mask.dense_dim(); d++) {
    view_size[d + 1] = mask.size(mask.sparse_dim() + d);
  }

  Tensor t_view = t.view(view_size);
  // TODO: Re-audit this; it used to be an indexSelect directly into r_values
  at::index_select_out(r_values, t_view, 0, indices);

  return r;
}

SparseTensor sparse_mask_cuda(const Tensor& t, SparseTensorRef mask) {
  SparseTensor r = at::empty({0}, t.options().layout(kSparse));
  sparse_mask_out_cuda(r, t, mask.tref);
  return r;
}

}} // namespace at::native
