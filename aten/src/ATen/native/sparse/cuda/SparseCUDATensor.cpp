#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/sparse/SparseUtils.h>

namespace at { namespace native {

SparseTensor& sparse_mask_out_cuda(SparseTensor& r, const Tensor& t, const SparseTensor& mask) {
  AT_CHECK(mask.is_coalesced(), "sparse_mask: mask is uncoalesced");
  AT_CHECK(mask.sizes().equals(t.sizes()), "sparse_mask: operands have incompatible sizes; self has size ",
      t.sizes(), " but mask has size ", mask.sizes());
  AT_ASSERT(t.is_cuda()); // dispatch argument
  AT_CHECK(mask.is_cuda(), "sparse_mask: expected 'mask' to be CUDA, but got CPU");
  AT_CHECK(r.is_cuda(), "sparse_mask: expected 'out' to be CUDA, but got CPU");
  AT_CHECK(_check_device({r, t, mask}),
      "sparse_mask: arguments are located on different devices; self is on device ", t.get_device(),
      ", mask is on device ", mask.get_device(), ", out is on device ", r.get_device());
  resize_as_sparse_(r, mask);
  if (mask._nnz() == 0) {
    return r.zero_();
  }
  LongTensor mask_indices = mask._indices();
  Tensor mask_values = mask._values();
  Tensor r_values = r._values().type().tensor(mask_values.sizes());
  _alias_into_sparse(r, mask_indices.clone(), r_values);
  _get_sparse_impl(r)->set_coalesced(mask.is_coalesced());
  _get_sparse_impl(r)->set_nnz(mask._nnz());

  LongTensor indices = at::zeros({mask._nnz()}, mask_indices.options());

  for (int64_t d = 0; d < mask._sparseDims(); d++) {
    indices.mul_(mask.size(d));
    // This used to use a buffer but I deoptimized it
    indices.add_(mask_indices.select(0, d));
  }

  std::vector<int64_t> view_size(1 + mask._denseDims());
  view_size[0] = -1;
  for (int64_t d = 0; d < mask._denseDims(); d++) {
    view_size[d + 1] = mask.size(mask._sparseDims() + d);
  }

  Tensor t_view = t.view(view_size);
  // TODO: Re-audit this; it used to be an indexSelect directly into r_values
  at::index_select_out(r_values, t_view, 0, indices);

  return r;
}

SparseTensor sparse_mask_cuda(const Tensor& t, SparseTensorRef mask) {
  SparseTensor r = t.type().toSparse().tensor();
  sparse_mask_out_cuda(r, t, mask.tref);
  return r;
}

// Technically, this is not actually CUDA specific
int64_t get_device_sparse_cuda(const Tensor& self) {
  return self._values().get_device();
}

}} // namespace at::native
