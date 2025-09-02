#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/mul.h>
#endif

namespace at::native {

using namespace at::sparse;

Tensor& add_out_dense_sparse_mps(Tensor& out, const Tensor& dense, const SparseTensor& sparse, const Scalar& alpha);

Tensor& add_out_dense_sparse_mps(
    Tensor& out,
    const Tensor& dense,
    const SparseTensor& sparse,
    const Scalar& alpha) {
  TORCH_CHECK(dense.is_mps(),  "add: expected 'self' to be an MPS tensor, got ", dense.device());
  TORCH_CHECK(sparse.is_mps(), "add: expected 'other' to be an MPS tensor, got ", sparse.device());
  TORCH_CHECK(out.is_mps(),    "add: expected 'out' to be an MPS tensor, got ", out.device());
  TORCH_CHECK(dense.sizes().equals(sparse.sizes()),
              "add: expected 'self' and 'other' to have same size, but self has size ",
              dense.sizes(), " while other has size ", sparse.sizes(),
              " (FYI: dense-sparse addition does not currently support broadcasting)");

  const int64_t nnz = sparse._nnz();
  if (nnz == 0) {
    out.resize_as_(dense);
    out.copy_(dense);
    return out;
  }

  auto commonDtype = at::result_type(dense, sparse);
  TORCH_CHECK(canCast(commonDtype, out.scalar_type()),
              "Can't convert result type ", commonDtype, " to output ", out.scalar_type());

  Tensor r;
  const bool need_separate_buffer = out.is_same(dense) || (out.scalar_type() != commonDtype);
  if (need_separate_buffer) {
    r = at::empty(dense.sizes(), out.options().dtype(commonDtype));
  } else {
    r = out;
    r.resize_as_(dense);
  }

  Tensor dense_buffer = dense.to(commonDtype);
  if (!r.is_same(dense_buffer)) {
    r.copy_(dense_buffer);
  }

  Tensor indices = sparse._indices();
  Tensor values  = sparse._values().to(commonDtype);
  if (values.numel() == 0) {
    if (!out.is_same(r)) {
      out.resize_as_(dense);
      out.copy_(r);
    }
    return out;
  }

  const int64_t nDim  = r.dim();
  const int64_t nDimI = sparse.sparse_dim();
  TORCH_CHECK(nDimI >= 0 && nDimI <= nDim,
              "Invalid sparse_dim=", nDimI, " for dense tensor of dim ", nDim);

  Tensor indices1D = at::sparse::flatten_indices(indices, sparse.sizes()).contiguous();

  int64_t view_rows = 1;
  int64_t view_cols = 1;
  for (int64_t i = 0; i < nDimI; i++) {
    view_rows *= r.size(i);
  }
  for (int64_t i = nDimI; i < nDim; i++) {
    view_cols *= r.size(i);
  }

  if (view_cols == 1) {
    Tensor r_flat = r.reshape({view_rows});
    Tensor values_1d  = values.reshape({nnz});
    r_flat.index_add_(0, indices1D, values_1d, alpha);
  } else {
    Tensor r_view = r.view({view_rows, view_cols});
    Tensor values_2d  = values.reshape({nnz, view_cols});
    r_view.index_add_(0, indices1D, values_2d, alpha);
  }

  if (!out.is_same(r)) {
    out.resize_as_(dense);
    out.copy_(r);
  }
  return out;
}


SparseTensor& add_out_sparse_mps(const SparseTensor& self,
                                 const SparseTensor& other,
                                 const Scalar& alpha,
                                 SparseTensor& out) {
  TORCH_CHECK(other.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  TORCH_CHECK(self.is_mps(),  "add: expected 'self' to be MPS, but got ", self.device());
  TORCH_CHECK(other.is_mps(), "add: expected 'other' to be MPS, but got ", other.device());
  TORCH_CHECK(out.is_mps(),   "add: expected 'out' to be MPS, but got ", out.device());
  if (!self.is_sparse()) {
    return add_out_dense_sparse_mps(out, self, other, alpha);
  }
  auto commonDtype = at::result_type(self, other);
  TORCH_CHECK(canCast(commonDtype, out.scalar_type()),
              "Can't convert result type ", commonDtype, " to output ", out.scalar_type());

  TORCH_CHECK(self.sizes().equals(other.sizes()),
              "add: expected 'self' and 'other' to have same size, but ", self.sizes(), " != ", other.sizes());

  if (other._nnz() == 0) {
    out.resize_as_(self);
    Tensor vals = self._values();
    if (vals.scalar_type() != out.scalar_type()) {
      vals = vals.to(out.scalar_type());
    }
    alias_into_sparse(out, self._indices(), vals);
    out._coalesced_(self.is_coalesced());
    return out;
  }

  if (self._nnz() == 0) {
    out.resize_as_(other);
    Tensor vals = other._values();
    if (!alpha.isIntegral(false) || alpha.to<double>() != 1.0) {
      vals = at::mul(vals, alpha);
    }
    if (vals.scalar_type() != out.scalar_type()) {
      vals = vals.to(out.scalar_type());
    }
    alias_into_sparse(out, other._indices(), vals);
    out._coalesced_(other.is_coalesced());
    return out;
  }

  TORCH_CHECK(is_same_density(self, other),
              "add: expected 'self' and 'other' to have same density, but 'self' has ",
              self.sparse_dim(), " sparse dimensions while 'other' has ", other.sparse_dim(), " sparse dimensions");

  Tensor t_indices_ = self._indices();
  Tensor s_indices_ = other._indices();

  Tensor t_values_ = self._values().to(commonDtype);
  Tensor s_values_ = other._values().to(commonDtype);
  if (!alpha.isIntegral(false) || alpha.to<double>() != 1.0) {
    s_values_ = at::mul(s_values_, alpha);
  }

  Tensor r_indices_ = at::cat({t_indices_, s_indices_}, 1);
  Tensor r_values_  = at::cat({t_values_,  s_values_ }, 0);

  SparseTensor tmp = empty({0}, out.options().dtype(commonDtype));
  tmp.resize_as_(other);
  alias_into_sparse(tmp, r_indices_, r_values_);
  tmp = _coalesce_sparse_mps(tmp);

  out.resize_as_(other);
  Tensor out_vals = tmp._values();
  if (out.scalar_type() != commonDtype) {
    out_vals = out_vals.to(out.scalar_type());
  }
  alias_into_sparse(out, tmp._indices(), out_vals);
  out._coalesced_(tmp.is_coalesced());

  return out;
}

} // namespace at::native