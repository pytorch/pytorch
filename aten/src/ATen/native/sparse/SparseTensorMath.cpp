#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>

#include <TH/THBlasUtils.h>

namespace at { namespace native {

using namespace at::sparse;

// --------------------------------------------------------------------
// Utility functions
// --------------------------------------------------------------------

namespace {
  LongTensor _to_csr(const int64_t* indices, int64_t dim, int64_t nnz) {
    int64_t h, i, hp0, hp1;
    LongTensor csr = native::zeros({dim + 1}, kLong);

    // TODO: eliminate this conditional when zero-size dims supported correctly
    if (nnz > 0) {
      auto csr_accessor = csr.accessor<int64_t, 1>();
      // Convert the sparse matrix to CSR format
#pragma omp parallel for private(i, h, hp0, hp1) schedule(static) if (nnz > 10000)
      for (i=0; i<nnz; i++) {
        hp0 = indices[i];
        hp1 = (i+1 == nnz) ?  dim : indices[i+1];
        if (hp0 != hp1) for (h = hp0; h < hp1; h++) {
          csr_accessor[h+1] = i+1;
        }
      }
    }
    return csr;
  }

}

// --------------------------------------------------------------------
// zero_(SparseTensor)
// --------------------------------------------------------------------

// hummu hummu
SparseTensor& zero_sparse_(SparseTensor& self) {
  AT_ASSERT(self.is_sparse());
  at::zeros_out(self, get_sparse_impl(self)->sizes());
  return self._coalesced_(true);
}

// NB: Don't need zeros, zeros_like, already implemented in TensorFactories

// --------------------------------------------------------------------
// mul(SparseTensor, Scalar)
// --------------------------------------------------------------------

static Tensor scalar_tensor(Scalar s) {
  auto tensor = scalar_to_tensor(s);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

SparseTensor& mul_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const Tensor& value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(value.dim() == 0);

  if (is_same_tensor(r, t)) {
    r._values().mul_(value);
  } else {
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because mul_out takes Tensor&
    at::mul_out(r_values, t._values(), value);
    get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

SparseTensor& mul_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, Scalar value) {
  return mul_out_sparse_zerodim(r, t, scalar_tensor(value));
}

// --------------------------------------------------------------------
// log1p(SparseTensor)
// --------------------------------------------------------------------

// TODO: add in-place variant

SparseTensor& log1p_out_sparse(SparseTensor& r, const SparseTensor& t) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());

  if (is_same_tensor(r, t)) {
    // don't have in-place log1p for uncoalesced input because coalesce() is not in-place
    AT_CHECK(
      r.is_coalesced(), "log1p: in-place on uncoalesced tensors is not supported yet!");
  }
  else {
    copy_sparse_to_sparse_(r, t.coalesce());
  }
  r._values().log1p_();
  return r;
}

SparseTensor& log1p_sparse_(SparseTensor& t) {
  AT_CHECK(t.is_coalesced(), "log1p: in-place on uncoalesced tensors is not supported yet!");
  return log1p_out_sparse(t, t);
}

// --------------------------------------------------------------------
// pow(SparseTensor, Scalar)
// --------------------------------------------------------------------

// TODO: add in-place variant

SparseTensor& pow_out_sparse_scalar(SparseTensor& r, const SparseTensor& t_, Scalar value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t_.is_sparse());
  AT_CHECK(value.toDouble() != 0, "pow: cannot raise to zeroth power on sparse tensor; it would make the result tensor dense");

  // This coalesce is why we can't easily provide an inplace variant
  SparseTensor t = t_.coalesce();

  r.resize_as_(t);
  auto indices = r._indices();
  indices.resize_as_(t._indices());
  indices.copy_(t._indices());
  Tensor r_values = r._values(); // Sigh... needed because pow_out takes Tensor&
  at::pow_out(r_values, t._values(), value);
  get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
  return r._coalesced_(t.is_coalesced());
}

SparseTensor pow_sparse_scalar(const SparseTensor& t, Scalar value) {
  SparseTensor r = at::empty({0}, t.options());
  pow_out_sparse_scalar(r, t, value);
  return r;
}

// --------------------------------------------------------------------
// div(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& div_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const Tensor& value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(value.dim() == 0);

  if (is_same_tensor(r, t)) {
    r._values().div_(value);
  } else {
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because div_out takes Tensor&
    at::div_out(r_values, t._values(), value);
    get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

SparseTensor& div_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, Scalar value) {
  return div_out_sparse_zerodim(r, t, scalar_tensor(value));
}

// --------------------------------------------------------------------
// norm(SparseTensor, Scalar)
// --------------------------------------------------------------------

// Only supports floating point, FYI
Tensor norm_sparse(const SparseTensor& self, Scalar value) {
  AT_ASSERT(self.is_sparse());

  return self.coalesce()._values().norm(value);
}

// --------------------------------------------------------------------
// add(SparseTensor, SparseTensor, Scalar)  [broadcasts]
// --------------------------------------------------------------------

SparseTensor& add_out_sparse_cpu(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, Scalar value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(!t.is_cuda());  // the dispatch argument
  AT_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  AT_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  AT_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());

  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r, src, value);
  }

  AT_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz(), max_nnz = t_nnz + s_nnz;
  bool t_coalesced = t.is_coalesced(), s_coalesced = src.is_coalesced();
  int64_t sparse_dim = src.sparse_dim();
  LongTensor t_indices = t._indices();
  Tensor t_values = t._values();
  LongTensor src_indices = src._indices();
  Tensor s_values = src._values();
  LongTensor r_indices = at::empty({sparse_dim, max_nnz}, t_indices.options());
  Tensor r_values = new_values_with_size_of(s_values, max_nnz).zero_();
  r.resize_as_(src);
  get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);

  int64_t blockSize = r_values.stride(0);
  int64_t cmp, d;
  int64_t r_i = 0, t_i = 0, s_i = 0;

  // NB: relies on nnz tests above
  auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
  auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
  auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

  AT_DISPATCH_ALL_TYPES(
      t_values.type(), "cadd_sparse", [&] {
        scalar_t* t_values_ptr = t_values.data<scalar_t>();
        scalar_t* s_values_ptr = s_values.data<scalar_t>();
        scalar_t* r_values_ptr = r_values.data<scalar_t>();
        scalar_t cast_value = value.to<scalar_t>();
        while (t_i < t_nnz || s_i < s_nnz) {
          if (t_i >= t_nnz) {
            cmp = -1;
          } else if (s_i >= s_nnz) {
            cmp = 1;
          } else {
            cmp = 0;
            for (d = 0; d < sparse_dim; d++) {
              if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
                cmp = 1;
                break;
              }
              if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
                cmp = -1;
                break;
              }
            }
          }
          if (cmp >= 0) {
            for (d = 0; d < sparse_dim; d++) {
              r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
            }
            if (t_values.numel() > 0) {  // We add all elements from t_values to r_values only if t_values is not an empty tensor
              THBlas_axpy<scalar_t>(blockSize, 1,
                t_values_ptr + t_i * blockSize, 1,
                r_values_ptr + r_i * blockSize, 1);
            }
            t_i++;
          }
          if (cmp <= 0) {
            for (d = 0; d < sparse_dim; d++) {
              r_indices_accessor[d][r_i] = src_indices_accessor[d][s_i];
            }
            if (s_values.numel() > 0) {  // We add all elements from s_values to r_values only if s_values is not an empty tensor
              THBlas_axpy<scalar_t>(blockSize, cast_value,
                s_values_ptr + s_i * blockSize, 1,
                r_values_ptr + r_i * blockSize, 1);
            }
            s_i++;
          }
          r_i++;
        }
      }
  );

  get_sparse_impl(r)->set_nnz_and_narrow(r_i);
  // TODO: I think it may be possible to track inside the loop and
  // detect when we are uncoalesced (e.g., by observing that an
  // index goes backwards) which may be more precise than using the
  // coalesced flag here.  But this is easy.
  return r._coalesced_(t_coalesced && s_coalesced);
}

// --------------------------------------------------------------------
// add(Tensor, SparseTensor, Scalar)
//    formerly known as spcadd
// --------------------------------------------------------------------

template <typename scalar_t>
void add_dense_sparse_worker_cpu(Tensor& r, Scalar value, const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {
  int64_t k;

  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto values_accessor = values.accessor<scalar_t, 1>();

  scalar_t* r_ptr = r.data<scalar_t>();
  scalar_t cast_value = value.to<scalar_t>();

  #pragma omp parallel for private(k)
  for (k = 0; k < sparse._nnz(); k++) {
    int64_t index = r.storage_offset();
    for (int64_t d = 0; d < sparse.sparse_dim(); d++) {
      index += r.stride(d) * indices_accessor[d][k];
    }
    r_ptr[index] += cast_value * values_accessor[k];
  }
}

Tensor& add_out_dense_sparse_cpu(Tensor& r, const Tensor& dense, SparseTensorRef sparse__, Scalar value) {
  const SparseTensor& sparse_ = sparse__.tref;

  AT_ASSERT(!r.is_sparse());
  AT_ASSERT(!dense.is_sparse());
  AT_ASSERT(sparse_.is_sparse());

  AT_ASSERT(!dense.is_cuda()); // dispatch argument
  AT_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  AT_CHECK(!sparse_.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  AT_CHECK(dense.sizes().equals(sparse_.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", sparse_.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  r.resize_as_(dense);
  SparseTensor sparse = sparse_.coalesce();

  LongTensor indices = sparse._indices();
  Tensor values = sparse._values();
  int64_t nDim = dense.dim();
  int64_t nDimI = sparse.sparse_dim();

  if (!is_same_tensor(r, dense)) r.copy_(dense);
  if (sparse._nnz() == 0) return r;

  // accessors rely on nnz test
  if (nDim > nDimI) {
    auto indices_accessor = indices.accessor<int64_t, 2>();
    for (int64_t k = 0; k < sparse._nnz(); k++) {
      Tensor dstBuffer = r;
      for (int64_t d = 0; d < sparse.sparse_dim(); d++) {
        dstBuffer = dstBuffer.select(0, indices_accessor[d][k]);
      }
      Tensor srcBuffer = values.select(0, k);
      dstBuffer.add_(srcBuffer, value);
    }
  } else {
    AT_DISPATCH_ALL_TYPES(
        values.type(), "add_dense_sparse", [&] {
          add_dense_sparse_worker_cpu<scalar_t>(r, value, sparse, indices, values);
        });
  }
  return r;
}

// --------------------------------------------------------------------
// mul(SparseTensor, SparseTensor)  [broadcasts]
// --------------------------------------------------------------------

SparseTensor& mul_out_sparse_cpu(SparseTensor& r, const Tensor& t_, const Tensor& src_) {
  if (src_.dim() == 0) {
    return mul_out_sparse_zerodim(r, t_, src_);
  } else if (t_.dim() == 0) {
    return mul_out_sparse_zerodim(r, src_, t_);
  }

  AT_CHECK(t_.sizes().equals(src_.sizes()), "mul operands have incompatible sizes");
  AT_ASSERT(!t_.is_cuda()); // dispatch argument
  AT_CHECK(!r.is_cuda(), "mul: expected 'out' to be CPU tensor, but got CUDA tensor");
  AT_CHECK(!src_.is_cuda(), "mul: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  AT_CHECK(t_.sizes().equals(src_.sizes()), "mul: expected 'self' and 'other' to have same sizes, but ", t_.sizes(), " != ", src_.sizes());

  if (src_._nnz() == 0 || t_._nnz() == 0) {
    r.resize_as_(src_);
    return r.zero_();
  }

  SparseTensor t = t_.coalesce();
  SparseTensor src = src_.coalesce();

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
  int64_t max_nnz = std::min(t_nnz, s_nnz);  // multiply by zero is zero, and can be dropped
  int64_t sparse_dim = src.sparse_dim();
  LongTensor t_indices = t._indices();
  Tensor t_values = t._values();
  LongTensor src_indices = src._indices();
  Tensor s_values = src._values();
  LongTensor r_indices = at::empty({sparse_dim, max_nnz}, t_indices.options());
  Tensor r_values = new_values_with_size_of(t_values, max_nnz).zero_();
  r.resize_as_(src);
  get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);

  int64_t match, d;
  int64_t r_i = 0, t_i = 0, s_i = 0;

  // NB: relies on nnz test above
  auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
  auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
  auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

  // Check if we can find matching indices, and if so, write an
  // entry to the result indices vector.  Returns true if matching
  // indices were found.
  auto index_preamble = [&]() {
    match = 1;
    for (d = 0; d < sparse_dim; d++) {
      if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
        t_i++;
        match = 0;
        break;
      }
      if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
        s_i++;
        match = 0;
        break;
      }
    }
    if (!match) return false;
    for (d = 0; d < sparse_dim; d++) {
      r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
    }
    return true;
  };

  if (t_values.dim() > 1) {
    while (t_i < t_nnz && s_i < s_nnz) {
      if (!index_preamble()) continue;
      r_values.select(0, r_i).addcmul_(t_values.select(0, t_i), s_values.select(0, s_i));
      r_i++;
      t_i++;
      s_i++;
    }
  } else {
    AT_DISPATCH_ALL_TYPES(
        r_values.type(), "mul_out_sparse", [&] {
          auto r_accessor = r_values.accessor<scalar_t, 1>();
          auto t_accessor = t_values.accessor<scalar_t, 1>();
          auto s_accessor = s_values.accessor<scalar_t, 1>();

          while (t_i < t_nnz && s_i < s_nnz) {
            if (!index_preamble()) continue;
            r_accessor[r_i] = t_accessor[t_i] * s_accessor[s_i];
            r_i++;
            t_i++;
            s_i++;
          }
        }
    );
  }

  get_sparse_impl(r)->set_nnz_and_narrow(r_i);
  return r._coalesced_(true);
}

// --------------------------------------------------------------------
// addmm(Tensor, SparseTensorRef, Tensor, Scalar, Scalar)  [broadcasts]
// --------------------------------------------------------------------

// NB: OMP pragmas have to get their own functions; can't put them in lambdas
template <typename scalar_t>
void s_addmm_out_sparse_dense_worker(int64_t nnz, int64_t dim_i, int64_t dim_j, int64_t dim_k, Tensor& r, Scalar beta, const Tensor& t, Scalar alpha, const Tensor& csr, const Tensor& indices, const Tensor& values, const Tensor& dense) {
  int64_t h, i;

  // r_ = alpha * sparse * dense
  scalar_t cast_alpha = alpha.to<scalar_t>();
  scalar_t cast_beta = beta.to<scalar_t>();
  if (cast_beta == 0) {
    r.zero_();
  } else if (cast_beta == 1) {
    if (!is_same_tensor(r, t)) {
      r.copy_(t);
    }
  } else {
    at::mul_out(r, t, scalar_to_tensor(beta));
  }

  auto csr_accessor = csr.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();

  auto values_accessor = values.accessor<scalar_t, 1>();
  scalar_t* dense_ptr = dense.data<scalar_t>();
  scalar_t* r_ptr = r.data<scalar_t>();

  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t r_stride0 = r.stride(0);
  int64_t r_stride1 = r.stride(1);
#pragma omp parallel for private(h, i) schedule(static) if (nnz > 10000)
  for (h = 0; h < dim_i; h++) {
    int64_t i_start = csr_accessor[h];
    int64_t i_end = csr_accessor[h+1];
    for (i = i_start; i < i_end; i++) {
      scalar_t val = values_accessor[i];
      int64_t col = indices_accessor[1][i];
      if (col >= 0 && col < dim_j) {
        THBlas_axpy<scalar_t>(dim_k,
            cast_alpha * val,
            dense_ptr + col * dense_stride0, dense_stride1,
            r_ptr + h * r_stride0, r_stride1);
      } else {
        AT_ERROR("addmm: index out of bound: ", col, " not between 1 and ", dim_j);
      }
    }
  }
};

Tensor& s_addmm_out_sparse_dense_cpu(
    Tensor& r,
    const Tensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  // TODO: This error message seems awfully opaque
  AT_ASSERT(!t.is_cuda());
  AT_CHECK(!r.is_cuda(), "addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  AT_CHECK(!sparse_.is_cuda(), "addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  AT_CHECK(!dense.is_cuda(), "addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  AT_CHECK(sparse_.sparse_dim() == 2, "addmm: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  AT_CHECK(sparse_.dense_dim() == 0, "addmm: scalar values expected, got ", sparse_.dense_dim(), "D values");
  AT_CHECK(dense.dim() == 2, "addmm: matrices expected, got ", dense.dim(), "D tensor");

  SparseTensor sparse = sparse_.coalesce();

  // ixj * jxk = ixk
  int64_t dim_i = sparse.size(0);
  int64_t dim_j = sparse.size(1);
  int64_t dim_k = dense.size(1);

  AT_CHECK(dense.size(0) == dim_j,
      "addmm: Argument #3 (dense): Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  AT_CHECK(t.size(0) == dim_i,
      "addmm: Argument #1 (t): Expected dim 0 size ", dim_i, ", got ", t.size(0));
  AT_CHECK(t.size(1) == dim_k,
      "addmm: Argument #1 (t): Expected dim 1 size ", dim_k, ", got ", t.size(1));

  r.resize_({dim_i, dim_k});

  int64_t nnz        = sparse._nnz();

  if (nnz == 0) {
    at::mul_out(r, t, r.type().scalarTensor(beta));
    return r;
  }

  LongTensor indices = sparse._indices();
  Tensor values      = sparse._values();
  LongTensor csr = _to_csr(indices.data<int64_t>(), dim_i, nnz);

  AT_DISPATCH_ALL_TYPES(
      values.type(), "addmm_sparse_dense", [&] {
        s_addmm_out_sparse_dense_worker<scalar_t>(nnz, dim_i, dim_j, dim_k, r, beta, t, alpha, csr, indices, values, dense);
      }
  );

  return r;

}

Tensor s_addmm_sparse_dense_cpu(
    const Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  Tensor r = at::empty({0}, t.options());
  s_addmm_out_sparse_dense_cpu(r, t, sparse, dense, beta, alpha);
  return r;
}

Tensor& s_addmm_sparse_dense_cpu_(
    Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  return s_addmm_out_sparse_dense_cpu(t, t, sparse, dense, beta, alpha);
}


// --------------------------------------------------------------------
// hspmm(SparseTensor mat1, Tensor mat2)
// --------------------------------------------------------------------

SparseTensor& hspmm_out_sparse_cpu(SparseTensor& r, const SparseTensor& sparse_, const Tensor& dense) {
  // TODO: Make this a real argument
  Scalar alpha = 1;

  AT_ASSERT(!sparse_.is_cuda()); // dispatch argument
  AT_CHECK(!r.is_cuda(), "hspmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  AT_CHECK(!dense.is_cuda(), "hspmm: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  AT_CHECK(sparse_.sparse_dim() == 2,
      "hspmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  AT_CHECK(sparse_.dense_dim() == 0,
      "hspmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  AT_CHECK(dense.dim() == 2,
      "hspmm: Argument #3: matrices expected, got ", dense.dim(), "D tensor");

  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  AT_CHECK(dense.size(0) == k,
      "hspmm: Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));

  get_sparse_impl(r)->raw_resize_(1, 1, {m, n});

  SparseTensor sparse = sparse_.coalesce();

  int64_t nnz = sparse._nnz();

  if (nnz == 0) {
    r.zero_();
    return r;
  }

  LongTensor indices = at::empty({1, nnz}, at::initialTensorOptions().dtype(kLong));

  // Initialize the sparse matrix that will be used with spaddmm to send rows
  // from the dense matrix to rows of the output's value tensor
  SparseTensor newSparse = sparse.clone();
  LongTensor spIndices = newSparse._indices();
  LongTensor valueIndices = spIndices.select(0, 0);

  // Compute output indices
  auto valueIndices_accessor = valueIndices.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();

  int64_t i = -1, prevIdx = -1;
  for (int64_t j = 0; j < nnz; j++) {
    int64_t currIdx = valueIndices_accessor[j];
    if (currIdx != prevIdx) {
      indices_accessor[0][++i] = currIdx;
      prevIdx = currIdx;
    }
    valueIndices_accessor[j] = i;
  }
  int64_t outNnz = i + 1;
  indices.resize_({1, outNnz});
  Tensor values = at::empty({outNnz, n}, dense.options());

  std::vector<int64_t> new_size = get_sparse_impl(newSparse)->sizes().vec();
  new_size[0] = outNnz;
  get_sparse_impl(newSparse)->raw_resize_(get_sparse_impl(newSparse)->sparse_dim(), get_sparse_impl(newSparse)->dense_dim(), new_size);

  // Compute output values tensor with sparse * dense multiplication
  s_addmm_out_sparse_dense_cpu(values, values, newSparse, dense, 0, alpha);
  get_sparse_impl(r)->set_indices_and_values_unsafe(indices, values);

  return r;
}

SparseTensor hspmm_sparse_cpu(const SparseTensor& sparse, const Tensor& dense) {
  SparseTensor r = at::empty({0}, sparse.options());
  hspmm_out_sparse_cpu(r, sparse, dense);
  return r;
}

// --------------------------------------------------------------------
// sspaddmm
// --------------------------------------------------------------------

SparseTensor& _sspaddmm_out_cpu(
    SparseTensor& r,
    const SparseTensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  AT_ASSERT(!t.is_cuda()); // dispatch argument
  AT_CHECK(!r.is_cuda(), "sspaddmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  AT_CHECK(!sparse_.is_cuda(), "sspaddmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  AT_CHECK(!dense.is_cuda(), "sspaddmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  AT_CHECK(sparse_.sparse_dim() == 2,
      "sspaddmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  AT_CHECK(sparse_.dense_dim() == 0,
      "sspaddmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  AT_CHECK(dense.dim() == 2,
      "sspaddmm: Argument #2: matrices expected, got ", dense.dim(), "D tensor");

  SparseTensor sparse = sparse_.coalesce();

  // ixj * jxk = ixk
  int64_t dim_i = sparse.size(0);
  int64_t dim_j = sparse.size(1);
  int64_t dim_k = dense.size(1);

  // NB: This has to occur before the checks, because r may alias t.
  // See test_saddmm
  get_sparse_impl(r)->raw_resize_(2, 0, {dim_i, dim_k});

  AT_CHECK(dense.size(0) == dim_j,
      "sspaddmm: Argument #3: Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  AT_CHECK(t.size(0) == dim_i,
      "sspaddmm: Argument #1: Expected dim 0 size ", dim_i, ", got ", t.size(0));
  AT_CHECK(t.size(1) == dim_k,
      "sspaddmm: Argument #1: Expected dim 1 size ", dim_k, ", got ", t.size(1));

  int64_t nnz        = sparse._nnz();
  LongTensor indices = sparse._indices();
  Tensor values      = sparse._values();

  LongTensor csr = _to_csr(indices.data<int64_t>(), dim_i, nnz);

  int64_t t_nnz = t._nnz();
  int64_t r_nnz = nnz * dim_k + t_nnz;
  LongTensor newi = at::empty({2, r_nnz}, kLong);
  LongTensor newv = native::zeros({r_nnz}, values.options());

  if (t_nnz != 0) {
    LongTensor narrowi = newi.narrow(1, 0, t_nnz);
    Tensor narrowv = newv.narrow(0, 0, t_nnz);

    narrowi.copy_(t._indices());
    narrowv.copy_(t._values());
    newv.mul_(beta);
  }

  // sparse = sparse * dense
  int64_t p = t_nnz;

  auto csr_accessor = csr.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto newi_accessor = newi.accessor<int64_t, 2>();

  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t newv_stride0 = newv.stride(0);

  AT_DISPATCH_ALL_TYPES(
      values.type(), "sspmm", [&] {
        auto values_accessor = values.accessor<scalar_t, 1>();
        scalar_t* dense_ptr = dense.data<scalar_t>();
        scalar_t* newv_ptr = newv.data<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();

        for (int64_t h = 0; h < dim_i; h++) {
          int64_t i_start = csr_accessor[h];
          int64_t i_end = csr_accessor[h+1];
          for (int64_t i = i_start; i < i_end; i++) {
            scalar_t val = values_accessor[i];
            int64_t col = indices_accessor[1][i];
            if (col >= 0 && col < dim_j) {
              THBlas_axpy<scalar_t>(dim_k,
                  cast_alpha * val,
                  dense_ptr + col * dense_stride0, dense_stride1,
                  newv_ptr + p * newv_stride0, 1);
            } else {
              AT_ERROR("index out of bound. sspmm: ", col, " not between 1 and ", dim_j);
            }
          }
          // Fill up the indices with the right values
          if (i_start != i_end) {
            for (int64_t i = 0; i < dim_k; i++) {
              newi_accessor[0][p+i] = h;
              newi_accessor[1][p+i] = i;
            }
            p += dim_k;
          }
        }
      }
  );

  // to avoid a clone
  get_sparse_impl(r)->set_indices_and_values_unsafe(newi, newv);
  get_sparse_impl(r)->set_nnz_and_narrow(p);

  return r;
}

// sparse, sparse, sparse, dense, real, real -> sparse
Tensor& _sspaddmm_out_only_sparse(Tensor& result, const Tensor& self,
    const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  AT_ERROR("tensor.sspaddmm(...) can only be called on sparse tensors");
}

// sparse, dense -> sparse
Tensor smm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  at::sspaddmm_out(result, result, self, mat2, 0.0, 1.0);
  return result;
}

// sparse, sparse, dense, real, real -> sparse
Tensor sspaddmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    Scalar beta, Scalar alpha) {
  auto result = at::empty({0}, self.options());
  at::sspaddmm_out(result, self, mat1, mat2, beta, alpha);
  return result;
}

}} // namespace at::native
