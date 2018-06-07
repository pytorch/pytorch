#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/native/BlasUtils.h>

namespace at { namespace native {

namespace {
  // TODO: Expose this for real in ATen, some day?
  // NB: Doesn't preserve data.
  Tensor _new_values_with_size_of(const Tensor& values, int64_t nnz) {
    if (values.dim() == 0) { // values tensor uninitialized
      return values.type().tensor({nnz});
    } else {
      std::vector<int64_t> size = values.sizes();
      size[0] = nnz;
      return values.type().tensor(size);
    }
  }
}

// Just for documentary purposes
using SparseTensor = Tensor;
using LongTensor = Tensor;

namespace {
  // TODO: This is a temporary stop-gap, to allow us to perform some private
  // functionality.  Our eventual plan is to fill out the PUBLIC API with
  // enough functions so that math functions don't need to rely on this.
  SparseTensorImpl* _get_sparse_impl(const SparseTensor& self) {
    if (!self.is_sparse()) AT_ERROR("_internal_get_SparseTensorImpl: not a sparse tensor");
    return static_cast<SparseTensorImpl*>(self.unsafeGetTensorImpl());
  }
}

// hummu hummu
SparseTensor& zero_sparse_(SparseTensor& self) {
  if (self._indices().dim()) {
    // TODO: To be fixed when we support zero-size dims
    self._indices().resize_({0});
  }
  if (self._values().dim()) {
    self._values().resize_({0});
  }
  _get_sparse_impl(self)->set_nnz(0);
  _get_sparse_impl(self)->set_coalesced(true); // NB: This is new
  return self;
}

// NB: Don't need zeros, zeros_like, already implemented in TensorFactories

// TODO: Where it would be helpful, de-out some of these functions.  They are
// written this way because that's how it was done in THS.

// TODO: put this into the public API
bool isSameTensor(const Tensor& lhs, const Tensor& rhs) {
  return lhs.unsafeGetTensorImpl() == rhs.unsafeGetTensorImpl();
}

SparseTensor& mul_out_sparse(SparseTensor& r, const SparseTensor& t, Scalar value) {
  if (isSameTensor(r, t)) {
    r._values().mul_(value);
  } else {
    r.resize_as_(t);
    r._indices().copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because mul_out takes Tensor&
    mul_out(r_values, t._values(), value);
    _get_sparse_impl(r)->set_nnz(t._nnz());
    _get_sparse_impl(r)->set_coalesced(t.is_coalesced());
  }
  return r;
}

SparseTensor& pow_out_sparse(SparseTensor& r, const SparseTensor& t, Scalar value) {
  AT_CHECK(value.toDouble() != 0, "cannot raise to zeroth power on sparse tensor; it would make the result tensor dense");

  if (isSameTensor(r, t)) {
    r._values().pow_(value);
  } else {
    r.resize_as_(t);
    r._indices().copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because pow_out takes Tensor&
    pow_out(r_values, t._values(), value);
    _get_sparse_impl(r)->set_nnz(t._nnz());
    _get_sparse_impl(r)->set_coalesced(t.is_coalesced());
  }
  return r;
}

SparseTensor pow_sparse(const SparseTensor& t, Scalar value) {
  SparseTensor r = t.type().tensor();
  pow_out_sparse(r, t, value);
  return r;
}

SparseTensor& div_out_sparse(SparseTensor& r, const SparseTensor& t, Scalar value) {
  if (isSameTensor(r, t)) {
    r._values().div_(value);
  } else {
    r.resize_as_(t);
    r._indices().copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because div_out takes Tensor&
    div_out(r_values, t._values(), value);
    _get_sparse_impl(r)->set_nnz(t._nnz());
    _get_sparse_impl(r)->set_coalesced(t.is_coalesced());
  }
  return r;
}

// Only supports floating point, FYI
Tensor norm_sparse(const SparseTensor& self, Scalar value) {
  return self.coalesce()._values().norm(value);
}

// Internal function, I guess?
bool is_same_density(const SparseTensor& self, const SparseTensor& src) {
  return self._dimI() == src._dimI() && self._dimV() == src._dimV();
}

SparseTensor& cadd_out_sparse_cpu(SparseTensor& r, const SparseTensor& t, Scalar value, const SparseTensor& src) {
  AT_CHECK(t.sizes().equals(src.sizes()), "cadd operands have incompatible sizes");
  if (src._nnz() == 0) {
    return r.copy_(t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse(r, src, value);
  }

  AT_CHECK(is_same_density(t, src), "cadd operands have incompatible desnitities");

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz(), max_nnz = t_nnz + s_nnz;
  bool t_coalesced = t.is_coalesced(), s_coalesced = src.is_coalesced();
  int64_t dimI = src._dimI();
  LongTensor t_indices = t._indices();
  Tensor t_values = t._values();
  LongTensor src_indices = src._indices();
  Tensor s_values = src._values();
  LongTensor r_indices = t_indices.type().tensor({dimI, max_nnz});
  Tensor r_values = _new_values_with_size_of(t_values, max_nnz).zero_();
  r.resize_as_(src);
  _get_sparse_impl(r)->set_indices_and_values(r_indices, r_values);  // TODO: sigh

  int64_t blockSize = r_values.stride(0);
  int64_t cmp, d;
  int64_t r_i = 0, t_i = 0, s_i = 0;

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
            for (d = 0; d < dimI; d++) {
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
            for (d = 0; d < dimI; d++) {
              r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
            }
            thblas::axpy<scalar_t>(blockSize, 1,
              t_values_ptr + t_i * blockSize, 1,
              r_values_ptr + r_i * blockSize, 1);
            t_i++;
          }
          if (cmp <= 0) {
            for (d = 0; d < dimI; d++) {
              r_indices_accessor[d][r_i] = src_indices_accessor[d][s_i];
            }
            thblas::axpy<scalar_t>(blockSize, cast_value,
              s_values_ptr + s_i * blockSize, 1,
              r_values_ptr + r_i * blockSize, 1);
            s_i++;
          }
          r_i++;
        }
      }
  );

  _get_sparse_impl(r)->set_nnz(r_i);
  // TODO: I think it may be possible to track inside the loop and
  // detect when we are uncoalesced (e.g., by observing that an
  // index goes backwards) which may be more precise than using the
  // coalesced flag here.  But this is easy.
  _get_sparse_impl(r)->set_coalesced(t_coalesced && s_coalesced);

  return r;
}

SparseTensor& csub_out_sparse_cpu(SparseTensor& r, const SparseTensor& t, Scalar value, const SparseTensor& src) {
  // UGH... We're doing two dispatches on scalar type here for no good reason.
  // NB: I tried adding an operator- to Scalar, but there isn't any good way
  // to negate the tensor, because I have a TensorBase...
  AT_DISPATCH_ALL_TYPES(
      t.type(), "csub_sparse", [&] {
        scalar_t cast_value = value.to<scalar_t>();
        cadd_out_sparse_cpu(r, t, -cast_value, src);
      }
  );
  return r;
}

/* Internal slice operations */

// NB: This function used to take in buffers which can be reused across calls;
// we dropped it because there is no select_out implementation at the moment
// NB: This function is called in a tight loop!
template <typename scalar_t>
static void _mul_slice_sparse(
    const Tensor& dst,
    TensorAccessor<scalar_t, 1> dst_accessor,
    const Tensor& src1,
    TensorAccessor<scalar_t, 1> src1_accessor,
    const Tensor& src2,
    TensorAccessor<scalar_t, 1> src2_accessor,
    int64_t dim,
    int64_t dstIdx,
    int64_t src1Idx,
    int64_t src2Idx
    ) {
  if (src1.dim() > 1) {
    Tensor src1Buffer = src1.select(dim, src1Idx);
    Tensor src2Buffer = src2.select(dim, src2Idx);
    Tensor dstBuffer = dst.select(dim, dstIdx);
    dstBuffer.addcmul_(src1Buffer, src2Buffer);
  } else {
    dst_accessor[dstIdx] = src1_accessor[src1Idx] * src2_accessor[src2Idx];
  }
}


// NB: divslice was removed as dead code

SparseTensor& cmul_out_sparse_cpu(SparseTensor& r, const SparseTensor& t_, const SparseTensor& src_) {
  AT_CHECK(t_.sizes().equals(src_.sizes()), "cadd operands have incompatible sizes");
  if (src_._nnz() == 0 || t_._nnz() == 0) {
    return r.zero_();
  }

  SparseTensor t = t_.coalesce();
  SparseTensor src = src_.coalesce();

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
  int64_t max_nnz = std::min(t_nnz, s_nnz);  // multiply by zero is zero, and can be dropped
  int64_t dimI = src._dimI();
  LongTensor t_indices = t._indices();
  Tensor t_values = t._values();
  LongTensor src_indices = src._indices();
  Tensor s_values = src._values();
  LongTensor r_indices = t_indices.type().tensor({dimI, max_nnz});
  Tensor r_values = _new_values_with_size_of(t_values, max_nnz).zero_();
  r.resize_as_(src);
  _get_sparse_impl(r)->set_indices_and_values(r_indices, r_values);  // TODO: sigh

  int64_t match, d;
  int64_t r_i = 0, t_i = 0, s_i = 0;
  auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
  auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
  auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

  AT_DISPATCH_ALL_TYPES(
      r_values.type(), "cmul_out_sparse", [&] {
        auto r_accessor = r_values.accessor<scalar_t, 1>();
        auto t_accessor = t_values.accessor<scalar_t, 1>();
        auto s_accessor = s_values.accessor<scalar_t, 1>();

        while (t_i < t_nnz && s_i < s_nnz) {
          match = 1;
          for (d = 0; d < dimI; d++) {
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
          if (!match) continue;
          for (d = 0; d < dimI; d++) {
            r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
          }
          _mul_slice_sparse<scalar_t>(r_values, r_accessor, t_values, t_accessor, s_values, s_accessor, 0, r_i, t_i, s_i);
          r_i++;
          t_i++;
          s_i++;
        }
      }
  );

  _get_sparse_impl(r)->set_nnz(r_i);
  _get_sparse_impl(r)->set_coalesced(true);

  return r;
}

Tensor& spaddcmul_out_sparse_cpu(Tensor& r, const Tensor& t, Scalar value, const SparseTensor& src1, const SparseTensor& src2) {
  SparseTensor intermediate = src1.mul(src2);
  add_out(r, t, intermediate, value); // aka spcadd
  return r;
}

LongTensor _to_csr(const int64_t* indices, int64_t dim, int64_t nnz) {
  int64_t h, i, hp0, hp1;
  LongTensor csr = at::CPU(kLong).zeros({dim + 1});

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
  return csr;
}

// NB: OMP pragmas have to get their own functions; can't put them in lambdas
template <typename scalar_t>
void spaddmm_out_worker(int64_t nnz, int64_t dim_i, int64_t dim_j, int64_t dim_k, Tensor& r, Scalar beta, const Tensor& t, Scalar alpha, const Tensor& csr, const Tensor& indices, const Tensor& values, const Tensor& dense) {
  int64_t h, i;

  // r_ = alpha * sparse * dense
  scalar_t cast_alpha = alpha.to<scalar_t>();
  scalar_t cast_beta = beta.to<scalar_t>();
  if (cast_beta == 0) {
    r.zero_();
  } else if (cast_beta == 1) {
    if (!isSameTensor(r, t)) {
      r.copy_(t); // TODO: not convinced this will work
    }
  } else {
    mul_out(r, t, beta);
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
        thblas::axpy<scalar_t>(dim_k,
            cast_alpha * val,
            dense_ptr + col * dense_stride0, dense_stride1,
            r_ptr + h * r_stride0, r_stride1);
      } else {
        AT_ERROR("index out of bound. spmm: ", col, " not between 1 and ", dim_j);
      }
    }
  }
};

Tensor& spaddmm_out_sparse_cpu(
    Tensor& r,
    Scalar beta,
    const Tensor& t,
    Scalar alpha,
    const SparseTensor& sparse_,
    const Tensor& dense
) {
  // TODO: This error message seems awfully opaque
  AT_CHECK(sparse_._dimI() == 2, "matrices expected, got ", sparse_._dimI(), "D tensor");
  AT_CHECK(sparse_._dimV() == 0, "scalar values expected, got ", sparse_._dimV(), "D values");
  AT_CHECK(dense.dim() == 2, "matrices expected, got ", dense.dim(), "D tensor");

  SparseTensor sparse = sparse_.coalesce();

  // ixj * jxk = ixk
  int64_t dim_i = sparse.size(0);
  int64_t dim_j = sparse.size(1);
  int64_t dim_k = dense.size(1);

  r.resize_({dim_i, dim_k});

  AT_CHECK(dense.size(0) == dim_j,
      "Argument #3 (dense): Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  AT_CHECK(t.size(0) == dim_i,
      "Argument #1 (t): Expected dim 0 size ", dim_i, ", got ", t.size(0));
  AT_CHECK(t.size(1) == dim_k,
      "Argument #1 (t): Expected dim 1 size ", dim_k, ", got ", t.size(1));

  int64_t nnz        = sparse._nnz();
  LongTensor indices = sparse._indices();
  Tensor values      = sparse._values();
  LongTensor csr = _to_csr(indices.data<int64_t>(), dim_i, nnz);

  AT_DISPATCH_ALL_TYPES(
      values.type(), "spmm", [&] {
        spaddmm_out_worker<scalar_t>(nnz, dim_i, dim_j, dim_k, r, beta, t, alpha, csr, indices, values, dense);
      }
  );

  return r;

}

SparseTensor& sspaddmm_out_sparse_cpu(
    SparseTensor& r,
    Scalar beta,
    const SparseTensor& t,
    Scalar alpha,
    const SparseTensor& sparse_,
    const Tensor& dense
) {
  AT_CHECK(sparse_._dimI() == 2,
      "Argument #2: matrices expected, got ", sparse_._dimI(), "D tensor");
  AT_CHECK(sparse_._dimV() == 0,
      "Argument #2: scalar values expected, got ", sparse_._dimV(), "D values");
  AT_CHECK(dense.dim() == 2,
      "Argument #2: matrices expected, got ", dense.dim(), "D tensor");

  SparseTensor sparse = sparse_.coalesce();

  // ixj * jxk = ixk
  int64_t dim_i = sparse.size(0);
  int64_t dim_j = sparse.size(1);
  int64_t dim_k = dense.size(1);

  r.resize_({dim_i, dim_k});

  AT_CHECK(dense.size(0) == dim_j,
      "Argument #3: Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  AT_CHECK(t.size(0) == dim_i,
      "Argument #1: Expected dim 0 size ", dim_i, ", got ", t.size(0));
  AT_CHECK(t.size(1) == dim_k,
      "Argument #1: Expected dim 1 size ", dim_k, ", got ", t.size(1));

  int64_t nnz        = sparse._nnz();
  LongTensor indices = sparse._indices();
  Tensor values      = sparse._values();

  LongTensor csr = _to_csr(indices.data<int64_t>(), dim_i, nnz);

  int64_t t_nnz = t._nnz();
  int64_t r_nnz = nnz * dim_k + t_nnz;
  LongTensor newi = at::CPU(kLong).tensor({2, r_nnz});
  LongTensor newv = values.type().zeros({r_nnz});

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
              thblas::axpy<scalar_t>(dim_k,
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
  _get_sparse_impl(r)->set_indices(newi);
  _get_sparse_impl(r)->set_values(newv);
  _get_sparse_impl(r)->set_nnz(p);

  return r;
}

SparseTensor& hspmm_out_sparse_cpu(SparseTensor& r, Scalar alpha, const SparseTensor& sparse_, const Tensor& dense) {
  AT_CHECK(sparse_._dimI() == 2,
      "Argument #2: matrices expected, got ", sparse_._dimI(), "D tensor");
  AT_CHECK(sparse_._dimV() == 0,
      "Argument #2: scalar values expected, got ", sparse_._dimV(), "D values");
  AT_CHECK(dense.dim() == 2,
      "Argument #2: matrices expected, got ", dense.dim(), "D tensor");

  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  AT_CHECK(dense.size(0) == k,
      "Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));
  _get_sparse_impl(r)->raw_resize_(1, 1, {m, n});

  SparseTensor sparse = sparse_.coalesce();

  int64_t nnz = sparse._nnz();
  LongTensor indices = at::CPU(kLong).tensor({1, nnz});

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
  Tensor values = dense.type().tensor({outNnz, n});
  _get_sparse_impl(newSparse)->_sizes_mut()[0] = outNnz; // TODO: use something safer

  // Compute output values tensor with sparse * dense multiplication
  spaddmm_out_sparse_cpu(values, 0, values, alpha, newSparse, dense);
  _get_sparse_impl(r)->set_indices_and_values(indices, values);  // TODO: sigh

  return r;
}


template <typename scalar_t>
void spcadd_out_worker(Tensor& r, Scalar value, const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {
  int64_t k;

  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto values_accessor = values.accessor<int64_t, 1>();

  scalar_t* r_ptr = r.data<scalar_t>();
  scalar_t cast_value = value.to<scalar_t>();

  #pragma omp parallel for private(k)
  for (k = 0; k < sparse._nnz(); k++) {
    int64_t index = r.storage_offset();
    for (int64_t d = 0; d < sparse._dimI(); d++) {
      index += r.stride(d) * indices_accessor[d][k];
    }
    r_ptr[index] += cast_value * values_accessor[k];
  }
}

Tensor& spcadd_out_sparse_cpu(Tensor& r, const Tensor& dense, Scalar value, const SparseTensor& sparse_) {
  r.resize_as_(dense);
  SparseTensor sparse = sparse_.coalesce();

  LongTensor indices = sparse._indices();
  Tensor values = sparse._values();
  int64_t nDim = dense.dim();
  int64_t nDimI = sparse._dimI();

  if (!isSameTensor(r, dense)) r.copy_(dense);

  if (nDim > nDimI) {
    auto indices_accessor = indices.accessor<int64_t, 2>();
    for (int64_t k = 0; k < sparse._nnz(); k++) {
      Tensor dstBuffer = r;
      for (int64_t d = 0; d < sparse._dimI(); d++) {
        dstBuffer = dstBuffer.select(0, indices_accessor[d][k]);
      }
      Tensor srcBuffer = values.select(0, k);
      dstBuffer.add_(srcBuffer, value);
    }
  } else {
    AT_DISPATCH_ALL_TYPES(
        values.type(), "spcadd", [&] {
          spcadd_out_worker<scalar_t>(r, value, sparse, indices, values);
        });
  }
  return r;
}

}} // namespace at::native
