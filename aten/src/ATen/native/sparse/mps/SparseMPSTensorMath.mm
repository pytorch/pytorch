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
#include <ATen/ops/mul_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/mul.h>
#endif

namespace at::native {

using namespace at::sparse;
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Mul_metallib.h>
#endif

static SparseTensor& mul_out_dense_sparse_mps(
    const Tensor& dense,
    const Tensor& sparse,
    SparseTensor& out) {

  TORCH_CHECK(sparse.is_sparse(), "mul: expected 'sparse' to be sparse COO");
  TORCH_CHECK(sparse.is_mps(), "mul: expected 'sparse' to be MPS, got ", sparse.device());
  TORCH_CHECK(out.is_mps(), "mul: expected 'out' to be MPS, got ", out.device());

  const bool scalar_like = (dense.dim() == 0) || (dense.numel() == 1);
  TORCH_CHECK(dense.is_mps() || scalar_like,
              "mul: expected 'dense' to be MPS or scalar-like, got ", dense.device());

  const int64_t nnz = sparse._nnz();
  out.resize_as_(sparse);

  auto commonDtype = at::result_type(dense, sparse);
  TORCH_CHECK(canCast(commonDtype, out.scalar_type()),
              "Can't convert result type ", commonDtype, " to output ", out.scalar_type());

  auto indices = sparse._indices().contiguous();
  auto values  = sparse._values().to(commonDtype).contiguous();

  if (nnz == 0) {
    auto empty_vals = values.narrow(0, 0, 0);
    alias_into_sparse(out,
                      indices.narrow(1, 0, 0),
                      (out.scalar_type() == commonDtype) ? empty_vals
                                                          : empty_vals.to(out.scalar_type()));
    out._coalesced_(sparse.is_coalesced());
    return out;
  }

  if (scalar_like) {
    auto scalar = dense;
    if (dense.numel() == 1 && dense.dim() > 0) {
      scalar = dense.view({});
    }
    scalar = scalar.to(values.options());
    auto out_vals = values.mul(scalar);
    if (out.scalar_type() != commonDtype) {
      out_vals = out_vals.to(out.scalar_type());
    }

    alias_into_sparse(out, indices, out_vals);
    out._coalesced_(sparse.is_coalesced());
    return out;
  }

  TORCH_CHECK(dense.sizes().equals(sparse.sizes()),
              "mul(dense, sparse): sizes must match exactly (no broadcasting): ",
              dense.sizes(), " vs ", sparse.sizes());

  const int64_t ndim_i = sparse.sparse_dim();
  const int64_t ndim = dense.dim();
  TORCH_CHECK(
    ndim_i <= ndim,
    "mul(dense, sparse): sparse_dim=", ndim_i, " exceeds dense.dim()=", ndim);

  // Prepare shapes
  int64_t view_rows = 1, view_cols = 1;
  for (int64_t i = 0; i < ndim_i; ++i) view_rows *= sparse.size(i);
  for (int64_t i = ndim_i; i < ndim; ++i) view_cols *= sparse.size(i);

  auto dense_mps = dense.to(commonDtype).contiguous().reshape({view_rows, view_cols});
  auto out_vals = at::empty_like(values, values.options());

  const uint32_t u_view_cols = static_cast<uint32_t>(view_cols);
  const uint32_t u_nnz = static_cast<uint32_t>(nnz);
  const uint32_t u_ndim_i = static_cast<uint32_t>(ndim_i);

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("dense_sparse_mul_kernel_" + mps::scalarToMetalTypeString(values));
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:pso];

      const uint32_t gridWidth = u_view_cols;
      const uint32_t gridDepth = u_nnz;
      MTLSize gridSize = MTLSizeMake(gridWidth, 1, gridDepth);

      const uint32_t maxThreadsPerGroup = pso.maxTotalThreadsPerThreadgroup;
      const uint32_t tew = pso.threadExecutionWidth;
      uint32_t tgWidth  = std::min(gridWidth, tew);
      MTLSize threadgroupSize = MTLSizeMake(tgWidth, 1, 1);

      mtl_setArgs(
        computeEncoder,
        dense_mps,
        values,
        out_vals,
        indices,
        sparse.sizes(),
        std::array<uint32_t, 3>{u_nnz, u_ndim_i, u_view_cols}
      );

      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });

  Tensor final_vals = out_vals;
  if (out.scalar_type() != commonDtype) {
    final_vals = final_vals.to(out.scalar_type());
  }

  alias_into_sparse(out, indices, final_vals);
  out._coalesced_(sparse.is_coalesced());
  return out;
}


SparseTensor& mul_out_sparse_mps(const Tensor& t_, const Tensor& src_, SparseTensor& r_) {
  TORCH_CHECK(r_.is_mps(), "mul: expected 'out' to be MPS, but got ", r_.device());

  // Dense x sparse fallback (keep dense first)
  if (!t_.is_sparse() || !src_.is_sparse()) {
    const Tensor& dense  = t_.is_sparse() ? src_ : t_;
    const Tensor& sparse = t_.is_sparse() ? t_   : src_;
    return mul_out_dense_sparse_mps(dense, sparse, r_);
  }

  TORCH_CHECK(t_.is_mps(),   "mul: expected 'self' to be MPS, but got ", t_.device());
  TORCH_CHECK(src_.is_mps(), "mul: expected 'other' to be MPS, but got ", src_.device());
  TORCH_CHECK(t_.sparse_dim() == src_.sparse_dim(),
              "mul(sparse, sparse): must have same sparse_dim, got ",
              t_.sparse_dim(), " vs ", src_.sparse_dim());
  TORCH_CHECK(t_.sizes().equals(src_.sizes()),
              "mul(sparse, sparse): sizes must match exactly (no broadcasting).");

  // Coalesce and early-exit on structurally empty operands
  auto lhs = t_.coalesce();
  auto rhs = src_.coalesce();
  const int64_t lhs_nnz = lhs._nnz();
  const int64_t rhs_nnz = rhs._nnz();
  if (!lhs_nnz || !rhs_nnz) {
    r_.resize_as_(lhs);
    return r_.zero_();
  }

  // dtype checks and promotion
  auto commonDtype = at::result_type(lhs, rhs);
  TORCH_CHECK(canCast(commonDtype, r_.scalar_type()),
              "Can't convert result type ", commonDtype, " to output ", r_.scalar_type());

  const int64_t ndim_i = lhs.sparse_dim();

  // ndim_i == 0, at most one structural entry
  if (ndim_i == 0) {
    r_.resize_as_(lhs);
    const bool has = (lhs_nnz && rhs_nnz);

    auto out_indices = lhs._indices().narrow(1, 0, has ? 1 : 0);

    Tensor lhs_vals = lhs._values().to(commonDtype);
    Tensor rhs_vals = rhs._values().to(commonDtype);
    lhs_vals = lhs_vals.narrow(0, 0, has ? 1 : 0);
    rhs_vals = rhs_vals.narrow(0, 0, has ? 1 : 0);

    Tensor out_values = lhs_vals.mul(rhs_vals);
    if (r_.scalar_type() != commonDtype) {
      out_values = out_values.to(r_.scalar_type());
    }

    alias_into_sparse(r_, out_indices, out_values);
    r_._coalesced_(true);
    return r_;
  }

  // General path, intersect keys, then gather + multiply on GPU
  const auto device = r_.device();
  auto stream = getCurrentMPSStream();

  auto lhs_indices = lhs._indices();
  auto rhs_indices = rhs._indices();
  auto lhs_values  = lhs._values().to(commonDtype);
  auto rhs_values  = rhs._values().to(commonDtype);

  // Flatten sparse indices to keys
  auto lhs_keys = flatten_indices(lhs_indices, lhs.sizes());
  auto rhs_keys = flatten_indices(rhs_indices, rhs.sizes());

  // Intersect sorted keys (search the shorter in the longer)
  const bool A_is_lhs = (lhs_nnz <= rhs_nnz);
  const int64_t lenA = A_is_lhs ? lhs_nnz : rhs_nnz;
  const int64_t lenB = A_is_lhs ? rhs_nnz : lhs_nnz;
  auto A_keys = A_is_lhs ? lhs_keys : rhs_keys;
  auto B_keys = A_is_lhs ? rhs_keys : lhs_keys;

  auto outA_idx = at::empty({lenA}, at::device(device).dtype(kLong));
  auto outB_idx = at::empty({lenA}, at::device(device).dtype(kLong));
  auto counter = at::zeros({1}, at::device(device).dtype(kInt));

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("intersect_binary_search");
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, A_keys, B_keys, outA_idx, outB_idx, counter,
                  static_cast<uint32_t>(lenB), A_is_lhs);
      mtl_dispatch1DJob(enc, pso, static_cast<uint32_t>(lenA));
    }
  });

  const uint32_t M = counter.item<int32_t>(); // number of structural matches

  r_.resize_as_(lhs);

  auto out_indices = at::empty({ndim_i, static_cast<int64_t>(M)}, at::device(device).dtype(at::kLong));
  auto lhs_match = outA_idx.narrow(0, 0, M);
  auto rhs_match = outB_idx.narrow(0, 0, M);
  auto out_val_sizes = lhs_values.sizes().vec();
  out_val_sizes[0] = static_cast<int64_t>(M);
  auto out_values = at::empty(out_val_sizes, lhs_values.options());

  const uint32_t cols = static_cast<uint32_t>(
      lhs_values.numel() / std::max<int64_t>(1, lhs_nnz));

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc(
          "fused_gather_mul_kernel_" + mps::scalarToMetalTypeString(lhs_values));
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];

      const uint32_t tew  = pso.threadExecutionWidth;
      uint32_t tgW = std::min(cols, tew);
      MTLSize grid = MTLSizeMake(cols, 1, M);
      MTLSize tgs  = MTLSizeMake(tgW, 1, 1);

      mtl_setArgs(enc,
                  lhs_values, rhs_values,
                  lhs_match, rhs_match,
                  lhs_indices, out_indices,
                  out_values,
                  std::array<uint32_t, 2>{static_cast<uint32_t>(ndim_i), static_cast<uint32_t>(lhs_nnz)},
                  std::array<uint32_t, 2>{M, cols});
      [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    }
  });

  if (r_.scalar_type() != commonDtype) {
    out_values = out_values.to(r_.scalar_type());
  }

  alias_into_sparse(r_, out_indices, out_values);
  r_._coalesced_(true);
  return r_;
}

static Tensor& add_out_dense_sparse_mps(
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