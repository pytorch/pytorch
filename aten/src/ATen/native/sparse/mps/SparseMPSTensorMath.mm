#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/repeat_interleave_native.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/_sparse_sparse_matmul_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/mul.h>
#endif

namespace at::native {

using namespace at::sparse;
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SparseTensorMath_metallib.h>
#endif

static Tensor& s_addmm_out_sparse_dense_mps(
    Tensor& r,
    const Tensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha) {
  TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: sparse_dim must be 2, got ", sparse_.sparse_dim());
  TORCH_CHECK(sparse_.dense_dim() == 0, "addmm: sparse values must be 0-dense-dim, got ", sparse_.dense_dim());
  TORCH_CHECK(dense.dim() == 2, "addmm: 'dense' must be 2D, got ", dense.dim());
  TORCH_CHECK(t.dim() == 2, "addmm: 't' must be 2D, got ", t.dim());

  const int64_t I = sparse_.size(0);
  const int64_t J = sparse_.size(1);
  const int64_t K = dense.size(1);

  TORCH_CHECK(dense.size(0) == J,
      "addmm: dense (mat2) dim0 must be ", J, ", got ", dense.size(0));
  TORCH_CHECK(t.size(0) == I && t.size(1) == K,
      "addmm: 't' shape must be (", I, ", ", K, "), got (", t.size(0), ", ", t.size(1), ")");

  r.resize_({I, K});

  auto sparse = sparse_.coalesce();
  const int64_t nnz = sparse._nnz();

  if (nnz == 0 || I == 0 || K == 0) {
    at::mul_out(r, t, beta);
    return r;
  }

  const auto v_dtype = sparse._values().scalar_type();
  const auto d_dtype = dense.scalar_type();
  const auto t_dtype = t.scalar_type();
  auto compute_dtype = c10::promoteTypes(c10::promoteTypes(v_dtype, d_dtype), t_dtype);

  TORCH_CHECK(canCast(compute_dtype, r.scalar_type()),
              "Can't convert computed type ", compute_dtype, " to output ", r.scalar_type());

  auto indices2d = sparse._indices().contiguous();
  auto values = sparse._values().to(compute_dtype);
  auto dense_c = dense.to(compute_dtype).contiguous();
  auto t_c = t.to(compute_dtype).contiguous();

  const bool out_needs_cast = (r.scalar_type() != compute_dtype) || !r.is_contiguous();
  Tensor out_buf = out_needs_cast
      ? at::empty({I, K}, r.options().dtype(compute_dtype))
      : r;
  auto out_contig = out_buf.contiguous();

  auto device = r.device();
  auto stream = getCurrentMPSStream();

  const float alpha_f = alpha.to<float>();
  const float beta_f  = beta.to<float>();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      const std::string func = "spmm_addmm_coo_" + mps::scalarToMetalTypeString(values);
      auto pso = lib.getPipelineStateForFunc(func);
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];

      const uint32_t tew = pso.threadExecutionWidth;
      const uint32_t gridX = static_cast<uint32_t>(K);
      const uint32_t gridZ = static_cast<uint32_t>(I);
      const uint32_t tgW = std::min<uint32_t>(gridX, tew);

      MTLSize grid = MTLSizeMake(gridX, 1, gridZ);
      MTLSize tgs = MTLSizeMake(tgW, 1, 1);

      mtl_setArgs(enc,
                  indices2d,
                  values,
                  dense_c,
                  t_c,
                  out_contig,
                  std::array<uint32_t, 3>{static_cast<uint32_t>(I),
                                           static_cast<uint32_t>(J),
                                           static_cast<uint32_t>(K)},
                  std::array<float, 2>{alpha_f, beta_f},
                  static_cast<uint32_t>(nnz));
      [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    }
  });

  if (out_needs_cast) {
    r.copy_(out_contig.to(r.scalar_type()));
  }

  return r;
}


static void build_batch_ptr_mps(
    const Tensor& indices_dim0,
    int64_t B,
    Tensor& batch_ptr
) {
  // Builds an array of pointers which point to each batches elements. Example:
  // idx_b = [0, 0, 0, 1, 1, 2, 2, 2, 2]  // 9 non-zero elements
  //          └─────┘  └──┘  └─────────┘
  //          batch 0  batch 1  batch 2
  // batch_ptr = [0, 3, 5, 9]
  //              │  │  │  └─ end of batch 2 (total nnz)
  //              │  │  └──── batch 2 starts at index 5
  //              │  └─────── batch 1 starts at index 3
  //              └────────── batch 0 starts at index 0
  TORCH_CHECK(indices_dim0.is_mps() && batch_ptr.is_mps(), "MPS device expected");
  auto device = indices_dim0.device();
  auto stream = getCurrentMPSStream();

  const int64_t nnz = indices_dim0.numel();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("build_batch_ptr_from_sorted_batches");
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];

      const uint32_t tew = pso.threadExecutionWidth;
      const uint32_t Q = static_cast<uint32_t>(B + 1);
      const uint32_t tgW = std::min<uint32_t>(Q, tew);
      MTLSize grid = MTLSizeMake(Q, 1, 1);
      MTLSize tgs  = MTLSizeMake(tgW, 1, 1);

      mtl_setArgs(enc,
                  indices_dim0,
                  batch_ptr,
                  std::array<uint32_t, 2>{static_cast<uint32_t>(nnz),
                                          static_cast<uint32_t>(B)});
      [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    }
  });
}

static void build_row_ptr_per_batch_mps(
    const Tensor& rows,
    const Tensor& batch_ptr,
    int64_t B,
    int64_t I,
    Tensor& row_ptr
) {
  // Build per-batch CSR-style row pointer arrays from row indices sorted by batch
  // Given:
  //   rows: 1-D array of length nnz with row ids in [0, I), sorted within each batch
  //   batch_ptr: length B+1, where [batch_ptr[b], batch_ptr[b+1]) is the subrange for batch b
  // Produces:
  //   - row_ptr: shape [B, I+1]
  //
  // Example (B = 2, I = 4):
  // rows       = [0,   0,   1,  3,  0,   2,    2]   // 7 non-zero elements
  //               └─── batch 0 ──┘  └─ batch 1 ─┘
  // batch_ptr  = [0, 4, 7]
  //               │  │  └─ end of batch 1 (total nnz)
  //               │  └──── end of batch 0/start of batch 1
  //               └─────── start of batch 0
  //
  // per-batch row pointers (I+1 entries each):
  //   row_ptr[0] = [0, 2, 3, 3, 4]
  //   row_ptr[1] = [0, 1, 1, 3, 3]
  // laid out in memory: [0, 2, 3, 3, 4,  0, 1, 1, 3, 3]
  TORCH_CHECK(rows.is_mps() && batch_ptr.is_mps() && row_ptr.is_mps(), "MPS device expected");
  auto stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("build_row_ptr_from_sorted_rows_by_batch");
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];

      const uint32_t tew = pso.threadExecutionWidth;
      const uint32_t Qx = static_cast<uint32_t>(I + 1);
      const uint32_t Qy = static_cast<uint32_t>(B);
      const uint32_t tgW = std::min<uint32_t>(Qx, tew);

      MTLSize grid = MTLSizeMake(Qx, Qy, 1);
      MTLSize tgs = MTLSizeMake(tgW, 1, 1);

      mtl_setArgs(enc,
                  rows,
                  batch_ptr,
                  row_ptr,
                  std::array<uint32_t, 2>{static_cast<uint32_t>(I),
                                           static_cast<uint32_t>(B)});
      [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    }
  });
}

Tensor& bmm_out_sparse_mps(const SparseTensor& self_, const Tensor& mat2_, Tensor& result_) {
  TORCH_CHECK(result_.is_mps(), "bmm_sparse: expected 'out' to be MPS, got ", result_.device());
  TORCH_CHECK(self_.is_mps(),  "bmm_sparse: expected 'self' to be MPS, got ", self_.device());
  TORCH_CHECK(mat2_.is_mps(),  "bmm_sparse: expected 'mat2' to be MPS, got ", mat2_.device());

  TORCH_CHECK(self_.dense_dim() == 0, "bmm_sparse: Tensor 'self' must have 0 dense dims, but has ", self_.dense_dim());
  TORCH_CHECK(self_.sparse_dim() == 3, "bmm_sparse: Tensor 'self' must have 3 sparse dims, but has ", self_.sparse_dim());
  TORCH_CHECK(mat2_.dim() == 3, "bmm_sparse: Tensor 'mat2' must have 3 dims, but has ", mat2_.dim());

  TORCH_CHECK(self_.size(0) == mat2_.size(0), "bmm_sparse: 'self.size(0)' and 'mat2.size(0)' must match");
  TORCH_CHECK(self_.size(2) == mat2_.size(1), "bmm_sparse: 'self.size(2)' and 'mat2.size(1)' must match");

  const int64_t B = self_.size(0);
  const int64_t I = self_.size(1);
  const int64_t J = self_.size(2);
  const int64_t K = mat2_.size(2);

  auto self = self_.coalesce();
  const int64_t nnz = self._nnz();
  if (nnz == 0) {
    return result_.zero_();
  }

  const auto computeDtype = at::kFloat;

  auto indices = self._indices();
  auto values  = self._values();

  auto values_c = values.scalar_type() == computeDtype ? values : values.to(computeDtype);
  auto mat2_c = mat2_.scalar_type()   == computeDtype ? mat2_   : mat2_.to(computeDtype);
  auto mat2_contig = mat2_c.contiguous();

  auto idx_b = indices.select(0, 0).contiguous();
  auto idx_i = indices.select(0, 1).contiguous();
  auto idx_j = indices.select(0, 2).contiguous();

  // builds an array of pointers of where the batch_idx's pointer starts and ends
  // look in function for better explanation
  auto batch_ptr = at::empty({B + 1}, at::device(result_.device()).dtype(kLong));
  build_batch_ptr_mps(idx_b, B, batch_ptr);
  // build row_ptr per batch: for each (b, i) get [start, end) into rows/cols/vals
  auto row_ptr = at::empty({B * (I + 1)}, at::device(result_.device()).dtype(kLong));
  build_row_ptr_per_batch_mps(idx_i, batch_ptr, B, I, row_ptr);

  const bool out_needs_cast = (result_.scalar_type() != computeDtype) || !result_.is_contiguous();
  Tensor out_buf = out_needs_cast
      ? at::empty({B, I, K}, result_.options().dtype(computeDtype))
      : result_;
  auto out_contig = out_buf.contiguous();

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("spmm_bmm_coo_rows_grouped_" + mps::scalarToMetalTypeString(values));
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];

      const uint32_t tew = pso.threadExecutionWidth;
      const uint32_t tgW = std::min<uint32_t>((uint32_t)K, tew);

      // One threadgroup per (row i, batch b), lanes cover K
      MTLSize grid = MTLSizeMake(tgW, (uint32_t)I, (uint32_t)B);
      MTLSize tgs  = MTLSizeMake(tgW, 1, 1);

      mtl_setArgs(enc,
                  idx_i,
                  idx_j,
                  values_c,
                  mat2_contig,
                  out_contig,
                  row_ptr,
                  std::array<uint32_t, 4>{(uint32_t)B, (uint32_t)I, (uint32_t)J, (uint32_t)K});
      [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    }
  });
  if (out_needs_cast) {
    result_.copy_(out_contig.to(result_.scalar_type()));
  }
  return result_;
}

Tensor bmm_sparse_mps(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::zeros({self.size(0), self.size(1), mat2.size(2)}, mat2.options());
  return bmm_out_sparse_mps(self, mat2, result);
}

Tensor& addmm_out_sparse_dense_mps(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_out_sparse_dense_mps(result, *b_self, mat1, mat2, beta, alpha);
}

Tensor addmm_sparse_dense_mps(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha
) {
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  Tensor result = at::empty({0}, self.options());
  return s_addmm_out_sparse_dense_mps(result, *b_self, mat1, mat2, beta, alpha);
}

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
    auto out_vals = values.mul(dense.to(values.options()));
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

  auto lhs_indices = lhs._indices().contiguous();
  auto rhs_indices = rhs._indices().contiguous();
  auto lhs_values  = lhs._values().to(commonDtype).contiguous();
  auto rhs_values  = rhs._values().to(commonDtype).contiguous();

  // Flatten sparse indices to keys
  auto lhs_keys = flatten_indices(lhs_indices, lhs.sizes().slice(0, ndim_i));
  auto rhs_keys = flatten_indices(rhs_indices, rhs.sizes().slice(0, ndim_i));

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
  auto dense_sizes_vec = lhs.sizes().slice(ndim_i).vec();
  int64_t cols64 = 1;
  for (auto s : dense_sizes_vec) cols64 *= s;
  const uint32_t cols = static_cast<uint32_t>(std::max<int64_t>(cols64, 1));

  auto to2d = [&](Tensor t, int64_t nnz) -> Tensor {
    const int64_t t_cols = t.numel() / nnz;
    if (t_cols == cols64) {
      return t.view({nnz, cols64});
    }
    return t.view({nnz, 1}).expand({nnz, cols64}).contiguous();
  };

  // make both sides 2d [nnz, cols] buffers so the kernel can index it
  auto lhs_vals2d = to2d(lhs_values, lhs_nnz);
  auto rhs_vals2d = to2d(rhs_values, rhs_nnz);

  std::vector<int64_t> out_val_sizes;
  out_val_sizes.reserve(1 + dense_sizes_vec.size());
  out_val_sizes.push_back(static_cast<int64_t>(M));
  out_val_sizes.insert(out_val_sizes.end(), dense_sizes_vec.begin(), dense_sizes_vec.end());
  auto out_values = at::empty(out_val_sizes, lhs_values.options());

  if (M > 0) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto pso = lib.getPipelineStateForFunc(
            "fused_gather_mul_kernel_" + mps::scalarToMetalTypeString(lhs_values));
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:pso];

        const uint32_t tew = pso.threadExecutionWidth;
        const uint32_t gridW = std::max<uint32_t>(cols, 1u);
        const uint32_t tgW = std::min(gridW, tew);
        MTLSize grid = MTLSizeMake(gridW, 1, M);
        MTLSize tgs  = MTLSizeMake(tgW, 1, 1);

        mtl_setArgs(enc,
                    lhs_vals2d, rhs_vals2d,
                    lhs_match, rhs_match,
                    lhs_indices, out_indices,
                    out_values,
                    std::array<uint32_t, 2>{static_cast<uint32_t>(ndim_i), static_cast<uint32_t>(lhs_nnz)},
                    std::array<uint32_t, 2>{M, cols});
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
      }
    });
  }

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

using OptTensor = std::optional<Tensor>;


static void sparse_mask_apply_out_mps_kernel(
    Tensor& result,
    const Tensor& src_in,
    const Tensor& mask_in,
    bool accumulate_matches,
    bool require_same_sizes,
    bool coalesce_mask) {
  TORCH_CHECK(src_in.is_sparse() && mask_in.is_sparse(),
              "sparse_mask: expected both inputs to be sparse COO");
  TORCH_CHECK(src_in.is_mps() && mask_in.is_mps(),
              "sparse_mask: expected tensors to be on MPS device");
  TORCH_CHECK(src_in.sparse_dim() == mask_in.sparse_dim(),
              "sparse_mask: sparse_dim mismatch: ", src_in.sparse_dim(), " vs ", mask_in.sparse_dim());
  if (require_same_sizes) {
    TORCH_CHECK(src_in.sizes().equals(mask_in.sizes()),
                "sparse_mask: sizes must match exactly (no broadcasting)");
  }
  auto src  = src_in.coalesce();
  auto mask = coalesce_mask ? mask_in.coalesce() : mask_in;

  const int64_t src_nnz = src._nnz();
  const int64_t mask_nnz = mask._nnz();
  const int64_t sd = src.sparse_dim();
  result.sparse_resize_(mask.sizes(), mask.sparse_dim(), mask.dense_dim());

  auto commonDtype = at::result_type(src, mask);
  TORCH_CHECK(canCast(commonDtype, result.scalar_type()),
              "Can't convert result type ", commonDtype, " to output ", result.scalar_type());

  if (mask_nnz == 0) {
    alias_into_sparse(
        result,
        mask._indices().narrow(1, 0, 0),
        at::empty({0}, result.options().dtype(result.scalar_type())));
    result._coalesced_(mask.is_coalesced());
    return;
  }

  TORCH_CHECK(sd > 0 || (src_nnz <= 1 && mask_nnz <= 1),
              "sparse_mask: invalid sparse_dim or nnz");

  if (sd == 0) {
    auto out_indices = mask._indices().narrow(1, 0, 1);
    auto out_values = src_nnz
      ? src._values().narrow(0, 0, 1).to(commonDtype)
      : at::zeros({1}, at::device(result.device()).dtype(commonDtype));
    alias_into_sparse(result, out_indices, out_values);
    result._coalesced_(mask.is_coalesced());
    return;
  }

  if (src_nnz == 0) {
    auto out_indices = mask._indices().contiguous();
    auto src_values  = src._values().to(commonDtype);
    auto out_val_sizes = src_values.sizes().vec();
    out_val_sizes[0] = mask_nnz;
    auto out_values = at::zeros(out_val_sizes, src_values.options());
    alias_into_sparse(result, out_indices, out_values);
    result._coalesced_(mask.is_coalesced());
    return;
  }

  auto mask_indices = mask._indices().contiguous();
  auto src_indices = src._indices().contiguous();
  auto src_values = src._values().to(commonDtype).contiguous();

  auto mask_keys = flatten_indices(mask_indices, mask.sizes().slice(0, sd)).contiguous();
  auto src_keys  = flatten_indices(src_indices,  src.sizes().slice(0, sd)).contiguous();

  const bool A_is_src = (src_nnz <= mask_nnz);
  const int64_t lenA = A_is_src ? src_nnz  : mask_nnz;
  const int64_t lenB = A_is_src ? mask_nnz : src_nnz;
  auto A_keys = A_is_src ? src_keys  : mask_keys;
  auto B_keys = A_is_src ? mask_keys : src_keys;

  const auto device = result.device();
  auto stream = getCurrentMPSStream();

  auto outA_idx = at::empty({lenA}, at::device(device).dtype(at::kLong));
  auto outB_idx = at::empty({lenA}, at::device(device).dtype(at::kLong));
  auto counter = at::zeros({1}, at::device(device).dtype(at::kInt));

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("intersect_binary_search");
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, A_keys, B_keys, outA_idx, outB_idx, counter,
                  static_cast<uint32_t>(lenB), A_is_src);
      mtl_dispatch1DJob(enc, pso, static_cast<uint32_t>(lenA));
    }
  });

  const int64_t M = static_cast<int64_t>(counter.item<int32_t>());

  auto out_val_sizes = src_values.sizes().vec();
  out_val_sizes[0] = mask_nnz;
  auto out_values = at::zeros(out_val_sizes, src_values.options());

  if (M > 0) {
    auto src_match = outA_idx.narrow(0, 0, M);
    auto mask_match = outB_idx.narrow(0, 0, M);

    auto src_rows = src_values.index_select(0, src_match);
    if (accumulate_matches) {
      out_values.index_add_(0, mask_match, src_rows);
    } else {
      out_values.index_copy_(0, mask_match, src_rows);
    }
  }

  alias_into_sparse(result, mask_indices, out_values);
  result._coalesced_(mask.is_coalesced());
}

static void sparse_mask_intersection_out_mps_kernel(
    Tensor& result,
    const Tensor& lhs,
    const Tensor& rhs,
    const OptTensor& = std::nullopt) {
  sparse_mask_apply_out_mps_kernel(
      result,
      /*src_in=*/lhs,
      /*mask_in=*/rhs,
      /*accumulate_matches=*/false,
      /*require_same_sizes=*/false,
      /*coalesce_mask=*/false);
}

Tensor sparse_sparse_matmul_mps(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_CHECK(mat1_.is_sparse() && mat2_.is_sparse(),
              "sparse_sparse_matmul_mps: both inputs must be sparse COO tensors");
  TORCH_CHECK(mat1_.is_mps() && mat2_.is_mps(),
              "sparse_sparse_matmul_mps: both inputs must be on MPS device");
  TORCH_CHECK(mat1_.dim() == 2 && mat2_.dim() == 2,
              "sparse_sparse_matmul_mps: both inputs must be 2D matrices");
  TORCH_CHECK(mat1_.dense_dim() == 0 && mat2_.dense_dim() == 0,
              "sparse_sparse_matmul_mps: only scalar values supported (dense_dim == 0)");
  TORCH_CHECK(mat1_.size(1) == mat2_.size(0),
              "mat1 and mat2 shapes cannot be multiplied (", mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");
  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
              "sparse_sparse_matmul_mps: mat1 dtype ", mat1_.scalar_type(),
              " does not match mat2 dtype ", mat2_.scalar_type());

  const auto device = mat1_.device();

  auto A = mat1_.coalesce();
  auto B = mat2_.coalesce();

  const auto I = A.size(0);
  const auto K = A.size(1);
  const auto N = B.size(1);

  const auto nnzA = A._nnz();
  const auto nnzB = B._nnz();

  // Early empty result, return an empty, coalesced tensor
  if (I == 0 || N == 0 || K == 0 || nnzA == 0 || nnzB == 0) {
    auto empty_idx = at::empty({2, 0}, at::device(device).dtype(at::kLong));
    auto empty_val = at::empty({0}, at::device(device).dtype(mat1_.scalar_type()));
    auto out = _sparse_coo_tensor_unsafe(empty_idx, empty_val, {I, N}, mat1_.options());
    out._coalesced_(true);
    return out;
  }

  const auto computeDtype = at::result_type(mat1_, mat2_);

  auto A_idx = A._indices().contiguous();
  auto A_val = A._values().to(computeDtype).contiguous();
  auto A_i = A_idx.select(0, 0).contiguous();
  auto A_k = A_idx.select(0, 1).contiguous();

  auto B_idx = B._indices().contiguous();
  auto B_val = B._values().to(computeDtype).contiguous();
  auto B_k = B_idx.select(0, 0).contiguous();
  auto B_j = B_idx.select(0, 1).contiguous();

  // csr-style row pointers for B by k (the shared dimension)
  Tensor row_ptr_B;
  {
    auto batch_ptr = at::tensor({0LL, nnzB}, at::device(device).dtype(at::kLong));
    row_ptr_B = at::empty({K + 1}, at::device(device).dtype(at::kLong));
    build_row_ptr_per_batch_mps(B_k, batch_ptr, /*B=*/1, /*I=*/K, row_ptr_B);
  }

  auto row_ptr_B_lo = row_ptr_B.narrow(0, 0, K);
  auto row_ptr_B_hi = row_ptr_B.narrow(0, 1, K);
  auto deg_B = row_ptr_B_hi.sub(row_ptr_B_lo);

  auto counts = deg_B.index_select(0, A_k);

  const int64_t P = counts.sum().item<int64_t>();
  if (P == 0) {
    auto empty_idx = at::empty({2, 0}, at::device(device).dtype(at::kLong));
    auto empty_val = at::empty({0}, at::device(device).dtype(mat1_.scalar_type()));
    auto out = _sparse_coo_tensor_unsafe(empty_idx, empty_val, {I, N}, mat1_.options());
    out._coalesced_(true);
    return out;
  }

  auto group_ids = repeat_interleave_mps(counts);

  // exclusive cumsum of counts
  auto offsets = cumsum(counts, /*dim=*/0).sub(counts);
  auto offsets_gather = offsets.index_select(0, group_ids);
  auto within = at::arange(P, at::device(device).dtype(at::kLong)).sub(offsets_gather);

  // Map each output element to its source B row and position
  auto k_per_out = A_k.index_select(0, group_ids);
  auto start_in_B = row_ptr_B.index_select(0, k_per_out);
  auto seg_index = start_in_B.add(within);

  // Assemble candidate coo pairs and values
  auto i_out = A_i.index_select(0, group_ids).contiguous();
  auto j_out = B_j.index_select(0, seg_index).contiguous();
  auto vA_out = A_val.index_select(0, group_ids).contiguous();
  auto vB_out = B_val.index_select(0, seg_index).contiguous();
  auto v_out = vA_out.mul(vB_out);

  // build (2, P) indices
  auto out_indices = at::empty({2, P}, at::device(device).dtype(at::kLong)).contiguous();
  out_indices.select(0, 0).copy_(i_out);
  out_indices.select(0, 1).copy_(j_out);

  auto result = _sparse_coo_tensor_unsafe(
      out_indices, v_out, {I, N}, mat1_.options().dtype(computeDtype));

  result = result.coalesce();

  if (result.scalar_type() != mat1_.scalar_type()) {
    auto cast_vals = result._values().to(mat1_.scalar_type());
    auto out = _sparse_coo_tensor_unsafe(result._indices(), cast_vals, {I, N}, mat1_.options());
    out._coalesced_(true);
    return out;
  }
  return result;
}

REGISTER_MPS_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_mps_kernel);
} // namespace at::native