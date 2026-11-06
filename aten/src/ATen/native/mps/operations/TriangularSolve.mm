#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <ATen/native/mps/operations/TriangularSolve.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/eye.h>
#include <ATen/ops/linalg_solve_triangular_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/triangular_solve_native.h>
#endif

#include <algorithm>
#include <limits>

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/TriangularSolve_metallib.h>
#endif

// Metal substitution kernel: one threadgroup per right-hand side, reducing the
// dot product against the already-solved prefix across the group. conjugate
// selects adjoint (A^H) instead of plain transpose (A^T) when transpose is set;
// it is a no-op for real inputs. A must be contiguous; B and out may each be
// row- or column-major.
static void triangular_solve_metal(const Tensor& A_,
                                   const Tensor& B_,
                                   bool upper,
                                   bool transpose,
                                   bool conjugate,
                                   bool unitriangular,
                                   const Tensor& out) {
  using namespace mps;
  const uint64_t batchSize =
      std::accumulate(A_.sizes().begin(), A_.sizes().end() - 2, 1ULL, std::multiplies<uint64_t>());
  const uint64_t n = A_.size(-1);
  const uint64_t k = B_.size(-1);

  TriangularSolveParams params;
  params.nbatch = safe_downcast<uint32_t, uint64_t>(batchSize);
  params.n = safe_downcast<uint32_t, uint64_t>(n);
  params.k = safe_downcast<uint32_t, uint64_t>(k);
  params.upper = upper;
  params.transpose = transpose;
  params.conj = conjugate;
  params.unit = unitriangular;

  const bool use_small_kernel =
      A_.scalar_type() == kFloat && (n == 16 || n == 32 || n == 64) && k >= kTriangularSolveTileSize;
  const bool general = !unitriangular || upper || transpose || k % kTriangularSolveTileSize != 0;
  // MPP computes offsets within each 16x32 tile in int32.
  constexpr uint64_t kMaxMppStride =
      (std::numeric_limits<int32_t>::max() - (kTriangularSolveMppCols - 1)) / (kTriangularSolveMppRows - 1);
  const bool use_mpp = use_small_kernel && has_mpp() && k <= kMaxMppStride;
  // Inverse application and its substitution fallback may reread both inputs.
  const auto a = use_small_kernel && A_.is_alias_of(out) ? A_.clone() : A_;
  const auto b = use_small_kernel && B_.is_alias_of(out) ? B_.clone() : B_;
  TORCH_INTERNAL_ASSERT(is_row_or_column_contiguous(b) && is_row_or_column_contiguous(out));
  params.b_col_major = !b.is_contiguous();
  params.x_col_major = !out.is_contiguous();

  const uint64_t elem_size = A_.element_size();
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(
          use_small_kernel
              ? fmt::format("triangular_solve_small_{}{}{}", use_mpp ? "mpp_" : "", general ? "general_" : "", n)
              : fmt::format("triangular_solve_{}", scalarToMetalTypeString(A_)));
      getMPSProfiler().beginProfileKernel(pso, "triangular_solve", {a, b}, stream);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, a, b, out, params);
      if (use_small_kernel) {
        [encoder dispatchThreadgroups:MTLSizeMake(batchSize, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(n / kTriangularSolveTileSize * c10::metal::simdgroup_size, 1, 1)];
        getMPSProfiler().endProfileKernel(pso, stream);
        return;
      }
      // Every substitution step reduces across the whole threadgroup, so don't
      // spread a short row over more threads than it has work for, and keep the
      // group a whole number of simdgroups so the reduction stays exact.
      constexpr uint64_t kSimd = c10::metal::simdgroup_size;
      constexpr uint64_t kMaxThreadsPerGroup = 8 * kSimd;
      const uint64_t maxThreads = std::min<uint64_t>(pso.maxTotalThreadsPerThreadgroup, kMaxThreadsPerGroup);
      const uint64_t tgSize = std::clamp<uint64_t>(round_up(n, kSimd), kSimd, maxThreads);
      // setThreadgroupMemoryLength rejects lengths that are not a multiple of 16.
      constexpr uint64_t kTGMemAlign = 16;
      const uint64_t redBytes = round_up(tgSize / kSimd * elem_size, kTGMemAlign);
      const uint64_t prefixBytes = round_up(n * elem_size, kTGMemAlign);
      // The caller sends anything bigger down the blocked path, which only ever
      // hands this kernel a block, so the prefix always fits.
      const uint64_t maxTGMem = [MPSDevice::getInstance()->device() maxThreadgroupMemoryLength];
      TORCH_INTERNAL_ASSERT(
          prefixBytes + redBytes <= maxTGMem, "triangular_solve: n=", n, " does not fit in threadgroup memory");
      [encoder setThreadgroupMemoryLength:prefixBytes atIndex:0];
      [encoder setThreadgroupMemoryLength:redBytes atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(tgSize * batchSize * k, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

// Right-looking blocked solve: solve the nb x nb diagonal block, then one GEMM
// to push its contribution into the rest of the panel. Substitution reads every
// element of A once per right-hand side and serializes n steps deep; blocking
// cuts that traffic by nb, shortens the chain to nb, and moves the rest into
// matmul. X holds B on entry and the solution on exit; M is op(A), possibly as
// a transposed/conjugated view, with its triangle given by upper.
static void triangular_solve_blocked(const Tensor& M, bool upper, bool unitriangular, int64_t nb, Tensor& X) {
  const int64_t n = M.size(-1);
  const int64_t k = X.size(-1);
  // M may be a transposed and/or conjugated view, so fold the batch dims with
  // flatten (a view here) rather than reshape, which would copy. matmul and
  // baddbmm_ take such views directly, which is what keeps a transposed or
  // adjoint solve from materializing an n x n copy of op(A).
  const Tensor M3 = M.dim() == 2 ? M.unsqueeze(0) : M.flatten(0, -3);
  Tensor X3 = X.dim() == 2 ? X.unsqueeze(0) : X.flatten(0, -3);
  const int64_t nbatch = M3.size(0);

  // An upper triangle is solved from the bottom block up.
  for (int64_t b = 0; b < n; b += nb) {
    const int64_t rows = std::min(nb, n - b);
    const int64_t i0 = upper ? n - b - rows : b;
    auto Xi = X3.narrow(1, i0, rows);
    // The kernel reads raw elements, so the diagonal block is materialized;
    // it is only nb x nb.
    auto diag = M3.narrow(1, i0, rows).narrow(2, i0, rows).resolve_conj().contiguous();
    // Inverting the diagonal block turns the panel solve into a matmul, and the
    // block is small enough that it does not measurably cost accuracy. But the
    // inverse itself is a solve with `rows` right-hand sides, so it only pays
    // off once there are at least that many to amortize it over.
    Tensor Xi_new;
    if (k >= rows) {
      Tensor eye = at::eye(rows, M.options().memory_format(std::nullopt)).expand({nbatch, rows, rows}).contiguous();
      Tensor dinv = at::empty_like(eye);
      triangular_solve_metal(diag, eye, upper, /*transpose=*/false, /*conjugate=*/false, unitriangular, dinv);
      Xi_new = at::matmul(dinv, Xi);
    } else {
      Xi_new = at::empty_like(Xi, at::MemoryFormat::Contiguous);
      triangular_solve_metal(
          diag, Xi.contiguous(), upper, /*transpose=*/false, /*conjugate=*/false, unitriangular, Xi_new);
    }
    Xi.copy_(Xi_new);
    const int64_t rest = upper ? i0 : n - i0 - rows;
    if (rest > 0) {
      const int64_t r0 = upper ? 0 : i0 + rows;
      // Accumulate in place: a plain sub_(matmul(...)) would allocate a result
      // per block, each a different size, which the caching allocator then
      // holds on to one block per size.
      X3.narrow(1, r0, rest).baddbmm_(M3.narrow(1, r0, rest).narrow(2, i0, rows), Xi_new, /*beta=*/1, /*alpha=*/-1);
    }
  }
}

void linalg_solve_triangular_mps_impl(const Tensor& A,
                                      const Tensor& B,
                                      bool upper,
                                      bool transpose,
                                      bool left,
                                      bool unitriangular,
                                      const Tensor& out,
                                      bool conjugate) {
  using namespace mps;

  checkInputsSolver(A, B, left, "linalg.solve_triangular");
  const auto scalar_type = A.scalar_type();
  TORCH_CHECK(scalar_type == kFloat || scalar_type == kComplexFloat,
              "linalg.solve_triangular(): MPS only supports float32 and complex64, got ",
              scalar_type);
  TORCH_CHECK(c10::canCast(B.scalar_type(), scalar_type),
              "linalg.solve_triangular(): can't cast B of dtype ",
              B.scalar_type(),
              " to A's dtype ",
              scalar_type);
  // MPS ops get no generated device check and the solvers write into out= directly,
  // so its device and dtype have to be validated here.
  TORCH_CHECK(out.device() == A.device(),
              "linalg.solve_triangular(): Expected out tensor on ",
              A.device(),
              " but got ",
              out.device());
  TORCH_CHECK(out.scalar_type() == scalar_type,
              "linalg.solve_triangular(): Expected out tensor of dtype ",
              scalar_type,
              " but got ",
              out.scalar_type());
  Tensor A_t, B_t;
  std::tie(B_t, A_t) = _linalg_broadcast_batch_dims(B, A, /*don't check errors*/ nullptr);
  // Same layout as linalg_solve_triangular_out on CPU/CUDA and the meta: F-contiguous
  // unless A is a conjugate transpose.
  if (A_t.mT().is_contiguous() && A_t.is_conj()) {
    at::native::resize_output(out, B_t.sizes());
  } else if (at::native::resize_output_check(out, B_t.sizes())) {
    out.resize_(B_t.mT().sizes(), MemoryFormat::Contiguous);
    out.transpose_(-2, -1);
  }

  if (A.numel() == 0 || B.numel() == 0 || out.numel() == 0) {
    out.zero_();
    return;
  }

  // The substitution kernel reads raw elements, so materialize conjugated views.
  Tensor A_ = A_t.is_conj() ? A_t.resolve_conj() : A_t;
  if (!A_.is_contiguous()) {
    A_ = A_.clone(at::MemoryFormat::Contiguous);
  }

  // Fold the side into the transpose rather than materializing op(A): solving
  // X op(A) = B is the same as solving op(A)^T X^T = B^T, so the right case only
  // costs the O(nk) transposes of B and X, never an O(n^2) copy of A.
  const bool kernel_transpose = transpose != !left;
  // Substitution reads all of A once per right-hand side and runs a chain of n
  // dependent steps; the blocked solve shortens that chain to nb and turns the
  // rest into matmul, which wins for both dtypes. Below a handful of blocks the
  // per-block dispatches cost more than the substitution they replace.
  // nb trades a costlier diagonal-block inverse against fewer trailing matmuls.
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kMinBlocks = 4;
  const bool blocked = A_.size(-1) > kMinBlocks * kBlockSize;

  // Both solvers read B and write the solution Y of op(A) Y = B in either
  // dense row- or column-major layout. Y is out itself for a left solve and
  // out^T for a right one, so only a strided or conj/neg out needs a temporary.
  // The blocked solve is the exception: its panel updates are enough faster on
  // a row-major X to pay for the copy.
  const auto is_direct = [](const Tensor& t) { return is_row_or_column_contiguous(t) && !t.is_conj() && !t.is_neg(); };
  const Tensor Y = left ? out : out.mT();
  Tensor X = is_direct(Y) && (!blocked || Y.is_contiguous()) ? Y : at::empty(Y.sizes(), Y.options());
  // A right-hand side the solvers cannot read directly is staged in X and solved
  // in place: the blocked path works in place, and the kernel reads b[t] before
  // writing x[t] at every step, taking everything else from threadgroup memory.
  Tensor Brhs = left ? B_t : B_t.mT();
  // Like on CPU/CUDA, a B of another dtype is converted to A's here; autocast relies on it in backward.
  if (!is_direct(Brhs) || Brhs.scalar_type() != scalar_type) {
    X.copy_(Brhs);
    Brhs = X;
  }

  if (blocked) {
    Tensor M = kernel_transpose ? A_.mT() : A_;
    if (conjugate) {
      M = M.conj();
    }
    if (!X.is_same(Brhs)) {
      X.copy_(Brhs);
    }
    triangular_solve_blocked(M, upper != kernel_transpose, unitriangular, kBlockSize, X);
  } else {
    triangular_solve_metal(A_, Brhs, upper, kernel_transpose, conjugate, unitriangular, X);
  }
  if (!X.is_same(Y)) {
    Y.copy_(X);
  }
}

} // namespace mps

Tensor& linalg_solve_triangular_mps_out(const Tensor& A,
                                        const Tensor& B,
                                        bool upper,
                                        bool left,
                                        bool unitriangular,
                                        Tensor& out) {
  mps::linalg_solve_triangular_mps_impl(A, B, upper, /*transpose=*/false, left, unitriangular, out);
  return out;
}

Tensor linalg_solve_triangular_mps(const Tensor& A, const Tensor& B, bool upper, bool left, bool unitriangular) {
  Tensor out = at::empty({0}, A.options());
  mps::linalg_solve_triangular_mps_impl(A, B, upper, /*transpose=*/false, left, unitriangular, out);
  return out;
}

TORCH_IMPL_FUNC(triangular_solve_mps_out)
(const Tensor& self,
 const Tensor& A,
 bool upper,
 bool transpose,
 bool unitriangular,
 const Tensor& result,
 const Tensor& clone_A) {
  clone_A.copy_(A);
  mps::linalg_solve_triangular_mps_impl(A, self, upper, transpose, /*left=*/true, unitriangular, result);
}

} // namespace at::native
