#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/eye.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/real.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#endif

#include <algorithm>
#include <string>
#include <string_view>

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SVD_metallib.h>
#endif

static Tensor mps_orthonormal_complement(const Tensor& proj, int64_t want) {
  const int64_t dim = proj.size(-1);
  auto bsh = proj.sizes().slice(0, proj.dim() - 2).vec();
  std::vector<int64_t> out_sh = bsh;
  out_sh.push_back(dim);
  out_sh.push_back(want);
  Tensor out = at::zeros(out_sh, proj.options());
  // Pivot on the largest residual column norm, so columns of proj already in the span (e.g. zero ones) are skipped.
  Tensor res = at::real(proj.diagonal(0, -2, -1)).unsqueeze(-2);
  int64_t got = 0;
  for (int64_t c = 0; c < dim && got < want; ++c) {
    Tensor v = proj.take_along_dim(res.argmax(-1, true), -1);
    if (got > 0) {
      Tensor Qc = out.narrow(-1, 0, got);
      Tensor coeff = Qc.mH().matmul(v);
      v = v.sub(Qc.matmul(coeff));
    }
    Tensor nrm = at::linalg_vector_norm(v, 2, {-2}, true);
    Tensor safe = nrm.clamp_min(1e-12);
    Tensor vn = v.div(safe.to(v.dtype()));
    out.narrow(-1, got, 1).copy_(vn);
    res = res.sub(vn.abs().square().mT());
    got += 1;
  }
  return out;
}

static void svd_kernel_mps(const Tensor& A,
                           const bool full_matrices,
                           const bool compute_uv,
                           const std::optional<std::string_view>& driver,
                           const Tensor& U,
                           const Tensor& S,
                           const Tensor& Vh,
                           const Tensor& info) {
  using namespace mps;
  // Metal has no float64; only float32 / complex64 run on GPU.
  TORCH_CHECK(A.scalar_type() == kFloat || A.scalar_type() == kComplexFloat,
              "linalg.svd: the MPS backend supports only float32 and complex64. Got ",
              A.scalar_type(),
              ". Move the tensor to CPU for other dtypes.");
  TORCH_CHECK(!driver.has_value(), "linalg.svd: driver= is not supported on MPS.");

  if (A.numel() == 0) {
    return;
  }

  const int64_t m = A.size(-2);
  const int64_t n = A.size(-1);
  const int64_t k = std::min(m, n);
  const int64_t batch = c10::multiply_integers(A.sizes().slice(0, A.dim() - 2));

  const int64_t elem_size = A.element_size();
  const int64_t wmax = std::max(m, n);
  const int64_t staging_bytes = wmax * k * elem_size;
  const int64_t tg_limit = static_cast<int64_t>([MPSDevice::getInstance()->device() maxThreadgroupMemoryLength]);
  const int64_t v_staging_bytes = k * k * elem_size;
  const bool stage_v = compute_uv && (staging_bytes + v_staging_bytes <= tg_limit);
  const bool too_large = (staging_bytes > tg_limit);

  // Only matrices too big to stage in threadgroup memory fall back to CPU. The
  // native kernel wins at every batch size for a tensor already on-device: the
  // CPU fallback pays two data transfers on top of the linalg.svd convergence
  // check's mandatory device->host sync, so gating small inputs to CPU was 3-6x
  // slower, not faster.
  if (too_large) {
    TORCH_WARN_ONCE("linalg.svd: matrix too large to stage in MPS threadgroup memory (",
                    staging_bytes,
                    " > ",
                    tg_limit,
                    " bytes); falling back to CPU.");
    auto [U_cpu, S_cpu, Vh_cpu] = at::linalg_svd(A.to(at::kCPU), full_matrices, driver);
    if (compute_uv) {
      const_cast<Tensor&>(U).copy_(U_cpu.to(at::kMPS));
      const_cast<Tensor&>(Vh).copy_(Vh_cpu.to(at::kMPS));
    }
    const_cast<Tensor&>(S).copy_(S_cpu.to(at::kMPS));
    return;
  }

  const bool transposed = m < n;
  // Kernel needs rows >= cols. For m<n run it on A^H: SVD(A^H)=(V,S,U^H), and
  // params.transposed tells the kernel to swap left/right into the right outputs.
  const int64_t wm = transposed ? n : m;
  // Kernel reads the raw buffer and ignores is_conj; resolve it, since
  // .contiguous() keeps the bit when the mH view is already contiguous.
  Tensor in = (transposed ? A.mH() : A).resolve_conj().contiguous().reshape({batch, wm, k});

  auto opts = A.options();
  const bool S_direct = S.is_contiguous() && S.scalar_type() == c10::toRealValueType(A.scalar_type());
  const bool info_direct = info.is_contiguous() && info.scalar_type() == kInt;
  Tensor S_k =
      S_direct ? S.reshape({batch, k}) : at::empty({batch, k}, opts.dtype(c10::toRealValueType(A.scalar_type())));
  // Device-memory accumulator only when V is not staged; tiny placeholder otherwise.
  Tensor Vacc_k = stage_v ? at::empty({1}, opts) : at::empty({batch, k, k}, opts);
  Tensor info_b = info_direct ? info.reshape({batch}) : at::empty({batch}, opts.dtype(kInt));
  // svdvals: U/Vh empty, so bind scratch for the kernel's (still-run) U writeback.
  Tensor U_scratch, Vh_scratch;
  if (!compute_uv) {
    U_scratch = at::empty({batch, wm, wm}, opts);
    Vh_scratch = at::empty({batch, wm, wm}, opts);
  }

  // Kernel writes straight into the column-major U/Vh outputs (first k cols/rows;
  // full_matrices fills the rest below).
  const int64_t u_ld = compute_uv ? U.size(-2) : wm;
  const int64_t u_bs = compute_uv ? U.size(-2) * U.size(-1) : wm * wm;
  const int64_t v_ld = compute_uv ? Vh.size(-2) : wm;
  const int64_t v_bs = compute_uv ? Vh.size(-2) * Vh.size(-1) : wm * wm;

  SvdParams params{static_cast<uint32_t>(wm),
                   static_cast<uint32_t>(k),
                   /*max_sweeps=*/30u,
                   /*tol=*/1e-6f,
                   static_cast<uint32_t>(u_ld),
                   static_cast<uint32_t>(u_bs),
                   static_cast<uint32_t>(v_ld),
                   static_cast<uint32_t>(v_bs),
                   compute_uv,
                   transposed,
                   stage_v};

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("svd_jacobi_{}", scalarToMetalTypeString(A)));
      getMPSProfiler().beginProfileKernel(pso, "svd_jacobi", {A}, stream);
      [enc setComputePipelineState:pso];
      Tensor Ubind = compute_uv ? U : U_scratch;
      Tensor Vbind = compute_uv ? Vh : Vh_scratch;
      mtl_setArgs(enc, in, Ubind, S_k, Vbind, Vacc_k, info_b, params);
      [enc setThreadgroupMemoryLength:wm * k * elem_size atIndex:0];
      [enc setThreadgroupMemoryLength:(stage_v ? k * k * elem_size : elem_size) atIndex:1];
      // One threadgroup per batch matrix; cap at 32 SIMD-groups (1024 threads).
      const NSUInteger simd = 32;
      const NSUInteger kMaxSimdGroups = 32;
      const NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
      const NSUInteger nPairs = (k + 1) / 2;
      const NSUInteger wantSG = std::min<NSUInteger>(std::max<NSUInteger>(nPairs, 1), kMaxSimdGroups);
      NSUInteger tgs = std::min<NSUInteger>(maxThreads, wantSG * simd);
      [enc dispatchThreads:MTLSizeMake(tgs * batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });

  if (!S_direct) {
    const_cast<Tensor&>(S).copy_(S_k.reshape(S.sizes()));
  }
  if (!info_direct) {
    const_cast<Tensor&>(info).copy_(info_b.reshape(info.sizes()));
  }

  if (compute_uv && full_matrices && (m != k || n != k)) {
    // full_matrices: complete cols k.. via an orthonormal basis of (I - Q Q^H).
    auto complete = [](Tensor M, int64_t kk) {
      const int64_t dim = M.size(-1);
      if (kk == dim)
        return;
      Tensor Q = M.narrow(-1, 0, kk);
      std::vector<int64_t> ish = M.sizes().slice(0, M.dim() - 2).vec();
      ish.push_back(dim);
      ish.push_back(dim);
      Tensor I = at::eye(dim, M.options()).expand(ish);
      Tensor proj = I.sub(Q.matmul(Q.mH()));
      // linalg_qr is float-only on MPS, so complex goes through Gram-Schmidt.
      Tensor comp;
      if (c10::isComplexType(proj.scalar_type())) {
        comp = mps_orthonormal_complement(proj, dim - kk);
      } else {
        comp = std::get<0>(at::linalg_qr(Q, "complete")).narrow(-1, kk, dim - kk);
      }
      M.narrow(-1, kk, dim - kk).copy_(comp);
    };
    if (m > k)
      complete(const_cast<Tensor&>(U), k);
    if (n > k)
      complete(const_cast<Tensor&>(Vh).mH(), k); // V = Vh^H
  }
}

static void lstsq_kernel_mps(const Tensor& a,
                             Tensor& b,
                             Tensor& rank,
                             Tensor& singular_values,
                             Tensor& infos,
                             double rcond,
                             std::string driver_name) {
  const auto scalar_type = a.scalar_type();
  const auto real_dtype = c10::toRealValueType(scalar_type);
  const auto m = a.size(-2);
  const auto n = a.size(-1);

  const bool sets_rank = (driver_name != "gels");
  const bool sets_singular_values = (driver_name == "gelsd" || driver_name == "gelss");

  const double rcond_value =
      rcond > 0 ? rcond : _get_epsilon(real_dtype) * static_cast<double>(std::max<int64_t>(m, n));

  // RHS occupies the first m rows of the (.., max(m, n), nrhs) buffer.
  Tensor rhs = b.narrow(-2, 0, m);

  Tensor U, S, Vh;
  std::tie(U, S, Vh) = at::linalg_svd(a, /*full_matrices=*/false);

  Tensor s_max = std::get<0>(S.max(/*dim=*/-1, /*keepdim=*/true));
  Tensor above = S.gt(s_max.mul(rcond_value));
  Tensor s_inv = at::where(above, S.reciprocal(), at::zeros({}, S.options()));

  Tensor tmp = at::matmul(U.mH(), rhs).mul(s_inv.unsqueeze(-1));
  Tensor solution = at::matmul(Vh.mH(), tmp); // (.., n, nrhs)

  if (m > n) {
    Tensor rss = at::matmul(a, solution).sub(rhs).abs().square().sum(/*dim=*/-2, /*keepdim=*/true);
    Tensor tail = b.narrow(-2, n, m - n);
    tail.zero_();
    tail.narrow(-2, 0, 1).copy_(rss.sqrt());
  }
  b.narrow(-2, 0, n).copy_(solution);

  if (sets_rank) {
    rank.copy_(above.sum(/*dim=*/-1).to(at::kLong));
  }
  if (sets_singular_values) {
    singular_values.copy_(S.to(real_dtype));
  }
  infos.zero_();
}

} // namespace mps

REGISTER_DISPATCH(svd_stub, mps::svd_kernel_mps);
REGISTER_DISPATCH(lstsq_stub, mps::lstsq_kernel_mps);

} // namespace at::native
