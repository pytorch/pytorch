#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Distance.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/metal/common.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cdist_backward.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/mps/kernels/Distance.h>
#include <cmath>

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Distance_metallib.h>
#endif

static void cdist_kernel_mps(Tensor& result, const Tensor& x1, const Tensor& x2, const double p) {
  // Promote fp16/bf16 to fp32 internally so the max-norm argmax selection
  // matches what the CPU reference does (which falls back to fp32 for
  // these dtypes anyway).
  const auto out_dtype = result.scalar_type();
  const bool promote = at::isReducedFloatingType(out_dtype);
  const auto compute_dtype = promote ? at::kFloat : out_dtype;
  auto diff = x1.to(compute_dtype).unsqueeze(-2).sub(x2.to(compute_dtype).unsqueeze(-3));
  auto out = at::linalg_vector_norm(diff, p, makeArrayRef<int64_t>(-1), /*keepdim=*/false, /*dtype=*/std::nullopt);
  if (promote) {
    out = out.to(out_dtype);
  }
  result.copy_(out.reshape(result.sizes()));
}

static void cdist_backward_kernel_mps(Tensor& result,
                                      const Tensor& grad,
                                      const Tensor& x1,
                                      const Tensor& x2,
                                      const double p,
                                      const Tensor& cdist) {
  if (p == 0.0) {
    result.fill_(0);
    return;
  }

  const int64_t batch_product = result.size(0);
  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t c = x1.size(-1);

  // p == 2: Euclidean identity, no kernel needed.
  if (p == 2.0) {
    Tensor coeff = at::where(cdist.eq(0), 0, grad.div(cdist));
    Tensor grad_x1 = x1.mul(coeff.sum(-1, /*keepdim=*/true)).sub(coeff.matmul(x2));
    result.copy_(grad_x1.reshape({batch_product, r1, c}));
    return;
  }

  // Kernel reads cdist as fp32. For p == inf with reduced precision the
  // fp16-rounded saved cdist would miss the argmax-c match, so recompute.
  const bool reduced = at::isReducedFloatingType(x1.scalar_type());
  const Tensor cdistc = (std::isinf(p) && reduced)
      ? at::linalg_vector_norm(x1.to(at::kFloat).unsqueeze(-2).sub(x2.to(at::kFloat).unsqueeze(-3)),
                               p,
                               makeArrayRef<int64_t>(-1),
                               /*keepdim=*/false,
                               /*dtype=*/std::nullopt)
            .contiguous()
      : cdist.to(at::kFloat).contiguous();

  const CdistBwdParams params{
      .B = batch_product,
      .P = r1,
      .R = r2,
      .D = c,
      .p_minus_1 = static_cast<float>(p - 1.0),
  };

  // Values must match `P_KIND` in kernels/Distance.metal.
  const int p_kind = std::isinf(p) ? 1 : (p == 1.0 ? 0 : 2);
  // TG_C choices must match REGISTER_CDIST_BACKWARD_FOR_TYPE in Distance.metal.
  constexpr uint32_t kSmallTG = c10::metal::simdgroup_size;
  constexpr uint32_t kLargeTG = 4 * c10::metal::simdgroup_size;
  const uint32_t tg_c = (c >= 2 * kSmallTG) ? kLargeTG : kSmallTG;
  const auto D_padded = c10::metal::ceil_div(static_cast<uint32_t>(c), tg_c) * tg_c;

  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc(
        fmt::format("cdist_backward_{}_p{}_tg{}", scalarToMetalTypeString(x1), p_kind, tg_c));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto compute_encoder = stream->commandEncoder();
        [compute_encoder setComputePipelineState:pso];
        mtl_setArgs(compute_encoder, x1, x2, grad, cdistc, result, params);
        const MTLSize grid = MTLSizeMake(D_padded, static_cast<NSUInteger>(r1), static_cast<NSUInteger>(batch_product));
        const MTLSize tg = MTLSizeMake(tg_c, 1, 1);
        [compute_encoder dispatchThreads:grid threadsPerThreadgroup:tg];
      }
    });
  }
}

static void pdist_forward_kernel_mps(Tensor& result, const Tensor& self, const double p) {
  const int64_t n = self.size(0);
  const PdistParams params{
      .N = n,
      .D = self.size(1),
      .p = static_cast<float>(p),
  };

  // Values must match `P_KIND` in kernels/Distance.metal.
  const int p_kind = (p == 0.0) ? 0 : (p == 1.0) ? 1 : (p == 2.0) ? 2 : (std::isinf(p) ? 3 : 4);

  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc(fmt::format("pdist_{}_p{}", scalarToMetalTypeString(self), p_kind));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto compute_encoder = stream->commandEncoder();
        [compute_encoder setComputePipelineState:pso];
        mtl_setArgs(compute_encoder, self, result, params);
        const auto width = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, static_cast<NSUInteger>(n));
        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(n), static_cast<NSUInteger>(n), 1);
        [compute_encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      }
    });
  }
}

// pdist is the strict upper triangle of cdist(self, self), and because the
// distance is symmetric in its two arguments, the total gradient of row i is
// the cdist gradient with respect to x1 once the per-pair gradients and
// distances are mirrored into full matrices. That lets the backward reuse the
// cdist backward kernel instead of duplicating it; the zero diagonal
// contributes nothing since that kernel skips zero coordinate differences.
static void pdist_backward_kernel_mps(Tensor& result,
                                      const Tensor& grad,
                                      const Tensor& self,
                                      const double p,
                                      const Tensor& dist) {
  const int64_t n = self.size(0);
  if (result.numel() == 0) {
    return;
  }
  if (p == 0.0 || n < 2 || self.size(1) == 0) {
    result.fill_(0);
    return;
  }

  const auto upper = at::ones({n, n}, self.options().dtype(at::kBool)).triu(1);

  auto grad_full = at::zeros({n, n}, self.options());
  grad_full.masked_scatter_(upper, grad);
  grad_full = grad_full.add(grad_full.t());

  auto dist_full = at::zeros({n, n}, self.options());
  dist_full.masked_scatter_(upper, dist);
  dist_full = dist_full.add(dist_full.t());

  result.copy_(at::_cdist_backward(grad_full.contiguous(), self, self, p, dist_full.contiguous()));
}

REGISTER_DISPATCH(cdist_stub, &cdist_kernel_mps)
REGISTER_DISPATCH(cdist_backward_stub, &cdist_backward_kernel_mps)
REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_mps)
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_mps)

} // namespace at::native
