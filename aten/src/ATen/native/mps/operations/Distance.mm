#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Distance.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/metal/common.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/where.h>
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
  const int64_t batches = x1.size(0);
  const int64_t x1_rows = x1.size(1);
  const int64_t x2_rows = x2.size(1);
  const int64_t dimensions = x1.size(2);
  const int p_kind = p == 0.0 ? 0 : p == 1.0 ? 1 : p == 2.0 ? 2 : std::isinf(p) ? 3 : 4;
  const bool scalar4 = p_kind != 4;
  const auto kernel =
      fmt::format("cdist_forward_{}{}_p{}", scalar4 ? "scalar4_" : "scalar_", scalarToMetalTypeString(x1), p_kind);
  const CdistFwdParams params{
      .B = batches,
      .P = x1_rows,
      .R = x2_rows,
      .D = dimensions,
      .p = static_cast<float>(p),
  };
  const NSUInteger inner = scalar4 ? c10::metal::ceil_div(x2_rows, int64_t{4}) : x2_rows;
  const NSUInteger outer = batches * x1_rows;
  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc(kernel);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto encoder = stream->commandEncoder();
        getMPSProfiler().beginProfileKernel(pso, "cdist_forward", {x1, x2});
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, x1, x2, result, params);
        mtl_dispatch2DJob(encoder, pso, inner, outer);
        getMPSProfiler().endProfileKernel(pso);
      }
    });
  }
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

REGISTER_DISPATCH(cdist_stub, &cdist_kernel_mps)
REGISTER_DISPATCH(cdist_backward_stub, &cdist_backward_kernel_mps)

} // namespace at::native
