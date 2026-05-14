#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/metal/common.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cdist_backward_native.h>
#include <ATen/ops/_cdist_forward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linalg_vector_norm.h>
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

Tensor _cdist_forward_mps(const Tensor& x1, const Tensor& x2, const double p, std::optional<int64_t> compute_mode) {
  TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
  TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
  TORCH_CHECK(x1.size(-1) == x2.size(-1),
              "X1 and X2 must have the same number of columns. X1: ",
              x1.size(-1),
              " X2: ",
              x2.size(-1));
  TORCH_CHECK(
      at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type());
  TORCH_CHECK(
      at::isFloatingType(x2.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type());
  TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");

  const int64_t mode = compute_mode.value_or(0);
  TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);

  // Promote fp16/bf16 to fp32 internally so the max-norm argmax selection
  // matches what the CPU reference does (which falls back to fp32 for
  // these dtypes anyway).
  const auto out_dtype = x1.scalar_type();
  const bool promote = at::isReducedFloatingType(out_dtype);
  const auto compute_dtype = promote ? at::kFloat : out_dtype;
  auto diff = x1.to(compute_dtype).unsqueeze(-2).sub(x2.to(compute_dtype).unsqueeze(-3));
  auto result = at::linalg_vector_norm(diff, p, makeArrayRef<int64_t>(-1), /*keepdim=*/false, /*dtype=*/std::nullopt);
  return promote ? result.to(out_dtype) : result;
}

Tensor _cdist_backward_mps(const Tensor& grad,
                           const Tensor& x1,
                           const Tensor& x2,
                           const double p,
                           const Tensor& cdist) {
  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t c = x1.size(-1);
  const std::vector<int64_t> expand_batch =
      infer_size(x1.sizes().slice(0, x1.dim() - 2), x2.sizes().slice(0, x2.dim() - 2));
  std::vector<int64_t> out_size(expand_batch);
  out_size.insert(out_size.end(), {r1, c});

  const int64_t batch_product = c10::multiply_integers(expand_batch);
  if (r1 == 0 || r2 == 0 || c == 0 || batch_product == 0 || p == 0.0) {
    return at::zeros(out_size, x1.options());
  }

  // p == 2: Euclidean identity, no kernel needed.
  if (p == 2.0) {
    Tensor coeff = at::where(cdist.eq(0), 0, grad.div(cdist));
    return x1.mul(coeff.sum(-1, /*keepdim=*/true)).sub(coeff.matmul(x2));
  }

  std::vector<int64_t> x2_expand_size(expand_batch);
  x2_expand_size.insert(x2_expand_size.end(), {r2, c});
  std::vector<int64_t> dist_expand_size(expand_batch);
  dist_expand_size.insert(dist_expand_size.end(), {r1, r2});

  const Tensor x1c = x1.expand(out_size).contiguous();
  const Tensor x2c = x2.expand(x2_expand_size).contiguous();
  const Tensor gradc = grad.expand(dist_expand_size).contiguous();
  // Kernel reads cdist as fp32. For p == inf with reduced precision the
  // fp16-rounded saved cdist would miss the argmax-c match, so recompute.
  const bool reduced = at::isReducedFloatingType(x1c.scalar_type());
  const Tensor cdistc = (std::isinf(p) && reduced)
      ? at::linalg_vector_norm(x1c.to(at::kFloat).unsqueeze(-2).sub(x2c.to(at::kFloat).unsqueeze(-3)),
                               p,
                               makeArrayRef<int64_t>(-1),
                               /*keepdim=*/false,
                               /*dtype=*/std::nullopt)
            .contiguous()
      : cdist.expand(dist_expand_size).to(at::kFloat).contiguous();
  const Tensor out = at::empty(out_size, x1.options());

  const CdistBwdParams params{
      .B = batch_product,
      .P = r1,
      .R = r2,
      .D = c,
      .p_minus_1 = static_cast<float>(p - 1.0),
  };

  // Values must match `P_KIND` in kernels/Cdist.metal.
  const int p_kind = std::isinf(p) ? 1 : (p == 1.0 ? 0 : 2);
  // TG_C choices must match REGISTER_CDIST_BACKWARD_FOR_TYPE in Cdist.metal.
  constexpr uint32_t kSmallTG = c10::metal::simdgroup_size;
  constexpr uint32_t kLargeTG = 4 * c10::metal::simdgroup_size;
  const uint32_t tg_c = (c >= 2 * kSmallTG) ? kLargeTG : kSmallTG;
  const auto D_padded = c10::metal::ceil_div(static_cast<uint32_t>(c), tg_c) * tg_c;

  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc(
        fmt::format("cdist_backward_{}_p{}_tg{}", scalarToMetalTypeString(x1c), p_kind, tg_c));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto compute_encoder = stream->commandEncoder();
        [compute_encoder setComputePipelineState:pso];
        mtl_setArgs(compute_encoder, x1c, x2c, gradc, cdistc, out, params);
        const MTLSize grid = MTLSizeMake(D_padded, static_cast<NSUInteger>(r1), static_cast<NSUInteger>(batch_product));
        const MTLSize tg = MTLSizeMake(tg_c, 1, 1);
        [compute_encoder dispatchThreads:grid threadsPerThreadgroup:tg];
      }
    });
  }

  return out;
}

} // namespace at::native
