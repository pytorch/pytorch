#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UnfoldBackward.h>
#include <ATen/native/mps/OperationUtils.h>

// Note on naming: it is unconventional.
// grad_in does not mean that it is a gradient wrt to input,
// grad_in/grad_out is just an input/output of unfold_backward kernel.
//
// unfold_backward, the algorithm is described in
// /native/cpu/UnfoldBackwardKernel.cpp

namespace at::native {
namespace {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/UnfoldBackward_metallib.h>
#endif

void unfold_backward_mps(Tensor& grad_out, const Tensor& grad_in, int64_t dim, int64_t size, int64_t step) {
  if (grad_in.numel() == 0) {
    return;
  }
  TORCH_CHECK(grad_in.ndimension() < 16, "unfold_backward_mps :Only up to 16-dim tensors supported");

  using namespace mps;
  dim = maybe_wrap_dim(dim, grad_out.dim());
  auto unfoldBackwardPSO = lib.getPipelineStateForFunc("unfold_backward_" + scalarToMetalTypeString(grad_in));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:unfoldBackwardPSO];
      std::array<uint32_t, 4> dim_size_step_ndim = {static_cast<uint32_t>(dim),
                                                    static_cast<uint32_t>(size),
                                                    static_cast<uint32_t>(step),
                                                    static_cast<uint32_t>(grad_out.ndimension())};
      mtl_setArgs(computeEncoder,
                  grad_in,
                  grad_out,
                  grad_in.strides(),
                  grad_out.sizes(),
                  grad_out.strides(),
                  dim_size_step_ndim);
      mtl_dispatch1DJob(computeEncoder, unfoldBackwardPSO, grad_out.numel());
    }
  });
}

} // anonymous namespace
REGISTER_DISPATCH(unfold_backward_stub, &unfold_backward_mps);
} // namespace at::native
