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

void unfold_backward_metal(
  Tensor& grad_out,
  const Tensor& grad_in,
  int64_t dim,
  int64_t size,
  int64_t step
) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds
  auto last_dim = maybe_wrap_dim(-1, grad_in.dim());

  auto grad_in_dim_stride = ensure_nonempty_stride(grad_in, dim);
  auto grad_in_last_dim_stride = ensure_nonempty_stride(grad_in, last_dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);

  auto grad_out_dim_stride = ensure_nonempty_stride(grad_out, dim);

  using namespace mps;
  auto unfoldBackwardPSO = lib.getPipelineStateForFunc("unfold_backward_" + scalarToMetalTypeString(grad_in));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:unfoldBackwardPSO];
      std::array<uint32_t, 4> dim_size_step_ndim = {static_cast<uint32_t>(dim), static_cast<uint32_t>(size), static_cast<uint32_t>(step), static_cast<uint32_t>(grad_out.ndimension())};
      mtl_setBuffer(computeEncoder, grad_in, 0);
      mtl_setBuffer(computeEncoder, grad_out, 1);
      mtl_setBytes(computeEncoder, grad_in.strides(), 2);
      mtl_setBytes(computeEncoder, grad_out.sizes(), 3);
      mtl_setBytes(computeEncoder, grad_out.strides(), 4);
      mtl_setBytes(computeEncoder, dim_size_step_ndim, 5);
      mtl_dispatch1DJob(computeEncoder, unfoldBackwardPSO, grad_out.numel());
    }
  });
}

} // anonymous namespace
REGISTER_DISPATCH(unfold_backward_stub, &unfold_backward_metal);
} // namespace at::native

