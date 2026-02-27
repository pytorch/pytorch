//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Cross.h>
#include <ATen/native/mps/OperationUtils.h>

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/CrossKernel_metallib.h>
#endif

void cross_mps_impl(const Tensor& out, const Tensor& input, const Tensor& other, int64_t dim) {
  // numThreads = number of cross triplets = numel / 3
  const auto numThreads = out.numel() / 3;
  // The dense kernel handles any contiguous layout: it uses dim_stride and a
  // closed-form index mapping that works for outermost, innermost, and any dim.
  const bool is_dense = out.is_contiguous() && input.is_contiguous() && other.is_contiguous();
  const std::string suffix = is_dense ? "dense" : "strided";

  // For the strided kernel, build squashed sizes (cross dim removed) for the
  // thread-index → position mapping, plus full element strides (all ndim dims,
  // including the cross dim) so the kernel can read strides[dim] directly.
  const int64_t ndim = out.dim();
  std::vector<int64_t> squashed_sizes, out_strides, input_strides, other_strides;
  if (!is_dense) {
    squashed_sizes.reserve(ndim - 1);
    out_strides.resize(ndim);
    input_strides.resize(ndim);
    other_strides.resize(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
      out_strides[d] = out.stride(d);
      input_strides[d] = input.stride(d);
      other_strides[d] = other.stride(d);
      if (d != dim) {
        squashed_sizes.push_back(out.size(d));
      }
    }
  }

  const auto mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto crossPSO = lib.getPipelineStateForFunc(fmt::format("cross_{}_{}", suffix, scalarToMetalTypeString(out)));
      getMPSProfiler().beginProfileKernel(crossPSO, "cross", {input, other});
      [computeEncoder setComputePipelineState:crossPSO];
      mtl_setArgs(computeEncoder, out, input, other);
      if (is_dense) {
        mtl_setArgs<3>(computeEncoder, static_cast<uint32_t>(out.stride(dim)));
      } else {
        mtl_setArgs<3>(computeEncoder,
                       squashed_sizes,
                       out_strides,
                       input_strides,
                       other_strides,
                       static_cast<uint32_t>(ndim),
                       static_cast<uint32_t>(dim));
      }
      mtl_dispatch1DJob(computeEncoder, crossPSO, numThreads);
      getMPSProfiler().endProfileKernel(crossPSO);
    }
  });
}
} // anonymous namespace

REGISTER_DISPATCH(cross_stub, &cross_mps_impl)
} // namespace at::native
