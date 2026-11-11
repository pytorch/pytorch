//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Cross.h>
#include <ATen/native/mps/OperationUtils.h>
#include <utility>

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/CrossKernel_metallib.h>
#endif

// Each iteration of `iter` (cross dim squashed to size 1) handles one triplet; dim_strides are
// the element strides along the cross dim of out, input and other.
void cross_strided_mps(TensorIteratorBase& iter, const std::array<int64_t, 3>& dim_strides) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      cross_strided_mps(sub_iter, dim_strides);
    }
    return;
  }
  const auto mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto crossPSO = lib.getPipelineStateForFunc("cross_strided_" + scalarToMetalTypeString(iter.output()));
      getMPSProfiler().beginProfileKernel(crossPSO, "cross", {iter.input(0), iter.input(1)}, mpsStream);
      [computeEncoder setComputePipelineState:crossPSO];
      bind_iter_tensors(computeEncoder, iter);
      mtl_setArgs<3>(computeEncoder,
                     dimDividers(iter.shape()),
                     iter.strides(0),
                     iter.strides(1),
                     iter.strides(2),
                     static_cast<uint32_t>(iter.ndim()),
                     dim_strides);
      mtl_dispatch1DJob(computeEncoder, crossPSO, iter.numel());
      getMPSProfiler().endProfileKernel(crossPSO, mpsStream);
    }
  });
}

void cross_mps_impl(const Tensor& out, const Tensor& input, const Tensor& other, int64_t dim) {
  TORCH_CHECK_NOT_IMPLEMENTED(out.scalar_type() != at::kBool, "linalg.cross: not implemented for 'Bool'");
  // The Metal kernel is selected by the output dtype and binds all tensors as raw buffers
  TORCH_CHECK(input.scalar_type() == other.scalar_type() && input.scalar_type() == out.scalar_type(),
              "linalg.cross: expected input, other and out to have the same dtype, but got ",
              input.scalar_type(),
              ", ",
              other.scalar_type(),
              " and ",
              out.scalar_type());
  // numThreads = number of cross triplets = numel / 3
  const auto numThreads = out.numel() / 3;
  // The dense kernel runs one dispatch, so contiguous inputs with more triplets than a grid holds are split
  // by the iterator path like strided ones.
  if (!out.is_contiguous() || !input.is_contiguous() || !other.is_contiguous() ||
      !std::in_range<uint32_t>(numThreads)) {
    auto iter = TensorIteratorConfig()
                    .add_output(out)
                    .add_const_input(input)
                    .add_const_input(other)
                    .resize_outputs(false)
                    .declare_static_shape(out.sizes(), /*squash_dims=*/dim)
                    .build();
    if (iter.numel() > 0) {
      cross_strided_mps(iter, {out.stride(dim), input.stride(dim), other.stride(dim)});
    }
    return;
  }

  // The dense kernel handles any contiguous layout: it uses dim_stride and a
  // closed-form index mapping that works for outermost, innermost, and any dim.
  const auto mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto crossPSO = lib.getPipelineStateForFunc("cross_dense_" + scalarToMetalTypeString(out));
      getMPSProfiler().beginProfileKernel(crossPSO, "cross", {input, other}, mpsStream);
      [computeEncoder setComputePipelineState:crossPSO];
      mtl_setArgs(computeEncoder, out, input, other, static_cast<uint32_t>(out.stride(dim)));
      mtl_dispatch1DJob(computeEncoder, crossPSO, numThreads);
      getMPSProfiler().endProfileKernel(crossPSO, mpsStream);
    }
  });
}
} // anonymous namespace

REGISTER_DISPATCH(cross_stub, &cross_mps_impl)
} // namespace at::native
