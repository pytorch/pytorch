//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
// For MTLLanguageVersion_3_1
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale_native.h>
#include <ATen/ops/_amp_update_scale_native.h>
#endif

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Amp_metallib.h>
#endif

static void _amp_foreach_non_finite_check_and_unscale_mps_impl(TensorList self,
                                                               at::Tensor& found_inf,
                                                               const at::Tensor& inv_scale) {
  found_inf.fill_(0);
  float inv_scale_val = inv_scale.item<float>();

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto ampPipelineState =
      lib.getPipelineStateForFunc("ampNonFiniteCheckAndUnscale_" + mps::scalarToMetalTypeString(self[0]));

  MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    auto computeEncoder = stream->commandEncoder();

    for (auto& scaled_grad : self) {
      TORCH_CHECK(scaled_grad.is_mps(), "Tensor is not on the MPS device.");
      if (scaled_grad.numel() == 0) {
        continue;
      }

      id<MTLBuffer> data_buffer = getMTLBufferStorage(scaled_grad);
      id<MTLBuffer> found_inf_buffer = getMTLBufferStorage(found_inf);

      uint32_t numel = static_cast<uint32_t>(scaled_grad.numel());

      uint32_t numThreadgroups = (numel + threadGroupSize.width - 1) / threadGroupSize.width;
      MTLSize gridSize = MTLSizeMake(numThreadgroups * threadGroupSize.width, 1, 1);
      [computeEncoder setComputePipelineState:ampPipelineState];
      mtl_setArgs(computeEncoder, data_buffer, found_inf_buffer, inv_scale_val, numel);
      [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
    }
  });
}

static void _amp_update_scale_mps_impl(Tensor& self,
                                       Tensor& growth_tracker,
                                       const Tensor& found_inf,
                                       double scale_growth_factor,
                                       double scale_backoff_factor,
                                       int64_t growth_interval) {
  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto ampUpdatePipelineState = lib.getPipelineStateForFunc("ampUpdateScale_" + mps::scalarToMetalTypeString(self));

  MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
  MTLSize gridSize = threadGroupSize;

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    auto computeEncoder = stream->commandEncoder();
    [computeEncoder setComputePipelineState:ampUpdatePipelineState];

    id<MTLBuffer> scaleBuffer = getMTLBufferStorage(self);
    id<MTLBuffer> growthTrackerBuffer = getMTLBufferStorage(growth_tracker);
    id<MTLBuffer> foundInfBuffer = getMTLBufferStorage(found_inf);
    float scaleGrowthFactorVal = static_cast<float>(scale_growth_factor);
    float scaleBackoffFactorVal = static_cast<float>(scale_backoff_factor);
    uint32_t growthIntervalVal = static_cast<uint32_t>(growth_interval);

    mtl_setArgs(computeEncoder,
                scaleBuffer,
                growthTrackerBuffer,
                foundInfBuffer,
                scaleGrowthFactorVal,
                scaleBackoffFactorVal,
                growthIntervalVal);
    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
  });
}
} // namespace mps

void _amp_foreach_non_finite_check_and_unscale_mps_(TensorList self,
                                                    at::Tensor& found_inf,
                                                    const at::Tensor& inv_scale) {
  mps::_amp_foreach_non_finite_check_and_unscale_mps_impl(self, found_inf, inv_scale);
}

Tensor& _amp_update_scale_mps_(Tensor& self,
                               Tensor& growth_tracker,
                               const Tensor& found_inf,
                               double scale_growth_factor,
                               double scale_backoff_factor,
                               int64_t growth_interval) {
  mps::_amp_update_scale_mps_impl(
      self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  return self;
}
} // namespace at::native