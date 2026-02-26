//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/MultiTensorApply.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale_native.h>
#include <ATen/ops/_amp_update_scale_native.h>
#endif

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Amp_metallib.h>
#endif
namespace mps {

static void _amp_non_finite_check_and_unscale_mps_single_impl(const Tensor& scaled_grad,
                                                              at::Tensor& found_inf,
                                                              const at::Tensor& inv_scale) {
  if (scaled_grad.numel() == 0) {
    return;
  }
  TORCH_CHECK(scaled_grad.is_mps(), "Tensor is not on the MPS device.");
  TORCH_CHECK(scaled_grad.numel() <= std::numeric_limits<uint32_t>::max(), "scaled_grad is too large");
  float inv_scale_val = inv_scale.item<float>();
  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto ampPipelineState =
      lib.getPipelineStateForFunc("ampNonFiniteCheckAndUnscaleSingle_" + mps::scalarToMetalTypeString(scaled_grad));

  const uint32_t threadsPerThreadgroup = 256;
  uint32_t numel = static_cast<uint32_t>(scaled_grad.numel());
  MTLSize threadGroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
  MTLSize gridSize = MTLSizeMake(numel, 1, 1);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    auto computeEncoder = stream->commandEncoder();
    [computeEncoder setComputePipelineState:ampPipelineState];
    mtl_setArgs(computeEncoder, scaled_grad, found_inf, inv_scale_val);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
  });
}

static void _amp_update_scale_mps_impl(Tensor& self,
                                       Tensor& growth_tracker,
                                       const Tensor& found_inf,
                                       float scale_growth_factor,
                                       float scale_backoff_factor,
                                       int32_t growth_interval) {
  auto stream = getCurrentMPSStream();
  auto ampUpdatePipelineState = lib.getPipelineStateForFunc("ampUpdateScale_" + mps::scalarToMetalTypeString(self));

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    auto computeEncoder = stream->commandEncoder();
    [computeEncoder setComputePipelineState:ampUpdatePipelineState];

    mtl_setArgs(
        computeEncoder, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
    mtl_dispatch1DJob(computeEncoder, ampUpdatePipelineState, 1);
  });
}

std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getAmpCPLState(const std::string& fname) {
  return {lib.getPipelineStateForFunc(fname), lib.getMTLFunction(fname)};
}
} // namespace mps

void _amp_foreach_non_finite_check_and_unscale_mps_(at::TensorList self,
                                                    at::Tensor& found_inf,
                                                    const at::Tensor& inv_scale) {
  if (self.size() == 0) {
    return;
  }
  TORCH_CHECK(inv_scale.is_mps(), "inv_scale must be a MPS tensor.");
  TORCH_CHECK(found_inf.is_mps(), "found_inf must be a MPS tensor.");
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");
  // Ensures client code (GradScaler) filtered scaled_grads by API restrictions.
  check_foreach_api_restrictions(self);

  // Prepare a vector of tensor lists.
  std::vector<std::vector<at::Tensor>> tensor_lists;
  if (can_use_fast_route(self)) {
    TORCH_CHECK(self[0].is_mps(), "scaled_grads must be MPS tensors.");
    tensor_lists.emplace_back(self.vec());
  } else {
    tensor_lists.resize(1);
    tensor_lists[0].reserve(self.size());
    auto expected_device = self[0].device();
    const auto expected_dtype = self[0].scalar_type();
    for (const at::Tensor& t : self) {
      // Ensure that GradScaler has filtered by device, layout, and dtype.
      TORCH_CHECK(t.is_mps(), "one of scaled_grads was not a MPS tensor.");
      TORCH_CHECK(t.device() == expected_device, "scaled_grads must be on the same device.");
      TORCH_CHECK(t.layout() == at::kStrided, "one of scaled_grads was not a strided tensor.");
      if (!t.is_non_overlapping_and_dense() || t.scalar_type() != expected_dtype) {
        // Fall back to the single-tensor implementation
        mps::_amp_non_finite_check_and_unscale_mps_single_impl(const_cast<at::Tensor&>(t), found_inf, inv_scale);
      } else {
        tensor_lists[0].push_back(t);
      }
    }
    if (tensor_lists[0].empty()) {
      return;
    }
  }

  std::string kernel_name =
      "ampNonFiniteCheckAndUnscale_" + mps::scalarToMetalTypeString(tensor_lists[0][0].scalar_type());
  mps::multi_tensor_apply<1>(kernel_name, tensor_lists, found_inf, inv_scale);
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