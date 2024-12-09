#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bucketize_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/searchsorted_native.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Bucketization_metallib.h>
#endif

static void searchsorted_mps_contiguous(Tensor& result,
                                        const Tensor& input,
                                        const Tensor& boundaries,
                                        const bool right,
                                        const Tensor& sorter) {
  TORCH_INTERNAL_ASSERT(input.is_contiguous());
  TORCH_INTERNAL_ASSERT(boundaries.is_contiguous());
  TORCH_INTERNAL_ASSERT(!sorter.defined() || sorter.is_contiguous());

  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();
  int64_t right_i64 = right;
  int64_t is_1d_boundaries = boundaries.dim() == 1;

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const std::string kernel = "searchsorted_" + scalarToMetalTypeString(input) + "_" +
          scalarToMetalTypeString(result) + (sorter.defined() ? "_sorter" : "");
      id<MTLComputePipelineState> bucketizationPSO = lib.getPipelineStateForFunc(kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(bucketizationPSO, kernel, {input, boundaries, sorter});

      [computeEncoder setComputePipelineState:bucketizationPSO];
      mtl_setArgs(computeEncoder, input, boundaries, result, idim_in, idim_bd, numel_in, right_i64, is_1d_boundaries);
      if (sorter.defined())
        mtl_setBuffer(computeEncoder, sorter, 8);

      // A threadGroup is equivalent to a cuda's block.
      int64_t maxThreadgroups = 1024;
      int64_t maxThreads = bucketizationPSO.maxTotalThreadsPerThreadgroup;
      NSUInteger tgSize = std::min(maxThreads, numel_in);
      MTLSize threadgroupsPerGrid = MTLSizeMake(std::min(maxThreadgroups, ceil_div<int64_t>(numel_in, tgSize)), 1, 1);
      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(bucketizationPSO);
    }
  });
}
} // namespace mps

Tensor& searchsorted_out_mps(const Tensor& sorted_sequence,
                             const Tensor& self,
                             bool out_int32,
                             bool right,
                             const std::optional<std::string_view> side_opt,
                             const std::optional<Tensor>& sorter_opt,
                             Tensor& result) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto sorter_maybe_owned = at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  searchsorted_pre_check(sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  resize_output(result, self.sizes());

  // we have two inputs to set right, pre_check checks that they aren't set to opposites
  right |= (side_opt && *side_opt == "right");
  if (self.numel() == 0) {
    return result;
  }

  // for non-contiguous result tensors, we write the output to a contiguous copy so we can later copy back, maintaining
  // the original result tensor
  Tensor out = result.contiguous();

  if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype() &&
      sorter.is_contiguous()) {
    mps::searchsorted_mps_contiguous(out, self, sorted_sequence, right, sorter);
  } else {
    Tensor trimmed_input;
    Tensor trimmed_boundaries;
    Tensor trimmed_sorter;
    searchsorted_maybe_trim_input_tensors(
        trimmed_input, trimmed_boundaries, trimmed_sorter, self, sorted_sequence, sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter = trimmed_sorter.defined() ? trimmed_sorter : sorter;
    mps::searchsorted_mps_contiguous(out, final_input, final_boundaries, right, final_sorter);
  }

  // if result is non-contiguous, we wrote the answer to a copied version, so we copy back to the original result tensor
  if (!result.is_contiguous()) {
    result.copy_(out);
  }
  return result;
}

Tensor& searchsorted_out_mps(const Tensor& sorted_sequence,
                             const Scalar& self,
                             bool out_int32,
                             bool right,
                             const std::optional<std::string_view> side_opt,
                             const std::optional<Tensor>& sorter_opt,
                             Tensor& result) {
  const Tensor& scalar_tensor = mps::wrapped_scalar_tensor_mps(self, sorted_sequence.device());
  return searchsorted_out_mps(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt, result);
}

Tensor searchsorted_mps(const Tensor& sorted_sequence,
                        const Tensor& self,
                        bool out_int32,
                        bool right,
                        const std::optional<std::string_view> side_opt,
                        const std::optional<Tensor>& sorter) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::searchsorted_out_mps(sorted_sequence, self, out_int32, right, side_opt, sorter, result);
  return result;
}

Tensor searchsorted_mps(const Tensor& sorted_sequence,
                        const Scalar& self,
                        bool out_int32,
                        bool right,
                        const std::optional<std::string_view> side_opt,
                        const std::optional<Tensor>& sorter) {
  const Tensor& scalar_tensor = mps::wrapped_scalar_tensor_mps(self, sorted_sequence.device());
  return searchsorted_mps(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter);
}

Tensor& bucketize_out_mps(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right, Tensor& result) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  at::native::searchsorted_out_mps(boundaries, self, out_int32, right, std::nullopt, std::nullopt, result);
  return result;
}

Tensor bucketize_mps(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::bucketize_out_mps(self, boundaries, out_int32, right, result);
  return result;
}

Tensor bucketize_mps(const Scalar& self, const Tensor& boundaries, bool out_int32, bool right) {
  return bucketize_mps(mps::wrapped_scalar_tensor_mps(self, boundaries.device()), boundaries, out_int32, right);
}

} // namespace at::native
