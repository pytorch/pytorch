//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/IndexKernels.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Resize.h>
// For MTLLanguageVersion_3_1
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {
namespace mps {

static IntArrayRef updateTensorBaseShape(const Tensor& self) {
  IntArrayRef base_shape = getIMPSAllocator()->getBufferShape(self.storage().data());
  // if there's no base_shape stored in MPSAllocator, then infer it from tensor's size and store it
  if (base_shape.size() == 0) {
    // IntArrayRef wouldn't own the data, so we use a static storage
    static const int64_t shape_1d = 1;
    // self.sizes().size() could be zero
    base_shape = self.sizes().size()
        ? self.sizes()
        : ((self.is_view() && self._base().sizes().size()) ? self._base().sizes() : IntArrayRef(&shape_1d, 1));

    // base_shape will be retained in MPSAllocator until buffer gets recycled
    if (self.storage().data())
      getIMPSAllocator()->setBufferShape(self.storage().data(), base_shape);
  }
  return base_shape;
}

// For both scatter and gather kernels, there are 4 specized ones (for 1D to 4D tensor)
// and one generic, for 5+D ones. Assumption (to be tested) about specialized kernels
// is that reduction of n-dimentional vector, where n is 2, should be slower
// than reduction of 2D one, as n is not known at compiler time, therefore compiler
// could not do loop unrolls, that is
// float sum(float* v, int n) {
//   float rc = 0;
//   for (int idx = 0; idx < n; idx++)
//    rc += v[idx];
//   return rc;
// }
// would be slower than
// float sum2(float* v) { return v[0] + v[1]; }
//
// TODOS:
//   - Benchmark on whether or not this is really the case
//   - Instantiate specialized tensors from template
//   - Have proper error checking for 64-bit tensors
//   - Add flavors for 64-bit tensors
//   - Merged both scatter and gather templates together, as they more or less alike

static std::string getGatherScatterFunctionName(ScalarType scalarType, int64_t dim, bool needsScatter) {
  std::string kernelName = needsScatter ? "scatter" : "gather";
  return kernelName + "_kernel_" + (dim < 5 ? std::to_string(dim == 0 ? 1 : dim) : "n");
}

static std::string genScatterGatherCvtFunc(const std::string& dtypeSrc, const std::string& dtypeDst, bool needsConj) {
  const bool srcComplex = dtypeSrc[dtypeSrc.size() - 1] == '2';
  const bool dstComplex = dtypeDst[dtypeDst.size() - 1] == '2';
  if (dstComplex) {
    // TODO: Document why explicit cast is needed only for bfloat types
    if (dtypeSrc == "bfloat") {
      return dtypeDst + "(float(x), 0.0)";
    }
    return dtypeDst + (srcComplex ? needsConj ? "(x.x, -x.y)" : "(x.x, x.y)" : "(x,  0.0)");
  }
  if (srcComplex) {
    // TODO: Document why explicit cast is needed only for bfloat types
    if (dtypeDst == "bfloat") {
      return "bfloat(x.x)";
    }
    return "x.x";
  }
  // TODO: Document why explicit cast is needed only for bfloat types
  if (dtypeDst == "bfloat") {
    return "bfloat(x)";
  }
  return dtypeSrc == "bfloat" ? dtypeDst + "(x)" : "(x)";
}

static MetalShaderLibrary scatterLib(SCATTER_OPS_TEMPLATE, 3);
static MetalShaderLibrary gatherLib(GATHER_OPS_TEMPLATE, 3);

static id<MTLComputePipelineState> getPipelineState(const std::string& kernel,
                                                    const std::string& dtypeSrc,
                                                    const std::string& dtypeDst,
                                                    bool needsScatter,
                                                    bool needsConj) {
  auto cvtFunc = genScatterGatherCvtFunc(dtypeSrc, dtypeDst, needsConj);
  return (needsScatter ? scatterLib : gatherLib).getPipelineStateForFunc(kernel, {dtypeSrc, dtypeDst, cvtFunc});
}

Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst) {
  Tensor output = dst;
  if (!dst.has_storage()) {
    output = at::empty(src.sizes(), src.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  }

  if (src.numel() == 0 || output.numel() == 0) {
    return dst;
  }

  uint32_t numThreads = output.numel();

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
    std::string functionName = getGatherScatterFunctionName(output.scalar_type(), output.dim(), /*needsScatter=*/false);
    id<MTLComputePipelineState> gatherPSO = getPipelineState(functionName,
                                                             scalarToMetalTypeString(src),
                                                             scalarToMetalTypeString(output),
                                                             /*needsScatter=*/false,
                                                             src.is_conj() != dst.is_conj());

    // this function call is a no-op if MPS Profiler is not enabled
    getMPSProfiler().beginProfileKernel(gatherPSO, functionName, {src, output});

    uint32_t kernel_size = src.sizes().size();
    std::vector<uint32_t> src_sizes(kernel_size == 0 ? 1 : kernel_size);
    std::vector<uint32_t> src_strides(kernel_size == 0 ? 1 : kernel_size);

    if (kernel_size == 0) {
      src_sizes[0] = src_strides[0] = 1;
    } else {
      for (const auto i : c10::irange(kernel_size)) {
        src_sizes[i] = (uint32_t)(src.sizes()[i]);
        src_strides[i] = (uint32_t)(src.strides()[i]);
      }
    }

    [computeEncoder setComputePipelineState:gatherPSO];
    mtl_setArgs(computeEncoder, src, dst.has_storage() ? dst : output, src_sizes, src_strides, numThreads);
    if (src.dim() > 4) {
      mtl_setBytes<int32_t>(computeEncoder, src.dim(), 5);
    }
    mtl_dispatch1DJob(computeEncoder, gatherPSO, numThreads);

    getMPSProfiler().endProfileKernel(gatherPSO);
  });

  return (dst.has_storage()) ? dst : output;
}

Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output) {
  if (src.numel() == 0 || output.numel() == 0) {
    return output;
  }

  uint32_t numThreads = src.numel();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      std::string functionName =
          getGatherScatterFunctionName(output.scalar_type(), output.dim(), /*needsScatter=*/true);
      id<MTLComputePipelineState> scatterPSO = getPipelineState(functionName,
                                                                scalarToMetalTypeString(src),
                                                                scalarToMetalTypeString(output),
                                                                /*needsScatter=*/true,
                                                                src.is_conj() != output.is_conj());

      getMPSProfiler().beginProfileKernel(scatterPSO, functionName, {src, output});

      uint32_t kernel_size = output.sizes().size();
      std::vector<uint32_t> output_sizes(kernel_size == 0 ? 1 : kernel_size);
      std::vector<uint32_t> output_strides(kernel_size == 0 ? 1 : kernel_size);

      if (kernel_size == 0) {
        output_sizes[0] = output_strides[0] = 1;
      } else {
        for (const auto i : c10::irange(kernel_size)) {
          output_sizes[i] = (uint32_t)(output.sizes()[i]);
          output_strides[i] = (uint32_t)(output.strides()[i]);
        }
      }

      [computeEncoder setComputePipelineState:scatterPSO];
      mtl_setArgs(computeEncoder, src, output, output_sizes, output_strides, numThreads);
      if (output.dim() > 4) {
        mtl_setBytes<int32_t>(computeEncoder, output.dim(), 5);
      }
      mtl_dispatch1DJob(computeEncoder, scatterPSO, numThreads);

      getMPSProfiler().endProfileKernel(scatterPSO);
    }
  });

  return output;
}

} // namespace mps

// implementation of as_strided() op
Tensor as_strided_tensorimpl_mps(const Tensor& self,
                                 IntArrayRef size,
                                 IntArrayRef stride,
                                 std::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result =
      detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);

  // creating the view graph will be deferred until gatherViewTensor() or scatterViewTensor() are called.
  // In as_strided, we just update the base shape of the buffer in order to retrieve it later
  // when we create/run the view graph.
  IntArrayRef base_shape = mps::updateTensorBaseShape(self);
  TORCH_INTERNAL_ASSERT(
      base_shape.size() > 0, "Failed to update the base shape of tensor's buffer at ", self.storage().data());

  return result;
}

} // namespace at::native
