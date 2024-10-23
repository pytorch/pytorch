//  Copyright © 2022 Apple Inc.
#include <optional>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/EmptyTensor.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/tril_indices_native.h>
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_indices_native.h>
#include <ATen/ops/triu_native.h>
#endif

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/TriangularOps_metallib.h>
#endif

TORCH_IMPL_FUNC(triu_mps_out)
(const Tensor& self, int64_t k, const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return;
  }
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "triu_mps_out" + mps::getTensorsStringKey({self}) + ":" + std::to_string(k);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* outputTensor = nil;

      MPSGraphTensor* minusOneTensor = [mpsGraph constantWithScalar:-1 dataType:MPSDataTypeInt32];

      if (k > 0) {
        MPSGraphTensor* diagMinusOneTensor = [mpsGraph constantWithScalar:(k - 1) dataType:MPSDataTypeInt32];
        MPSGraphTensor* onesTensor = [mpsGraph constantWithScalar:1 dataType:MPSDataTypeInt32];
        onesTensor = [mpsGraph broadcastTensor:onesTensor toShape:inputTensor.shape name:nil];
        MPSGraphTensor* maskTensor = [mpsGraph bandPartWithTensor:onesTensor
                                                   numLowerTensor:minusOneTensor
                                                   numUpperTensor:diagMinusOneTensor
                                                             name:nil];
        outputTensor = [mpsGraph selectWithPredicateTensor:maskTensor
                                       truePredicateTensor:[mpsGraph constantWithScalar:0 dataType:inputTensor.dataType]
                                      falsePredicateTensor:inputTensor
                                                      name:nil];
      } else {
        MPSGraphTensor* minusDiagTensor = [mpsGraph constantWithScalar:(-k) dataType:MPSDataTypeInt32];
        outputTensor = [mpsGraph bandPartWithTensor:inputTensor
                                     numLowerTensor:minusDiagTensor
                                     numUpperTensor:minusOneTensor
                                               name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(tril_mps_out)
(const Tensor& self, int64_t k, const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return;
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "tril_mps_out" + mps::getTensorsStringKey({self}) + ":" + std::to_string(k);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* outputTensor = nil;

      MPSGraphTensor* minusOneTensor = [mpsGraph constantWithScalar:-1 dataType:MPSDataTypeInt32];

      if (k >= 0) {
        MPSGraphTensor* diagTensor = [mpsGraph constantWithScalar:k dataType:MPSDataTypeInt32];
        outputTensor = [mpsGraph bandPartWithTensor:inputTensor
                                     numLowerTensor:minusOneTensor
                                     numUpperTensor:diagTensor
                                               name:nil];
      } else {
        MPSGraphTensor* negDiagMinusOneTensor = [mpsGraph constantWithScalar:(-k - 1) dataType:MPSDataTypeInt32];
        MPSGraphTensor* complementTensor = [mpsGraph bandPartWithTensor:inputTensor
                                                         numLowerTensor:negDiagMinusOneTensor
                                                         numUpperTensor:minusOneTensor
                                                                   name:nil];
        outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor secondaryTensor:complementTensor name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

Tensor tril_indices_mps(int64_t row,
                        int64_t col,
                        int64_t offset,
                        std::optional<ScalarType> dtype_opt,
                        std::optional<Layout> layout_opt,
                        std::optional<Device> device_opt,
                        std::optional<bool> pin_memory_opt) {
  check_args(row, col, layout_opt);

  auto tril_size = get_tril_size(row, col, offset);
  auto tensor = at::detail::empty_mps({2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt, std::nullopt);
  if (tril_size <= 0) {
    return tensor;
  }
  auto m_first_row = offset > 0 ? std::min<int64_t>(col, 1 + offset) : // upper bounded by col
      row + offset > 0; // either 0 or 1
  auto trapezoid_row_offset = std::max<int64_t>(0, -offset);
  auto rectangle_row_offset = trapezoid_row_offset + col - m_first_row + 1;
  int64_t rectangle_size = 0;
  if (rectangle_row_offset < row) {
    rectangle_size = (row - rectangle_row_offset) * col;
  }
  using namespace mps;
  auto trilPSO = lib.getPipelineStateForFunc("tril_indices_" + scalarToMetalTypeString(tensor));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:trilPSO];
      mtl_setBuffer(computeEncoder, tensor, 0);
      mtl_setBytes(computeEncoder, trapezoid_row_offset, 1);
      mtl_setBytes(computeEncoder, m_first_row, 2);
      mtl_setBytes(computeEncoder, col, 3);
      mtl_setBytes(computeEncoder, tril_size - rectangle_size, 4);
      mtl_setBytes(computeEncoder, tril_size, 5);
      mtl_dispatch1DJob(computeEncoder, trilPSO, tril_size);
    }
  });

  return tensor;
}

Tensor triu_indices_mps(int64_t row,
                        int64_t col,
                        int64_t offset,
                        std::optional<ScalarType> dtype_opt,
                        std::optional<Layout> layout_opt,
                        std::optional<Device> device_opt,
                        std::optional<bool> pin_memory_opt) {
  check_args(row, col, layout_opt);

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);
  auto tensor = at::detail::empty_mps({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt, std::nullopt);
  if (triu_size <= 0) {
    return tensor;
  }
  // # of triu elements in the first row
  auto m_first_row = offset > 0 ? std::max<int64_t>(col - offset, 0) : // upper bounded by col
      col;

  // size of the top rectangle
  int64_t rectangle_size = 0;
  if (offset < 0) {
    rectangle_size = std::min<int64_t>(row, -offset) * col;
  }
  using namespace mps;
  auto triuPSO = lib.getPipelineStateForFunc("triu_indices_" + scalarToMetalTypeString(tensor));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:triuPSO];
      mtl_setBuffer(computeEncoder, tensor, 0);
      mtl_setBytes(computeEncoder, std::max<int64_t>(0, offset), 1);
      mtl_setBytes(computeEncoder, m_first_row, 2);
      mtl_setBytes(computeEncoder, col, 3);
      mtl_setBytes(computeEncoder, rectangle_size, 4);
      mtl_setBytes(computeEncoder, triu_size, 5);
      mtl_dispatch1DJob(computeEncoder, triuPSO, triu_size);
    }
  });

  return tensor;
}
} // namespace at::native
