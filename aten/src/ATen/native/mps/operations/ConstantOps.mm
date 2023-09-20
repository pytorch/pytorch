//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/zero_native.h>

namespace at::native {

static Tensor& fill_scalar_mps_impl(Tensor& self, const Scalar& value) {
  using namespace mps;

  if (self.numel() == 0) {
    return self;
  }
  Tensor output = self;
  bool needsCopyToOutput = false;
  if (!self.is_contiguous() || self.storage_offset()) {
    output = at::empty(self.sizes(), self.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
    needsCopyToOutput = true;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    string key = "fill_scalar_mps_impl" + getTensorsStringKey(self) + ":" + to_string(value.toDouble());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto isBool = self.scalar_type() == c10::ScalarType::Bool;
      auto isUInt8 = self.scalar_type() == c10::ScalarType::Byte;
      auto dataType = !isUInt8 ? !isBool ? getMPSScalarType(self.scalar_type()) : MPSDataTypeInt8 : MPSDataTypeUInt32;
      // constantWithScalar does not work for boolTypes on MacOS-12.[34]
      // workaround by filing it as int8 tensor and than casting to bool
      // See https://github.com/pytorch/pytorch/issues/82427
      // constantWithScalar does not work for UInt8 Types on MacOS-12.[34]/Ventura preview
      // workaround by filing it as uint32 tensor and than casting to uint8
      // See https://github.com/pytorch/pytorch/issues/83692
      MPSGraphTensor* inputTensor = [mpsGraph constantWithScalar:value.toDouble()
                                                           shape:getMPSShape(self)
                                                        dataType:dataType];
      MPSGraphTensor* outputTensor = [mpsGraph identityWithTensor:inputTensor name:nil];
      if (isBool) {
        outputTensor = [mpsGraph castTensor:outputTensor toType:MPSDataTypeBool name:@"constWithBool-workaround"];
      }
      if (isUInt8) {
        outputTensor = [mpsGraph castTensor:outputTensor toType:MPSDataTypeUInt8 name:@"constWithUInt8-workaround"];
      }

      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, needsCopyToOutput ? output : self, nullptr, !needsCopyToOutput);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};

    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), /*feeds*/ nil, results);

    if (needsCopyToOutput) {
      self.copy_(output);
    }
  }

  return self;
}

// returns false if tensor cannot be filled with fillBuffer()
static bool fill_mps_tensor_(Tensor& self, uint8_t value) {
  if (self.is_contiguous()) {
    MPSStream* stream = getCurrentMPSStream();
    auto storage_byte_offset = self.storage_offset() * self.itemsize();
    stream->fill(mps::getMTLBufferStorage(self), 0, self.storage().nbytes(), storage_byte_offset);
    return true;
  }
  return false;
}

Tensor& zero_mps_(Tensor& self) {
  // check if it's possible to use fillBuffer() to fill the Tensor's storage
  if (fill_mps_tensor_(self, 0) == true)
    return self;
  return fill_scalar_mps_impl(self, 0.0f);
}

Tensor& fill_scalar_mps(Tensor& self, const Scalar& value) {
  if (value.toDouble() == 0.0 && fill_mps_tensor_(self, 0) == true)
    return self;
  return fill_scalar_mps_impl(self, value);
}

Tensor& fill_tensor_mps_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0,
              "fill_ only supports 0-dimension value tensor but got tensor with ",
              value.dim(),
              " dimensions.");
  Scalar scalar_value = value.item();
  if (scalar_value.toDouble() == 0.0 && fill_mps_tensor_(self, 0) == true)
    return self;
  return fill_scalar_mps_impl(self, scalar_value);
}

} // namespace at::native
