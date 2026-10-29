//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/vdot_native.h>
#endif

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace at::native {

namespace mps {

inline void dot_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.dim() == 1 && other.dim() == 1,
              "1D tensors expected, but got ",
              self.dim(),
              "D and ",
              other.dim(),
              "D tensors");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(),
              "dot : expected both vectors to have same dtype, but found ",
              self.scalar_type(),
              " and ",
              other.scalar_type());
  TORCH_CHECK(self.numel() == other.numel(),
              "inconsistent tensor size, expected tensor [",
              self.numel(),
              "] and src [",
              other.numel(),
              "] to have the same number of elements, but got ",
              self.numel(),
              " and ",
              other.numel(),
              " elements respectively");
  TORCH_CHECK(self.device() == other.device(),
              "Expected all tensors to be on the same device. Found: ",
              self.device(),
              ", ",
              other.device());
}
} // namespace mps

Tensor dot_mps(const Tensor& self, const Tensor& other) {
  using namespace mps;
  using CachedGraph = MPSBinaryCachedGraph;

  if (self.numel() == 0 & other.numel() == 0) {
    return zeros({}, self.options());
  }

  dot_check(self, other);

  auto output = at::empty({}, self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "dot_mps" + getTensorsStringKey({self, other});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* otherTensor = mpsGraphRankedPlaceHolder(mpsGraph, other);

      MPSGraphTensor* castSelf = nil;
      MPSGraphTensor* castOther = nil;

      if (self.scalar_type() == ScalarType::Short || self.scalar_type() == ScalarType::Byte ||
          self.scalar_type() == ScalarType::Char) {
        castSelf = [mpsGraph castTensor:selfTensor toType:MPSDataTypeInt32 name:@"castSelfTensor"];
        castOther = [mpsGraph castTensor:otherTensor toType:MPSDataTypeInt32 name:@"castOtherTensor"];
      } else {
        castSelf = selfTensor;
        castOther = otherTensor;
      }
      if (self.is_conj()) {
        castSelf = [mpsGraph conjugateWithTensor:selfTensor name:nil];
      }
      if (other.is_conj()) {
        castOther = [mpsGraph conjugateWithTensor:otherTensor name:nil];
      }

      MPSGraphTensor* dot = [mpsGraph multiplicationWithPrimaryTensor:castSelf
                                                      secondaryTensor:castOther
                                                                 name:@"multiplication"];

      MPSGraphTensor* dotProductTensor = [mpsGraph reductionSumWithTensor:dot axes:nil name:@"dotProduct"];

      if (self.scalar_type() == ScalarType::Short || self.scalar_type() == ScalarType::Byte ||
          self.scalar_type() == ScalarType::Char)
        dotProductTensor = [mpsGraph castTensor:dotProductTensor
                                         toType:getMPSDataType(self)
                                           name:@"castDotProductTensor"];

      newCachedGraph->inputTensor_ = selfTensor;
      newCachedGraph->otherTensor_ = otherTensor;
      newCachedGraph->outputTensor_ = dotProductTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, otherPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output;
}

Tensor vdot_mps(const Tensor& self, const Tensor& other) {
  // For real dtypes, vdot is identical to dot
  if (!self.is_complex()) {
    return dot_mps(self, other);
  }

  return dot_mps(self.conj(), other);
}

TORCH_IMPL_FUNC(addmv_out_mps)
(const Tensor& self,
 const Tensor& mat,
 const Tensor& vec,
 const Scalar& beta_,
 const Scalar& alpha_,
 const Tensor& result) {
  auto out = result.unsqueeze(1);
  at::addmm_out(out, self.expand({mat.size(0)}).unsqueeze(1), mat, vec.unsqueeze(1), beta_, alpha_);
}

} // namespace at::native
