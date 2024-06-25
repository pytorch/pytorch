//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/mm.h>
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
  TORCH_CHECK(self.scalar_type() != ScalarType::Long, "MPS: dot op doesn't support int64 input")

  using namespace mps;
  using CachedGraph = MPSBinaryCachedGraph;

  dot_check(self, other);

  auto output = at::empty({}, self.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string key = "dot_mps" + getTensorsStringKey({self, other});

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

static Tensor& addmv_out_mps_impl(const Tensor& self,
                                  const Tensor& mat,
                                  const Tensor& vec,
                                  const Scalar& beta_,
                                  const Scalar& alpha_,
                                  Tensor& result) {
  using namespace mps;

  TORCH_CHECK(mat.is_mps());
  TORCH_CHECK(vec.is_mps());
  TORCH_CHECK(result.is_mps());
  TORCH_CHECK(self.is_mps());

  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta_.toComplexDouble();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* matMulVecTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = at::mps::getCurrentMPSStream();
  Tensor matMulVec = at::mm(mat, vec.unsqueeze(1)).squeeze(1);

  @autoreleasepool {
    string key = "addmv_out_mps_impl" + getTensorsStringKey({self, matMulVec}) + ":" +
        std::to_string(beta_.toDouble()) + ":" + std::to_string(alpha_.toDouble());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* matMulVecTensor = mpsGraphRankedPlaceHolder(mpsGraph, matMulVec);
      MPSGraphTensor* selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      // Intermediates for beta and alpha
      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha_.toDouble()
                                                        dataType:getMPSScalarType(mat.scalar_type())];

      // Intermediates for multiplying by beta and alpha
      MPSGraphTensor* productTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor:matMulVecTensor
                                                                          secondaryTensor:alphaTensor
                                                                                     name:@"MM/alpha*(mat@vec)"];
      newCachedGraph->outputTensor_ = productTimesAlphaTensor;

      if (betaval != 0.0) {
        MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta_.toDouble()
                                                         dataType:getMPSScalarType(self.scalar_type())];

        MPSGraphTensor* selfTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:selfTensor
                                                                        secondaryTensor:betaTensor
                                                                                   name:@"MM/beta*input"];

        MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:productTimesAlphaTensor
                                                           secondaryTensor:selfTimesBetaTensor
                                                                      name:@"MM/beta*input + alpha*(mat@vec)"];

        newCachedGraph->outputTensor_ = outputTensor;
      }

      newCachedGraph->selfTensor_ = selfTensor;
      newCachedGraph->matMulVecTensor_ = matMulVecTensor;
    });

    Placeholder matMulVecPlaceholder = Placeholder(cachedGraph->matMulVecTensor_, matMulVec);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    feeds[matMulVecPlaceholder.getMPSGraphTensor()] = matMulVecPlaceholder.getMPSGraphTensorData();
    if (betaval != 0.0) {
      Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
      feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return result;
}

TORCH_IMPL_FUNC(addmv_out_mps)
(const Tensor& self,
 const Tensor& mat,
 const Tensor& vec,
 const Scalar& beta_,
 const Scalar& alpha_,
 const Tensor& result) {
  addmv_out_mps_impl(self, mat, vec, beta_, alpha_, const_cast<Tensor&>(result));
}

} // namespace at::native
