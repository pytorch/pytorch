//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif


namespace at {
namespace native {


Tensor dot_mps(
  const Tensor &self,
  const Tensor &other)
{
  using namespace mps;
  auto output = at::native::empty_mps({}, self.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);

  struct CachedGraph : public MPSCachedGraph
  {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor* selfTensor_ = nil;
      MPSGraphTensor* otherTensor_ = nil;
      MPSGraphTensor* outputTensor_ = nil;
  };
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string key = "dot_mps" + getTensorsStringKey({self, other});

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *selfTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor *otherTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, other);

          MPSGraphTensor *castSelf = nil;
          MPSGraphTensor *castOther = nil;

          if(self.scalar_type() == ScalarType::Short || self.scalar_type() == ScalarType::Byte
                                                     || self.scalar_type() == ScalarType::Char) {
            castSelf = [mpsGraph castTensor:selfTensor
                                     toType:MPSDataTypeInt32
                                       name:@"castSelfTensor"];
            castOther = [mpsGraph castTensor:otherTensor
                                      toType:MPSDataTypeInt32
                                        name:@"castOtherTensor"];
          } else {
            castSelf = selfTensor;
            castOther = otherTensor;
          }

          MPSGraphTensor *dot = [mpsGraph multiplicationWithPrimaryTensor: castSelf
                                                          secondaryTensor: castOther
                                                                     name: @"multiplication"];

          MPSGraphTensor *dotProductTensor = [mpsGraph reductionSumWithTensor: dot
                                                                         axes: nil
                                                                         name: @"dotProduct"];

          if(self.scalar_type() == ScalarType::Short || self.scalar_type() == ScalarType::Byte
                                                     || self.scalar_type() == ScalarType::Char)
            dotProductTensor = [mpsGraph castTensor:dotProductTensor
                                             toType:getMPSDataType(self.scalar_type())
                                               name:@"castDotProductTensor"];

          newCachedGraph->selfTensor_ = selfTensor;
          newCachedGraph->otherTensor_ = otherTensor;
          newCachedGraph->outputTensor_ = dotProductTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
    Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;
}

Tensor& addmv_out_mps_impl(
  const Tensor &self,
  const Tensor &mat,
  const Tensor &vec,
  const Scalar& beta_,
  const Scalar& alpha_,
  Tensor& result)
{
  using namespace mps;

  TORCH_CHECK(mat.is_mps());
  TORCH_CHECK(vec.is_mps());
  TORCH_CHECK(result.is_mps());
  TORCH_CHECK(self.is_mps());

  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta_.toComplexDouble();

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor_      = nil;
    MPSGraphTensor *matMulVecTensor_ = nil;
    MPSGraphTensor *outputTensor_    = nil;
  };
  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  MPSStream *stream = at::mps::getCurrentMPSStream();
  Tensor matMulVec = mm(mat, vec.unsqueeze(1)).squeeze(1);

  @autoreleasepool {
    string key = "addmv_out_mps_impl" + getTensorsStringKey({self, matMulVec})
                                       + ":" + to_string(beta_.toDouble())
                                       + ":" + to_string(alpha_.toDouble());
    CachedGraph* cachedGraph = nil;
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *matMulVecTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, matMulVec);
          MPSGraphTensor *selfTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, self);

          // Intermediates for beta and alpha
          MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar: alpha_.toDouble()
                                                            dataType: getMPSScalarType(mat.scalar_type())];

          // Intermediates for multiplying by beta and alpha
          MPSGraphTensor* productTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor:matMulVecTensor
                                                                              secondaryTensor:alphaTensor
                                                                                         name:@"MM/alpha*(mat@vec)"];
          newCachedGraph->outputTensor_ = productTimesAlphaTensor;

          if (betaval != 0.0)
          {
            MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar: beta_.toDouble()
                                                             dataType: getMPSScalarType(self.scalar_type())];

            MPSGraphTensor* selfTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor: selfTensor
                                                                            secondaryTensor: betaTensor
                                                                                       name: @"MM/beta*input"];

            MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor: productTimesAlphaTensor
                                                               secondaryTensor: selfTimesBetaTensor
                                                                          name: @"MM/beta*input + alpha*(mat@vec)"];

            newCachedGraph->outputTensor_ = outputTensor;
          }

          newCachedGraph->selfTensor_ = selfTensor;
          newCachedGraph->matMulVecTensor_ = matMulVecTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder matMulVecPlaceholder = Placeholder(cachedGraph->matMulVecTensor_, matMulVec);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =[NSMutableDictionary dictionary];
    feeds[matMulVecPlaceholder.getMPSGraphTensor()]   = matMulVecPlaceholder.getMPSGraphTensorData();
    if (betaval != 0.0)
    {
        Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
        feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    }

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}

TORCH_IMPL_FUNC(addmv_out_mps)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_, const Tensor& result) {
  addmv_out_mps_impl(self, mat, vec, beta_, alpha_, const_cast<Tensor&>(result));
}

} // namespace native
} // namespace at
