//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace mps {

typedef MPSGraphTensor* (^UnaryOpBlock)(MPSGraph*, MPSGraphTensor*);

void unary_op(const Tensor& self, const Tensor& output, std::string op_name, UnaryOpBlock unaryBlock)
{
  TORCH_CHECK_TYPE(self.scalar_type() != ScalarType::Long, "Operation '", op_name, "()' does not support input type 'int64' in MPS backend.");
  if (!output.is_same_size(self)) {
    output.resize_(self.sizes());
  }
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
  };
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self}, /*use_scalar_value*/ false);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph* () {
        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);
          newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* castTensor = newCachedGraph->inputTensor;
          // Integer input must be cast to float if output is float
          if (isIntegralType(self.scalar_type()) && isFloatingType(output.scalar_type())) {
            castTensor = castMPSTensor(mpsGraph, newCachedGraph->inputTensor, output.scalar_type());
          }
          newCachedGraph->outputTensor = unaryBlock(mpsGraph, castTensor);
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, output);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
  }
}

MPSGraphTensor* trunc_tensor(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)
{
  MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0
                                                   dataType:inputTensor.dataType];
  MPSGraphTensor* predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                        secondaryTensor:zeroTensor
                                                                    name:nil];
  return [mpsGraph selectWithPredicateTensor:predicateTensor
                         truePredicateTensor:[mpsGraph ceilWithTensor :inputTensor name:nil]
                        falsePredicateTensor:[mpsGraph floorWithTensor:inputTensor name:nil]
                                        name:nil];
};

} // namespace mps

TORCH_IMPL_FUNC(trunc_out_mps) (const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "trunc_out_mps",
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)
                  { return mps::trunc_tensor(mpsGraph, inputTensor); });
}

#define CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)              \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& output) {                \
  mps::unary_op(self, output, #func_out,                                              \
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)   \
                  { return [mpsGraph func_stub##WithTensor:inputTensor name:nil]; }); \
}

#define CREATE_MPS_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                         \
Tensor& func_out(const Tensor& self, Tensor& output) {                                \
  mps::unary_op(self, output, #func_out,                                              \
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)   \
                  { return [mpsGraph func_stub##WithTensor:inputTensor name:nil]; }); \
  return output;                                                                      \
}


CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(exp_out_mps, exponent)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(exp2_out_mps, exponentBase2)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(reciprocal_out_mps, reciprocal)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sqrt_out_mps, squareRoot)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(rsqrt_out_mps, reverseSquareRoot)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sign_out_mps, sign)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(neg_out_mps, negative)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log_out_mps, logarithm)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log10_out_mps, logarithmBase10)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log2_out_mps, logarithmBase2)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(ceil_out_mps, ceil)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(floor_out_mps, floor)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(round_out_mps, round)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(erf_out_mps, erf)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sin_out_mps, sin)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(cos_out_mps, cos)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(tan_out_mps, tan)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(asin_out_mps, asin)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(acos_out_mps, acos)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(atan_out_mps, atan)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sinh_out_mps, sinh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(cosh_out_mps, cosh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(tanh_out_mps, tanh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(asinh_out_mps, asinh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(acosh_out_mps, acosh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(atanh_out_mps, atanh)

CREATE_MPS_UNARY_TORCH_IMPL_FUNC(abs_out_mps, absolute)
CREATE_MPS_UNARY_TORCH_IMPL_FUNC(logical_not_out_mps, not)

TORCH_IMPL_FUNC(log1p_out_mps) (const Tensor& self, const Tensor& output)
{
    using namespace mps;
    if (!output.is_same_size(self)) {
      output.resize_(self.sizes());
    }
    struct CachedGraph : public MPSCachedGraph
    {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    };
    MPSGraphCache* cache_ = MPSGraphCache::getInstance();
    @autoreleasepool {
      string key = string("log1p_out_mps") + getTensorsStringKey({self});
      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

      if(!cachedGraph) {
        MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph* () {
          CachedGraph *newCachedGraph = nil;
          @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);
            newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
              MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0
                                                          shape:getMPSShape(self)
                                                       dataType:mps::getMPSDataType(self.scalar_type())];
              MPSGraphTensor* addedTensor = [mpsGraph additionWithPrimaryTensor:newCachedGraph->inputTensor
                                                         secondaryTensor:oneTensor
                                                                    name:nil];
            newCachedGraph->outputTensor = [mpsGraph logarithmWithTensor:addedTensor
                                                                    name:nil];
          }
          return newCachedGraph;
        });
        cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
      }

      Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor, self);
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, output);
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
        selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
      };
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
      };
      runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
    }
}

} // namespace native
} // namespace at
