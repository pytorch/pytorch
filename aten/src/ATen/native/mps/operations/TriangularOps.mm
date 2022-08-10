//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(triu_mps_out)
(const Tensor& self,
 int64_t k,
 const Tensor &output) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "triu_mps_out" + mps::getTensorsStringKey({self}) + ":" + std::to_string(k);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor = nil;

          MPSGraphTensor* minusOneTensor = [mpsGraph constantWithScalar:-1
                                                               dataType:MPSDataTypeInt32];

          if(k > 0) {
            MPSGraphTensor* diagMinusOneTensor = [mpsGraph constantWithScalar:(k-1)
                                                                     dataType:MPSDataTypeInt32];
            MPSGraphTensor* complementTensor = [mpsGraph bandPartWithTensor:inputTensor
                                                             numLowerTensor:minusOneTensor
                                                             numUpperTensor:diagMinusOneTensor
                                                                       name:nil];
            outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                  secondaryTensor:complementTensor
                                                             name:nil];
          }
          else {
            MPSGraphTensor* minusDiagTensor = [mpsGraph constantWithScalar:(-k)
                                                                  dataType:MPSDataTypeInt32];
            outputTensor = [mpsGraph bandPartWithTensor:inputTensor
                                         numLowerTensor:minusDiagTensor
                                         numUpperTensor:minusOneTensor
                                                   name:nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

}

TORCH_IMPL_FUNC(tril_mps_out)
(const Tensor& self,
 int64_t k,
 const Tensor &output) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "tril_mps_out" + mps::getTensorsStringKey({self}) + ":" + std::to_string(k);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor = nil;

          MPSGraphTensor* minusOneTensor = [mpsGraph constantWithScalar:-1
                                                               dataType:MPSDataTypeInt32];

          if(k >= 0) {
            MPSGraphTensor* diagTensor = [mpsGraph constantWithScalar:k
                                                             dataType:MPSDataTypeInt32];
            outputTensor = [mpsGraph bandPartWithTensor:inputTensor
                                         numLowerTensor:minusOneTensor
                                         numUpperTensor:diagTensor
                                                   name:nil];
          }
          else {
            MPSGraphTensor* negDiagMinusOneTensor = [mpsGraph constantWithScalar:(-k-1)
                                                                        dataType:MPSDataTypeInt32];
            MPSGraphTensor* complementTensor = [mpsGraph bandPartWithTensor:inputTensor
                                                             numLowerTensor:negDiagMinusOneTensor
                                                             numUpperTensor:minusOneTensor
                                                                       name:nil];
            outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                  secondaryTensor:complementTensor
                                                             name:nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

}

Tensor& diag_mps_out(const Tensor& self,
                     int64_t diagonal,
                     Tensor &output) {

  // Do checks, resize output
  IntArrayRef input_size = self.sizes();
  auto num_input_dims = input_size.size();
  // Input can only be 1D or 2D
  TORCH_CHECK(num_input_dims == 1 || num_input_dims == 2,
    "diag_mps_out: Input tensor must be 1D or 2D")

  if(num_input_dims == 1) {
    auto n = input_size[0];
    if(diagonal > 0)
      n += diagonal;
    else if(diagonal < 0)
      n -= diagonal;

    output.resize_({n, n});
  }
  else if(num_input_dims == 2) {
    auto num_diag_elements = std::min(input_size[0], input_size[1]);
    if(diagonal > 0) {
      TORCH_CHECK(input_size[1] - diagonal > 0, "Matrix not big enough for requested diagonal")
      num_diag_elements = std::min(input_size[0], input_size[1] - diagonal);
    }
    else if(diagonal < 0) {
      TORCH_CHECK(input_size[0] + diagonal > 0, "Matrix not big enough for requested diagonal")
      num_diag_elements = std::min(input_size[0] + diagonal, input_size[1]);
    }

    output.resize_({num_diag_elements});
  }

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {

    MPSShape* input_shape = getMPSShape(self);
    MPSShape* output_shape = getMPSShape(output);
    NSNumber* num_input_cols = nil;
    NSNumber* num_output_cols = nil;
    NSMutableArray<NSNumber*>* flat_input_shape = nil;
    NSMutableArray<NSNumber*>* flat_output_shape = nil;
    if(num_input_dims == 1) {
      num_output_cols = output_shape[1];
      flat_output_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
      flat_output_shape[0] = [NSNumber numberWithInt:[output_shape[0] intValue] * [output_shape[1] intValue]];
    }
    else if(num_input_dims == 2) {
      num_input_cols = input_shape[1];
      flat_input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
      flat_input_shape[0] = [NSNumber numberWithInt:[input_shape[0] intValue] * [input_shape[1] intValue]];
    }
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    string key = "diag_mps_out:" + getMPSTypeString(self.scalar_type()) + ":" + std::to_string(diagonal)
                                 + ":" + string([ns_shape_key UTF8String]);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          // TODO: Accept this as the flat version in 2D case
          MPSGraphTensor* inputTensor = nil;
          if(num_input_dims == 1)
           inputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(self.scalar_type()));
         else
           inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self.scalar_type()), flat_input_shape);

          MPSGraphTensor* outputTensor = nil;

          MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0
                                                           dataType:MPSDataTypeInt32];
          MPSGraphTensor* numDiagElementsRange = nil;
          MPSGraphTensor* diagOffset = nil;
          MPSGraphTensor* rowMultiplier = nil;
          MPSGraphTensor* rowIndices = nil;
          MPSGraphTensor* colIndices = nil;
          MPSGraphTensor* indicesTensor = nil;

          if(num_input_dims == 1) {
            int shape_data[1] = {[input_shape[0] intValue]};
            MPSGraphTensor* inputShapeTensor = [mpsGraph constantWithData:[NSData dataWithBytes:shape_data length:sizeof(int)]
                                                                    shape:@[@1]
                                                                 dataType:MPSDataTypeInt32];
            numDiagElementsRange = [mpsGraph coordinateAlongAxisTensor: zeroTensor
                                                       withShapeTensor: inputShapeTensor
                                                                  name: nil];
            diagOffset = [mpsGraph constantWithScalar:diagonal
                                             dataType:MPSDataTypeInt32];
            rowMultiplier = [mpsGraph constantWithScalar:[num_output_cols intValue]
                                                dataType:MPSDataTypeInt32];
          }
          else {
            int shape_data[1] = {[output_shape[0] intValue]};
            MPSGraphTensor* outputShapeTensor = [mpsGraph constantWithData:[NSData dataWithBytes:shape_data length:sizeof(int)]
                                                                     shape:@[@1]
                                                                  dataType:MPSDataTypeInt32];
            numDiagElementsRange = [mpsGraph coordinateAlongAxisTensor: zeroTensor
                                                       withShapeTensor: outputShapeTensor
                                                                  name: nil];
            diagOffset = [mpsGraph constantWithScalar:diagonal
                                             dataType:MPSDataTypeInt32];
            rowMultiplier = [mpsGraph constantWithScalar:[num_input_cols intValue]
                                                dataType:MPSDataTypeInt32];
          }

          if(diagonal >= 0) {
            rowIndices = numDiagElementsRange;
            colIndices = [mpsGraph additionWithPrimaryTensor:numDiagElementsRange
                                             secondaryTensor:diagOffset
                                                        name:nil];
          }
          else {
            rowIndices = [mpsGraph subtractionWithPrimaryTensor:numDiagElementsRange
                                                secondaryTensor:diagOffset
                                                           name:nil];;
            colIndices = numDiagElementsRange;
          }

          indicesTensor = [mpsGraph multiplicationWithPrimaryTensor:rowIndices
                                                    secondaryTensor:rowMultiplier
                                                               name:nil];
          indicesTensor = [mpsGraph additionWithPrimaryTensor:indicesTensor
                                              secondaryTensor:colIndices
                                                         name:nil];

          if(num_input_dims == 1) {
            // TODO: Scatter mode doesn't matter, so what should I set it to be?
            outputTensor = [mpsGraph scatterWithUpdatesTensor:inputTensor
                                                indicesTensor:indicesTensor
                                                        shape:flat_output_shape
                                                         axis:0
                                                         mode:MPSGraphScatterModeAdd
                                                         name:nil];
            outputTensor = [mpsGraph reshapeTensor:outputTensor
                                         withShape:output_shape
                                              name:nil];
          }
          else if(num_input_dims == 2) {
            outputTensor = [mpsGraph gatherWithUpdatesTensor:inputTensor
                                               indicesTensor:indicesTensor
                                                        axis:0
                                             batchDimensions:0
                                                        name:nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder();
    if(num_input_dims == 1)
      selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    else
      selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, flat_input_shape);

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;
}

} // namespace native
} // namespace at
