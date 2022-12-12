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

TORCH_IMPL_FUNC(gather_out_mps)
(const Tensor & self_arg,
 int64_t dim,
 const Tensor & index,
 bool sparse_grad,
 const Tensor & output) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  auto self = self_arg.dim() == 0 ? self_arg.view({1}) : self_arg;

  dim = at::maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(!sparse_grad, "sparse_grad not supported in MPS yet")

  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(self.scalar_type() == output.scalar_type(),
              "gather(): self and output must have the same scalar type");
  TORCH_CHECK(dim >= 0 && dim < self.dim(),
              "gather(): Indexing dim ", dim, " is out of bounds of tensor");


  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {

    MPSShape* input_shape = getMPSShape(self);
    NSString* ns_input_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    MPSShape* index_shape = getMPSShape(index);
    NSString* ns_index_shape_key = [[index_shape valueForKey:@"description"] componentsJoinedByString:@","];

    int num_input_dims = [input_shape count];
    int num_index_dims = [index_shape count];

    TORCH_CHECK(num_input_dims == num_index_dims, "Input and index must have same rank")

    // Determine if we need to slice into the input tensor
    bool needSlice = false;

    for(int i = 0; i < num_input_dims; i++) {
      TORCH_CHECK(i == dim || [index_shape[i] intValue] <= [input_shape[i] intValue], "Index dim must not exceed input dim except at gathering axis")
      if(i != dim && [index_shape[i] intValue] < [input_shape[i] intValue])
        needSlice = true;
    }

    string key = "gather_out_mps:" + getMPSTypeString(self.scalar_type()) + ":"
                                   + getMPSTypeString(index.scalar_type()) + ":"
                                   + std::to_string(dim) + ":"
                                   + [ns_input_shape_key UTF8String] + ":"
                                   + [ns_index_shape_key UTF8String];
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self.scalar_type()), input_shape);
          MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(index.scalar_type()), index_shape);

          MPSGraphTensor* getInput = nil;

          // Slice into the input tensor IF NEEDED
          if(needSlice) {
            NSMutableArray<NSNumber*> *starts = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
            NSMutableArray<NSNumber*> *ends = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
            NSMutableArray<NSNumber*> *strides = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

            for(int i = 0; i < num_input_dims; i++) {
              // All strides are 1
              strides[i] = @1;
              // All starts are 0
              starts[i] = @0;
              if(i != dim)
                ends[i] = index_shape[i];
              else
                ends[i] = input_shape[i];
            }

            getInput = [mpsGraph sliceTensor:inputTensor
                                         starts:starts
                                           ends:ends
                                        strides:strides
                                           name:nil];

          }
          else
            getInput = inputTensor;

          MPSGraphTensor* castIndexTensor = [mpsGraph castTensor:indexTensor
                                                          toType:getMPSDataType(ScalarType::Int)
                                                            name:(NSString * _Nonnull)nil];

          MPSGraphTensor* outputTensor = [mpsGraph gatherAlongAxis: (NSInteger) dim
                                                 withUpdatesTensor: getInput
                                                     indicesTensor: castIndexTensor
                                                              name: nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->indexTensor_ = indexTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, input_shape);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index, index_shape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      indexPlaceholder.getMPSGraphTensor() : indexPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

}

void scatter_mps_general
(const Tensor& self_arg,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const Tensor& output,
 string func_name,
 const c10::string_view reduce) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  auto self = self_arg.dim() == 0 ? self_arg.view({1}) : self_arg;

  dim = at::maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(self.scalar_type() == output.scalar_type() && output.scalar_type() == src.scalar_type(),
              "scatter(): self, src and output must have the same scalar type");
  TORCH_CHECK(dim >= 0 && dim < self.dim(),
              "scatter(): Indexing dim ", dim, " is out of bounds of tensor");


  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* srcTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {

    MPSShape* input_shape = getMPSShape(self);
    NSString* ns_input_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    MPSShape* index_shape = getMPSShape(index);
    NSString* ns_index_shape_key = [[index_shape valueForKey:@"description"] componentsJoinedByString:@","];
    MPSShape* src_shape = getMPSShape(src);
    NSString* ns_src_shape_key = [[src_shape valueForKey:@"description"] componentsJoinedByString:@","];

    int num_input_dims = [input_shape count];
    int num_index_dims = [index_shape count];
    int num_src_dims = [src_shape count];

    TORCH_CHECK(num_input_dims == num_index_dims && num_index_dims == num_src_dims, "Input, index and src must have same rank")

    // Do we need to slice into the src tensor?
    bool needSlice = false;
    bool inputNeedSlice = false;

    for(int i = 0; i < num_input_dims; i++) {
      TORCH_CHECK(i == dim || [index_shape[i] intValue] <= [input_shape[i] intValue], "Index dim must not exceed input dim except at gathering axis")
      TORCH_CHECK([index_shape[i] intValue] <= [src_shape[i] intValue], "Index dim must not exceed input dim except at gathering axis")
      if([index_shape[i] intValue] < [src_shape[i] intValue])
        needSlice = true;
      if(i != dim && [index_shape[i] intValue] < [input_shape[i] intValue])
        inputNeedSlice = true;
    }

    TORCH_CHECK(reduce != "mean", "Scatter reduce mean mode not yet supported in MPS")

    string reduce_key;

    if(reduce == "set")
      reduce_key = "set";
    else if(reduce == "sum")
      reduce_key = "sum";
    else if(reduce == "add")
      reduce_key = "add";
    else if(reduce == "prod")
      reduce_key = "prod";
    else if(reduce == "multiply")
      reduce_key = "multiply";
    else if(reduce == "amax")
      reduce_key = "amax";
    else if(reduce == "amin")
      reduce_key = "amin";

    string key = func_name + ":" + getMPSTypeString(self.scalar_type()) + ":"
                                 + getMPSTypeString(index.scalar_type()) + ":"
                                 + std::to_string(dim) + ":"
                                 + [ns_input_shape_key UTF8String] + ":"
                                 + [ns_index_shape_key UTF8String] + ":"
                                 + [ns_src_shape_key UTF8String] + ":"
                                 + reduce_key;
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self.scalar_type()), input_shape);
          MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(index.scalar_type()), index_shape);
          MPSGraphTensor* srcTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(src.scalar_type()), src_shape);

          MPSGraphTensor* getSrc = nil;
          MPSGraphTensor* getInput = nil;

          // Slice into the src tensor IF NEEDED
          if(needSlice) {
            NSMutableArray<NSNumber*> *starts = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
            NSMutableArray<NSNumber*> *ends = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
            NSMutableArray<NSNumber*> *strides = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

            for(int i = 0; i < num_input_dims; i++) {
              // All strides are 1
              strides[i] = @1;
              // All starts are 0
              starts[i] = @0;
              ends[i] = index_shape[i];
            }

            getSrc = [mpsGraph sliceTensor:srcTensor
                                    starts:starts
                                      ends:ends
                                   strides:strides
                                      name:nil];

          }
          else
            getSrc = srcTensor;

          // Use in case input needs to be smaller to get scatter
          NSArray<NSNumber*>* scatterInputShape = nil;

          // Slice into the input tensor IF NEEDED
          if(inputNeedSlice) {
            NSMutableArray<NSNumber*> *starts = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
            NSMutableArray<NSNumber*> *ends = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
            NSMutableArray<NSNumber*> *strides = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

            auto rc = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

            for(int i = 0; i < num_input_dims; i++) {
              // All strides are 1
              strides[i] = @1;
              // All starts are 0
              starts[i] = @0;
              if(i != dim) {
                ends[i] = index_shape[i];
                rc[i] = index_shape[i];
              }
              else {
                ends[i] = input_shape[i];
                rc[i] = input_shape[i];
              }
            }
            scatterInputShape = rc;

            getInput = [mpsGraph sliceTensor:inputTensor
                                      starts:starts
                                        ends:ends
                                     strides:strides
                                        name:nil];

          }
          else {
            getInput = inputTensor;
            scatterInputShape = input_shape;
          }

          MPSGraphTensor* outputTensor = nil;

          MPSGraphTensor* castIndexTensor = [mpsGraph castTensor:indexTensor
                                                          toType:getMPSDataType(ScalarType::Int)
                                                            name:(NSString * _Nonnull)nil];

          MPSGraphScatterMode scatter_mode;

          if(reduce_key == "set")
            scatter_mode = MPSGraphScatterModeSet;
          else if(reduce_key == "sum" || reduce_key == "add")
            scatter_mode = MPSGraphScatterModeAdd;
          else if(reduce_key == "prod" || reduce_key == "multiply")
            scatter_mode = MPSGraphScatterModeMul;
          else if(reduce_key == "amax")
            scatter_mode = MPSGraphScatterModeMax;
          else if(reduce_key == "amin")
            scatter_mode = MPSGraphScatterModeMin;

          if(!inputNeedSlice) {
            outputTensor = [mpsGraph scatterAlongAxis: (NSInteger) dim
                                       withDataTensor: getInput
                                        updatesTensor: getSrc
                                        indicesTensor: castIndexTensor
                                                 mode: scatter_mode
                                                 name: nil];
          }
          else {
            // Scatter this into the input with set mode
            MPSGraphTensor* scatterTensor = [mpsGraph scatterAlongAxis: (NSInteger) dim
                                                        withDataTensor: getInput
                                                         updatesTensor: getSrc
                                                         indicesTensor: castIndexTensor
                                                                  mode: scatter_mode
                                                                  name: nil];

            // Make an array of scatter indices tensors
            NSMutableArray<MPSGraphTensor*>* indicesTensors = [NSMutableArray<MPSGraphTensor*> arrayWithCapacity:num_input_dims];

            // 1. Concatenate the coord tensors
            // 2. Flatten the values
            // 3. Scatter into input with add mode

            std::vector<int> shape_data(num_input_dims);

            for(int i = 0; i < num_input_dims; i++) {
              shape_data[i] = {[scatterInputShape[i] intValue]};
            }

            MPSGraphTensor* scatterInputShapeTensor = [mpsGraph constantWithData:[NSData dataWithBytes:shape_data.data() length:num_input_dims * sizeof(int)]
                                                                           shape:@[[NSNumber numberWithInt:num_input_dims]]
                                                                        dataType:MPSDataTypeInt32];

            for(int i = 0; i < num_input_dims; i++) {
              MPSGraphTensor* axisTensor = [mpsGraph constantWithScalar:i
                                                               dataType:MPSDataTypeInt32];
              MPSGraphTensor* scatter_currentIndexTensor = [mpsGraph coordinateAlongAxisTensor: axisTensor
                                                                               withShapeTensor: scatterInputShapeTensor
                                                                                          name: nil];
              scatter_currentIndexTensor = [mpsGraph reshapeTensor:scatter_currentIndexTensor
                                                         withShape:@[@-1, @1]
                                                              name:nil];
              indicesTensors[i] = scatter_currentIndexTensor;
            }

            MPSGraphTensor* scatter_fullIndexTensor = [mpsGraph concatTensors:indicesTensors
                                                                    dimension:(NSInteger)1
                                                                         name:nil];

            MPSGraphTensor* flatValuesTensor = [mpsGraph reshapeTensor:scatterTensor
                                                             withShape:@[@-1]
                                                                  name:nil];

            outputTensor = [mpsGraph scatterNDWithDataTensor:inputTensor
                                               updatesTensor:flatValuesTensor
                                               indicesTensor:scatter_fullIndexTensor
                                             batchDimensions:0
                                                        mode:MPSGraphScatterModeSet
                                                        name:nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->srcTensor_ = srcTensor;
          newCachedGraph->indexTensor_ = indexTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, input_shape);
    Placeholder srcPlaceholder = Placeholder(cachedGraph->srcTensor_, src, src_shape);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index, index_shape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      srcPlaceholder.getMPSGraphTensor() : srcPlaceholder.getMPSGraphTensorData(),
      indexPlaceholder.getMPSGraphTensor() : indexPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

}

TORCH_IMPL_FUNC(scatter_src_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const Tensor& output) {

  scatter_mps_general(self, dim, index, src, output, "scatter_src_out_mps", "set");

}

TORCH_IMPL_FUNC(scatter_value_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const Tensor& output) {

  Tensor src = at::native::empty_mps(index.sizes(),
                                     self.scalar_type(),
                                     c10::nullopt,
                                     kMPS,
                                     c10::nullopt,
                                     self.suggest_memory_format());
  src.fill_(value);
  scatter_mps_general(self, dim, index, const_cast<Tensor&>(src), output, "scatter_value_out_mps", "set");

}

TORCH_IMPL_FUNC(scatter_reduce_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const c10::string_view reduce,
 const Tensor& output) {

  scatter_mps_general(self, dim, index, src, output, "scatter_reduce_out_mps", reduce);

}

TORCH_IMPL_FUNC(scatter_value_reduce_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const c10::string_view reduce,
 const Tensor& output) {

  Tensor src = at::native::empty_mps(index.sizes(),
                                     self.scalar_type(),
                                     c10::nullopt,
                                     kMPS,
                                     c10::nullopt,
                                     self.suggest_memory_format());
  src.fill_(value);
  scatter_mps_general(self, dim, index, const_cast<Tensor&>(src), output, "scatter_value_reduce_out_mps", reduce);

}

TORCH_IMPL_FUNC(scatter_add_mps_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const Tensor& output) {

  scatter_mps_general(self, dim, index, src, output, "scatter_add_mps_out", "add");
}

} // namespace native
} // namespace at
