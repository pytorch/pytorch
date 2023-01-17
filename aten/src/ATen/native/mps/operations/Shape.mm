//  Copyright © 2022 Apple Inc.

#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/mps/OperationUtils.h>

namespace at::native {

// topk
TORCH_IMPL_FUNC(topk_out_mps)
  (const Tensor& self,
  int64_t k,
  int64_t dim_,
  bool largest,
  bool sorted,
  const Tensor& values,
  const Tensor& indices)
{
  using namespace mps;
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
    k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
    "selected index k out of range");

  TORCH_CHECK( k <= 16 , "Currently topk on mps works only for k<=16 ");

  if (self.dim() == 0 && self.numel() == 1)
  {
      values.copy_(self);
      indices.zero_();
      return;
  }
  MPSStream* stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph
  {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor *selfTensor = nil, *valuesTensor = nil, *indicesTensor = nil;
  };
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  // MPSGraph topK is always sorted.
  @autoreleasepool
  {
      // Input as placeholders
      MPSShape* input_shape = getMPSShape(self);
      NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
      string key = string("topk:") + [ns_shape_key UTF8String] + ":" +
                             getMPSTypeString(self.scalar_type()) +
                             ":k" + to_string(k) + ":dim" + to_string(dim_) +
                             ":largest" + to_string(largest);
      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
      if(!cachedGraph)
      {
          cachedGraph = static_cast<CachedGraph*>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
          CachedGraph *newCachedGraph = nil;
          @autoreleasepool
          {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);
              newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self.scalar_type()), input_shape);
              if ((dim_ != -1 && dim_ != self.dim() - 1) && (!largest))
              {
                // transpose and negate
                  MPSGraphTensor *transposedInput = [mpsGraph transposeTensor: newCachedGraph->selfTensor
                                                                               dimension: (NSUInteger)self.dim()-1
                                                                               withDimension: (NSUInteger)dim_
                                                                               name: nil];
                  MPSGraphTensor * identity = [mpsGraph identityWithTensor: transposedInput
                                                                               name: nil];
                  MPSGraphTensor * negatedTransposedInput = [mpsGraph  negativeWithTensor:identity
                                                                                        name: nil];
                  NSArray<MPSGraphTensor *> * outputMPSGraphTensors = [mpsGraph
                                                                       topKWithSourceTensor:negatedTransposedInput
                                                                       k:((NSUInteger) k)
                                                                       name:nil];
                  MPSGraphTensor *valuesNegatedTransposed = outputMPSGraphTensors[0];
                  MPSGraphTensor *indicesTransposed = outputMPSGraphTensors[1];
                  MPSGraphTensor *valuesNegated = [mpsGraph transposeTensor: valuesNegatedTransposed
                                                                                        dimension: (NSUInteger)self.dim()-1
                                                                                    withDimension: (NSUInteger)dim_
                                                                                             name: nil];
                  newCachedGraph->valuesTensor = [mpsGraph negativeWithTensor:valuesNegated
                                                                         name: nil];
                  newCachedGraph->indicesTensor = [mpsGraph transposeTensor: indicesTransposed
                                                                            dimension: (NSUInteger)self.dim()-1
                                                                            withDimension: (NSUInteger)dim_
                                                                            name: nil];
              }
              else if (dim_ != -1 && dim_ != self.dim() - 1)
              {
                  MPSGraphTensor *transposedInput = [mpsGraph transposeTensor: newCachedGraph->selfTensor
                                                                               dimension: (NSUInteger)self.dim()-1
                                                                               withDimension: (NSUInteger)dim_
                                                                               name: nil];
                  MPSGraphTensor * identity = [mpsGraph identityWithTensor: transposedInput
                                                                               name: nil];
                  NSArray<MPSGraphTensor *> * outputMPSGraphTensors = [mpsGraph
                                                                       topKWithSourceTensor:identity
                                                                       k:((NSUInteger) k)
                                                                       name:nil];
                  MPSGraphTensor *valuesTransposed = outputMPSGraphTensors[0];
                  MPSGraphTensor *indicesTransposed = outputMPSGraphTensors[1];
                  newCachedGraph->valuesTensor = [mpsGraph transposeTensor:valuesTransposed
                                                                        dimension: (NSUInteger)self.dim()-1
                                                                        withDimension: (NSUInteger)dim_
                                                                        name: nil];
                  newCachedGraph->indicesTensor = [mpsGraph transposeTensor: indicesTransposed
                                                                            dimension: (NSUInteger)self.dim()-1
                                                                            withDimension: (NSUInteger)dim_
                                                                            name: nil];
              }
              else if (!largest)
              {
                  // only negate
                  MPSGraphTensor *negatedInput = [mpsGraph negativeWithTensor:newCachedGraph->selfTensor
                                                                        name: nil];
                  NSArray<MPSGraphTensor *> * outputMPSGraphTensors = [mpsGraph
                                                                       topKWithSourceTensor:negatedInput
                                                                       k:((NSUInteger) k)
                                                                       name:nil];
                  MPSGraphTensor *valuesNegated = outputMPSGraphTensors[0];
                  newCachedGraph->valuesTensor = [mpsGraph negativeWithTensor:valuesNegated
                                                                            name: nil];
                  newCachedGraph->indicesTensor = outputMPSGraphTensors[1];
              }
              else
              {
                  NSArray<MPSGraphTensor *> * outputMPSGraphTensors = [mpsGraph
                                                                         topKWithSourceTensor:newCachedGraph->selfTensor
                                                                         k:((NSUInteger) k)
                                                                         name:nil];
                  newCachedGraph->valuesTensor = outputMPSGraphTensors[0];
                  newCachedGraph->indicesTensor = outputMPSGraphTensors[1];
              }

          }
          return newCachedGraph;
        }));
      }
  Placeholder inputPlaceholder  = Placeholder(cachedGraph->selfTensor, self);
  // Outputs as placeholders
  Placeholder valuesPlaceholder = Placeholder(cachedGraph->valuesTensor, values);
  Placeholder indicesPlaceholder = Placeholder(cachedGraph->indicesTensor, indices);
  // Create dictionary of inputs and outputs
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =  nil;
  feeds = @{
  inputPlaceholder.getMPSGraphTensor() :
      inputPlaceholder.getMPSGraphTensorData()
  };
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
  valuesPlaceholder.getMPSGraphTensor() :
          valuesPlaceholder.getMPSGraphTensorData(),
  indicesPlaceholder.getMPSGraphTensor() :
        indicesPlaceholder.getMPSGraphTensorData()
  };

  runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

void check_shape_except_dim(const Tensor &first, const Tensor &second,
                            int dimension, int index)
{
  int first_dims = first.dim();
  int second_dims = second.dim();
  TORCH_CHECK(first_dims == second_dims,
      "Tensors must have same number of dimensions: got ", first_dims,
      " and ", second_dims);
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = at::native::size(first, dim);
    int64_t second_dim_size = at::native::size(second, dim);
    TORCH_CHECK(first_dim_size == second_dim_size,
        "Sizes of tensors must match except in dimension ", dim, ". Got ",
        static_cast<long long>(first_dim_size), " and ",
        static_cast<long long>(second_dim_size), " (The offending index is ",
        index, ")");
  }
}

TORCH_IMPL_FUNC(cat_out_mps)
      (const ITensorListRef& inputs,
       int64_t dimension,
       int64_t valid,
       bool all_contiguous,
       bool all_same_dtype,
       bool all_same_sizes_and_stride,
       MemoryFormat memory_format,
       const Tensor& out) {

  using namespace mps;
  if (out.numel() == 0) {
    return;
  }
  auto materialized_inputs = inputs.materialize();
  auto out_dtype = at::native::result_type(inputs);

  int idx = 0;
  for (const Tensor& t : materialized_inputs) {
    TORCH_CHECK(t.dim() > 0, "zero-dimensional tensor (at position ", idx, ") cannot be concatenated");
    auto lap = at::get_overlap_status(out, t);
    TORCH_CHECK(lap != at::MemOverlapStatus::Partial && lap != at::MemOverlapStatus::Full,
        "torch.cat(): unsupported operation: the input tensors cannot refer to any "
        "of the output memory locations. Found overlap in input tensor ", idx);
    idx++;
  }
  // Check for type promotion
  TORCH_CHECK(canCast(out_dtype, out.scalar_type()),
              "torch.cat(): input types can't be cast to the desired output type ", out.scalar_type());
  TORCH_CHECK(inputs.size() > 0,"torch.cat(): invalid number of inputs ", inputs.size());

  dimension = legacy_cat_wrap_dim(dimension, materialized_inputs);
  TORCH_CHECK(dimension >= 0, "torch.cat(): invalid dimension ", dimension);

  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible to cat empty tensors unless all the other tensors were
  // 1-dimensional, so we allowed these tensors to be "skipped".  We maintain
  // this behavior for backwards compatibility, but only for this specific size
  // (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  auto should_skip = [](const Tensor& t) {
    return t.dim() == 1 && at::native::size(t, 0) == 0;
  };
  at::assert_no_internal_overlap(out);

  Tensor notSkippedTensor;
  // Indices of tensors to be skipped because they're empty
  std::vector<int64_t> skipped_tensor_indices;
  // Tensors to be read
  std::vector<Tensor> input_tensors;
  int tensor_idx = 0;
  for (const Tensor& t : materialized_inputs) {
    if (t.numel() == 0 || should_skip(t)) {
      skipped_tensor_indices.push_back(tensor_idx);
      tensor_idx++;
      continue;
    }
    input_tensors.push_back(t);
    // TODO: Is this OK?
    notSkippedTensor = t;
    tensor_idx++;
  }
  // If all inputs are empty tensors, return an empty tensor
  if (!notSkippedTensor.defined()) {
    return;
  }
  for (const Tensor& t : inputs) {
    TORCH_CHECK(t.device() == notSkippedTensor.device(),
                "torch.cat(): all input tensors must be on the same device. Received ",
                t.device(), " and ", notSkippedTensor.device());
  }
  TORCH_CHECK(out.device() == notSkippedTensor.device(),
              "torch.cat(): all input tensors and out must be on the same device, but inputs are on ",
              notSkippedTensor.device(), " and out is on ", out.device());

  if (out.suggest_memory_format() == MemoryFormat::ChannelsLast) {
    out.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::Contiguous);
  }
  std::vector<int64_t> size(notSkippedTensor.sizes().vec());

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  idx = 0;
  for (const Tensor& tensor : materialized_inputs) {
    if (!should_skip(tensor)) {
      // TODO: Factor out `check_shape_except_dim`
      check_shape_except_dim(notSkippedTensor, tensor, dimension, idx);
      cat_dim_size += at::native::size(tensor, dimension);
      idx++;
    }
  }
  // Compute the size of the result
  size[dimension] = cat_dim_size;
  // skip resizing if size of result is same as expected
  if (out.sizes() != size) {
    out.resize_(size, memory_format);
  }
  if (out.numel() == 0) {
    return;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    std::vector<MPSGraphTensor*> inputTensors_;
    MPSGraphTensor* outputTensor_ = nil;
  };
  MPSGraphCache *cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "cat_out_mps:" + to_string(dimension) + getTensorsStringKey(input_tensors, /*short_dtype*/true) + ":" +
                 (memory_format == MemoryFormat::ChannelsLast ? "NHWC" : "NCHW");

    CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);
    if (!cachedGraph) {
      cachedGraph = cache_->CreateCachedGraphAs<CachedGraph>(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph *mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          auto len_tensor_array = inputs.size() - skipped_tensor_indices.size();
          std::vector<MPSGraphTensor*> castInputTensors(len_tensor_array);
          newCachedGraph->inputTensors_.reserve(len_tensor_array);

          for (const auto idx : c10::irange(len_tensor_array)) {
            const Tensor& tensor = input_tensors[idx];
            auto scalar_type = getMPSScalarType(tensor.scalar_type());
            if (tensor.scalar_type() == kBool) {
              scalar_type = MPSDataTypeInt8;
            }
            newCachedGraph->inputTensors_[idx] = mpsGraphRankedPlaceHolder(mpsGraph, scalar_type, getMPSShape(tensor, memory_format));
            if (tensor.scalar_type() != out_dtype) {
              castInputTensors[idx] = [mpsGraph castTensor:newCachedGraph->inputTensors_[idx]
                                                    toType:getMPSDataType(out_dtype)
                                                      name:@"castInput"];
            } else {
              castInputTensors[idx] = newCachedGraph->inputTensors_[idx];
            }
          }

          auto inputTensorsArray = [NSArray arrayWithObjects:castInputTensors.data()
                                                       count:len_tensor_array];
          MPSGraphTensor* outputTensor = [mpsGraph concatTensors:inputTensorsArray
                                                       dimension:dimension // Maybe convert this from int64_t -> int32
                                                            name:nil];
          if (getMPSDataType(out_dtype) == MPSDataTypeBool) {
            outputTensor = [mpsGraph castTensor:outputTensor
                                         toType:MPSDataTypeBool
                                           name:@"outputTensor"];
          }
          newCachedGraph->outputTensor_ = memory_format == MemoryFormat::ChannelsLast ?
                                         convertNHWCtoNCHW(mpsGraph, outputTensor) : outputTensor;
        }
        return newCachedGraph;
      });
    }

    std::vector<Placeholder> inputPlaceholders;
    int i = 0;
    int t_idx = 0;
    for (const Tensor& tensor : materialized_inputs) {
      if (std::find(skipped_tensor_indices.begin(), skipped_tensor_indices.end(), i) == skipped_tensor_indices.end()) {
        auto scalar_type = getMPSScalarType(tensor.scalar_type());
        if (tensor.scalar_type() == kBool) {
          scalar_type = MPSDataTypeInt8;
        }
        inputPlaceholders.emplace_back(cachedGraph->inputTensors_[t_idx], tensor,
                                       getMPSShape(tensor, memory_format),
                                       memory_format != MemoryFormat::ChannelsLast, scalar_type);
        t_idx++;
      }
      i++;
    }

    auto outputDataType = getMPSScalarType(out.scalar_type());
    if (!is_macos_13_or_newer() && out.scalar_type() == kBool) {
      outputDataType = MPSDataTypeInt8;
    }
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out, nil, false, outputDataType);

    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    for (auto& inputPlaceholder : inputPlaceholders) {
      feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    }
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
  }
}

} // namespace at::native
