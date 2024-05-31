//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat_native.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/topk_native.h>
#endif

namespace at::native {
namespace mps {

// Produces a shape with the `dim` dimension set to 0.
static std::vector<int64_t> getTopK0Shape(IntArrayRef sizes, const int64_t dim_) {
  const int sz = sizes.size();
  if (sz == 0) {
    return {0};
  }
  const int64_t dim = maybe_wrap_dim(dim_, sz);
  std::vector<int64_t> numbers(sz);

  for (int i = 0; i < sz; i++) {
    const int64_t sz_i = i != dim ? sizes[i] : 0;
    numbers[i] = sz_i;
  }
  return numbers;
}

static void check_shape_except_dim(const Tensor& first, const Tensor& second, int dimension, int index) {
  int first_dims = first.dim();
  int second_dims = second.dim();
  TORCH_CHECK(
      first_dims == second_dims, "Tensors must have same number of dimensions: got ", first_dims, " and ", second_dims);
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.size(dim);
    int64_t second_dim_size = second.size(dim);
    TORCH_CHECK(first_dim_size == second_dim_size,
                "Sizes of tensors must match except in dimension ",
                dim,
                ". Got ",
                static_cast<long long>(first_dim_size),
                " and ",
                static_cast<long long>(second_dim_size),
                " (The offending index is ",
                index,
                ")");
  }
}
} // namespace mps

// topk
TORCH_IMPL_FUNC(topk_out_mps)
(const Tensor& self, int64_t k, int64_t dim_, bool largest, bool sorted, const Tensor& values, const Tensor& indices) {
  using namespace mps;
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1), "selected index k out of range");

  if (!is_macos_13_or_newer() && (k > 16)) {
    TORCH_WARN_ONCE("torch.topk support for k>16 by MPS on MacOS 13+, please upgrade");
    Tensor cpu_indices = indices.clone().to("cpu");
    Tensor cpu_values = values.clone().to("cpu");
    at::topk_out(cpu_values, cpu_indices, self.to(at::Device(kCPU)), k, dim_, largest, sorted);
    values.copy_(cpu_values);
    indices.copy_(cpu_indices);
    return;
  }

  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  // Handle empty tensors
  if (self.numel() == 0) {
    values.copy_(self);
    indices.copy_(values.toType(at::ScalarType::Long));
    return;
  }
  // Handle k == 0 case. Needed because MPSGraph does not support k == 0.
  if (k == 0) {
    const auto out_shape = getTopK0Shape(self.sizes(), dim);
    values.resize_(out_shape);
    indices.copy_(values.toType(at::ScalarType::Long));
    return;
  }

  MPSStream* stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor = nil, *valuesTensor = nil, *indicesTensor = nil;
  };

  // MPSGraph topK is always sorted.
  @autoreleasepool {
    // Input as placeholders
    MPSShape* input_shape = getMPSShape(self);
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    string key = string("topk:") + [ns_shape_key UTF8String] + ":" + getMPSTypeString(self) + ":k" + to_string(k) +
        ":dim" + to_string(dim_) + ":largest" + to_string(largest);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), input_shape);

      if (is_macos_13_or_newer()) {
        MPSGraphTensor* castInputTensor = newCachedGraph->selfTensor;
        MPSDataType dataType = getMPSDataType(self);
        // #issue 104398441 sortWithTensor and argsortWithTensor
        if (dataType != MPSDataTypeInt32 && dataType != MPSDataTypeFloat32 && dataType != MPSDataTypeFloat16) {
          dataType = (dataType & MPSDataTypeFloatBit) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
          castInputTensor = [mpsGraph castTensor:newCachedGraph->selfTensor toType:dataType name:@"castInputTensor"];
        }
        MPSGraphTensor* sortedTensor = [mpsGraph sortWithTensor:castInputTensor
                                                           axis:(NSUInteger)dim
                                                     descending:largest
                                                           name:nil];
        sortedTensor = [mpsGraph sliceTensor:sortedTensor
                                   dimension:(NSUInteger)dim
                                       start:((NSUInteger)0)length:k
                                        name:nil];
        MPSGraphTensor* argSortedTensor = [mpsGraph argSortWithTensor:castInputTensor
                                                                 axis:(NSInteger)dim
                                                           descending:largest
                                                                 name:@"argmax_out"];
        argSortedTensor = [mpsGraph sliceTensor:argSortedTensor dimension:dim start:((NSUInteger)0)length:k name:nil];
        newCachedGraph->valuesTensor = sortedTensor;
        newCachedGraph->indicesTensor = argSortedTensor;

      } else {
        if ((dim_ != -1 && dim_ != self.dim() - 1) && (!largest)) {
          // transpose and negate
          MPSGraphTensor* transposedInput = [mpsGraph transposeTensor:newCachedGraph->selfTensor
                                                            dimension:(NSUInteger)self.dim() - 1
                                                        withDimension:(NSUInteger)dim_
                                                                 name:nil];
          MPSGraphTensor* identity = [mpsGraph identityWithTensor:transposedInput name:nil];
          MPSGraphTensor* negatedTransposedInput = [mpsGraph negativeWithTensor:identity name:nil];
          NSArray<MPSGraphTensor*>* outputMPSGraphTensors = [mpsGraph topKWithSourceTensor:negatedTransposedInput
                                                                                         k:((NSUInteger)k)name:nil];
          MPSGraphTensor* valuesNegatedTransposed = outputMPSGraphTensors[0];
          MPSGraphTensor* indicesTransposed = outputMPSGraphTensors[1];
          MPSGraphTensor* valuesNegated = [mpsGraph transposeTensor:valuesNegatedTransposed
                                                          dimension:(NSUInteger)self.dim() - 1
                                                      withDimension:(NSUInteger)dim_
                                                               name:nil];
          newCachedGraph->valuesTensor = [mpsGraph negativeWithTensor:valuesNegated name:nil];
          newCachedGraph->indicesTensor = [mpsGraph transposeTensor:indicesTransposed
                                                          dimension:(NSUInteger)self.dim() - 1
                                                      withDimension:(NSUInteger)dim_
                                                               name:nil];
        } else if (dim_ != -1 && dim_ != self.dim() - 1) {
          MPSGraphTensor* transposedInput = [mpsGraph transposeTensor:newCachedGraph->selfTensor
                                                            dimension:(NSUInteger)self.dim() - 1
                                                        withDimension:(NSUInteger)dim_
                                                                 name:nil];
          MPSGraphTensor* identity = [mpsGraph identityWithTensor:transposedInput name:nil];
          NSArray<MPSGraphTensor*>* outputMPSGraphTensors = [mpsGraph topKWithSourceTensor:identity
                                                                                         k:((NSUInteger)k)name:nil];
          MPSGraphTensor* valuesTransposed = outputMPSGraphTensors[0];
          MPSGraphTensor* indicesTransposed = outputMPSGraphTensors[1];
          newCachedGraph->valuesTensor = [mpsGraph transposeTensor:valuesTransposed
                                                         dimension:(NSUInteger)self.dim() - 1
                                                     withDimension:(NSUInteger)dim_
                                                              name:nil];
          newCachedGraph->indicesTensor = [mpsGraph transposeTensor:indicesTransposed
                                                          dimension:(NSUInteger)self.dim() - 1
                                                      withDimension:(NSUInteger)dim_
                                                               name:nil];
        } else if (!largest) {
          // only negate
          MPSGraphTensor* negatedInput = [mpsGraph negativeWithTensor:newCachedGraph->selfTensor name:nil];
          NSArray<MPSGraphTensor*>* outputMPSGraphTensors = [mpsGraph topKWithSourceTensor:negatedInput
                                                                                         k:((NSUInteger)k)name:nil];
          MPSGraphTensor* valuesNegated = outputMPSGraphTensors[0];
          newCachedGraph->valuesTensor = [mpsGraph negativeWithTensor:valuesNegated name:nil];
          newCachedGraph->indicesTensor = outputMPSGraphTensors[1];
        } else {
          NSArray<MPSGraphTensor*>* outputMPSGraphTensors = [mpsGraph topKWithSourceTensor:newCachedGraph->selfTensor
                                                                                         k:((NSUInteger)k)name:nil];
          newCachedGraph->valuesTensor = outputMPSGraphTensors[0];
          newCachedGraph->indicesTensor = outputMPSGraphTensors[1];
        }
      }
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->selfTensor, self);
    // Outputs as placeholders
    Placeholder valuesPlaceholder = Placeholder(cachedGraph->valuesTensor, values);
    Placeholder indicesPlaceholder = Placeholder(cachedGraph->indicesTensor, indices);
    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    auto results = dictionaryFromPlaceholders(valuesPlaceholder, indicesPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
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
                "of the output memory locations. Found overlap in input tensor ",
                idx);
    idx++;
  }
  // Check for type promotion
  TORCH_CHECK(canCast(out_dtype, out.scalar_type()),
              "torch.cat(): input types can't be cast to the desired output type ",
              out.scalar_type());
  TORCH_CHECK(inputs.size() > 0, "torch.cat(): invalid number of inputs ", inputs.size());

  dimension = legacy_cat_wrap_dim(dimension, materialized_inputs);
  TORCH_CHECK(dimension >= 0, "torch.cat(): invalid dimension ", dimension);

  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible to cat empty tensors unless all the other tensors were
  // 1-dimensional, so we allowed these tensors to be "skipped".  We maintain
  // this behavior for backwards compatibility, but only for this specific size
  // (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  auto should_skip = [](const Tensor& t) { return t.dim() == 1 && t.size(0) == 0; };
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
                t.device(),
                " and ",
                notSkippedTensor.device());
  }
  TORCH_CHECK(out.device() == notSkippedTensor.device(),
              "torch.cat(): all input tensors and out must be on the same device, but inputs are on ",
              notSkippedTensor.device(),
              " and out is on ",
              out.device());

  // TODO: For better performance by eliminating input tensor gathering and post transpose,
  // TODO: it is better to keep the out tensor's memory format.
  // TODO: dimension needs to be recomputed as:
  // TODO: dim = 0 --> dim = 0; dim = 1 or 2 --> dim = out.dim()- dim; otherwise dim = dim-1
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
      cat_dim_size += tensor.size(dimension);
      idx++;
    }
  }
  // Compute the size of the result
  size[dimension] = cat_dim_size;
  // skip resizing if size of result is same as expected
  if (out.sizes() != size) {
    out.resize_(size, MemoryFormat::Contiguous);
  }
  if (out.numel() == 0) {
    return;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    std::vector<MPSGraphTensor*> inputTensors_;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    string key =
        "cat_out_mps:" + to_string(dimension) + ":" + (memory_format == MemoryFormat::ChannelsLast ? "NHWC" : "NCHW");
    if (!all_same_dtype) {
      key += getTensorsStringKey(input_tensors, true, all_same_sizes_and_stride);
    } else {
      key += ":" + getMPSTypeString(input_tensors[0].scalar_type(), true) + ":" + to_string(inputs.size());
    }
    for (auto idx : skipped_tensor_indices) {
      key += "," + std::to_string(idx);
    }

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto len_tensor_array = inputs.size() - skipped_tensor_indices.size();
      std::vector<MPSGraphTensor*> castInputTensors(len_tensor_array);
      newCachedGraph->inputTensors_.reserve(len_tensor_array);

      for (const auto idx : c10::irange(len_tensor_array)) {
        const Tensor& tensor = input_tensors[idx];
        auto scalar_type = getMPSScalarType(tensor.scalar_type());
        if (tensor.scalar_type() == kBool) {
          scalar_type = MPSDataTypeInt8;
        }
        newCachedGraph->inputTensors_[idx] = mpsGraphUnrankedPlaceHolder(mpsGraph, scalar_type);
        if (tensor.scalar_type() != out_dtype) {
          castInputTensors[idx] = [mpsGraph castTensor:newCachedGraph->inputTensors_[idx]
                                                toType:getMPSDataType(out_dtype)
                                                  name:@"castInput"];
        } else {
          castInputTensors[idx] = newCachedGraph->inputTensors_[idx];
        }
      }

      auto inputTensorsArray = [NSArray arrayWithObjects:castInputTensors.data() count:len_tensor_array];
      MPSGraphTensor* outputTensor = [mpsGraph concatTensors:inputTensorsArray
                                                   dimension:dimension // Maybe convert this from int64_t -> int32
                                                        name:nil];
      if (getMPSDataType(out_dtype) == MPSDataTypeBool) {
        outputTensor = [mpsGraph castTensor:outputTensor toType:MPSDataTypeBool name:@"outputTensor"];
      }
      newCachedGraph->outputTensor_ = outputTensor;
    });

    std::vector<Placeholder> inputPlaceholders;
    int i = 0;
    int t_idx = 0;
    for (const Tensor& tensor : materialized_inputs) {
      if (std::find(skipped_tensor_indices.begin(), skipped_tensor_indices.end(), i) == skipped_tensor_indices.end()) {
        auto scalar_type = getMPSScalarType(tensor.scalar_type());
        if (tensor.scalar_type() == kBool) {
          scalar_type = MPSDataTypeInt8;
        }
        inputPlaceholders.emplace_back(cachedGraph->inputTensors_[t_idx], tensor, nullptr, true, scalar_type);
        t_idx++;
      }
      i++;
    }

    auto outputDataType = getMPSScalarType(out.scalar_type());
    if (!is_macos_13_or_newer() && out.scalar_type() == kBool) {
      outputDataType = MPSDataTypeInt8;
    }
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, out, /*mpsShape=*/nil, /*gatherTensorData=*/false, outputDataType);

    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    for (auto& inputPlaceholder : inputPlaceholders) {
      feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    }
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

} // namespace at::native
