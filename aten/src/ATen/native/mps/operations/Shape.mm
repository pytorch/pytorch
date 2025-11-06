//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Pool.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Shape.h>

#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat_native.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/topk_native.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Shape_metallib.h>
#endif

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

template <typename T>
std::string get_type_str();

template <>
std::string get_type_str<int64_t>() {
  return "int64_t";
}

template <>
std::string get_type_str<int32_t>() {
  return "int32_t";
}

// NOTE: `output` is expected to already have the correct size.
template <typename idx_type_t>
static void cat_out_mps_impl(const ITensorListRef& inputs, int64_t dimension, const Tensor& output) {
  CatSharedParams<idx_type_t> shared_params;

  shared_params.ndim = output.dim();
  shared_params.cat_dim = dimension;

  for (const auto dim : c10::irange(output.dim())) {
    shared_params.output_strides[dim] = safe_downcast<idx_type_t, int64_t>(output.stride(dim));
    shared_params.output_sizes[dim] = safe_downcast<idx_type_t, int64_t>(output.size(dim));
  }

  idx_type_t cat_dim_offset = 0;
  size_t input_idx = 0;
  MPSStream* stream = getCurrentMPSStream();

  // Launch a separate kernels for each input. This will produce some overhead.
  // In order to launch only one kernel to process all inputs, we would have to
  // copy all the input tensor data into a packed buffer, which would not be
  // ideal.
  for (const Tensor& input : inputs) {
    if (input.numel() == 0) {
      continue;
    }

    // Metal can only launch up to MAX_INT threads at one time. If the input has
    // more than that number of elements, launch multiple kernels with different
    // offsets into the data.
    const int64_t max_num_threads = static_cast<int64_t>(std::numeric_limits<int32_t>::max());

    for (int64_t numel_remaining = input.numel(); numel_remaining > 0; numel_remaining -= max_num_threads) {
      auto num_threads = std::min(max_num_threads, numel_remaining);
      CatInputParams<idx_type_t> input_params;

      input_params.cat_dim_offset = safe_downcast<idx_type_t, int64_t>(cat_dim_offset);
      input_params.input_element_offset = safe_downcast<idx_type_t, int64_t>(input.numel() - numel_remaining);

      for (const auto dim : c10::irange(input.dim())) {
        input_params.input_strides[dim] = safe_downcast<idx_type_t, int64_t>(input.stride(dim));
        input_params.input_sizes[dim] = safe_downcast<idx_type_t, int64_t>(input.size(dim));
      }

      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
          auto pipeline_state = lib.getPipelineStateForFunc(fmt::format("cat_{}_{}_{}",
                                                                        get_type_str<idx_type_t>(),
                                                                        scalarToMetalTypeString(input),
                                                                        scalarToMetalTypeString(output)));
          getMPSProfiler().beginProfileKernel(pipeline_state, "cat", {input});
          [computeEncoder setComputePipelineState:pipeline_state];
          mtl_setArgs(computeEncoder, input, output, shared_params, input_params);
          mtl_dispatch1DJob(computeEncoder, pipeline_state, num_threads);
          getMPSProfiler().endProfileKernel(pipeline_state);
        }
      });
    }

    cat_dim_offset += input.size(dimension);
    input_idx++;
  }
}
} // namespace mps

// topk
TORCH_IMPL_FUNC(topk_out_mps)
(const Tensor& self, int64_t k, int64_t dim_, bool largest, bool sorted, const Tensor& values, const Tensor& indices) {
  using namespace mps;
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1), "selected index k out of range");

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

  // issue #154890, raising error to prevent crash within MPSGraph until
  // workaround is implemented.
  TORCH_CHECK(self.dim() - dim <= 4, "On-going issue on MPSGraph topk when ndims() - axis > 4, see issue #154890");

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
    std::string key = std::string("topk:") + [ns_shape_key UTF8String] + ":" + getMPSTypeString(self) + ":k" +
        std::to_string(k) + ":dim" + std::to_string(dim_) + ":largest" + std::to_string(largest);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), input_shape);

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
  TORCH_CHECK(!inputs.empty(), "torch.cat(): invalid number of inputs ", inputs.size());

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

  std::vector<int64_t> size(notSkippedTensor.sizes().vec());

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  idx = 0;
  bool has_large_tensor = false;
  for (const Tensor& tensor : materialized_inputs) {
    if (isTooLargeForMPSGraph(tensor)) {
      has_large_tensor |= true;
    }
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

  has_large_tensor |= isTooLargeForMPSGraph(out);

  if (has_large_tensor) {
    return mps::cat_out_mps_impl<int64_t>(materialized_inputs, dimension, out);
  } else {
    return mps::cat_out_mps_impl<int32_t>(materialized_inputs, dimension, out);
  }
}

} // namespace at::native
