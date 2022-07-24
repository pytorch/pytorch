//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace mps {

// Pad operations (1D/2D/3D forward and backward)
Tensor& pad_out_template(Tensor &output, const Tensor &input_, IntArrayRef padding,
                         const c10::optional<Tensor>& grad_output_opt,
                         MPSGraphPaddingMode mode, const string op_name)
{
  const int padding_size = (int) padding.size();
  const int padding_dim = padding_size / 2; // either 1D, 2D, or 3D

  TORCH_CHECK(padding_size == 2 || padding_size == 4 || padding_size == 6,
              "invalid padding argument of size ", padding_size);

  const Tensor& grad_output_ = *(at::borrow_from_optional_tensor(grad_output_opt));
  const bool is_backward_pass = grad_output_.defined();

  int dim_w = padding_dim, dim_h = padding_dim - 1, dim_d = padding_dim - 2, dim_slices = 0;
  int64_t nbatch = 1, ndims = input_.ndimension();

  if (!is_backward_pass) {
    bool valid_dims = input_.size(1) != 0 && input_.size(padding_dim) != 0;
    TORCH_CHECK((ndims == 1 + padding_dim && valid_dims) ||
                (ndims == 2 + padding_dim && valid_dims && input_.size(1 + padding_dim) != 0),
                "3D or 4D (batch mode) tensor expected for input, but got: ", input_);
  }

  if (ndims == 2 + padding_dim) {
    nbatch = input_.size(0);
    dim_w++;
    dim_h++;
    dim_d++;
    dim_slices++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding_dim > 1 ? padding[2] : 0;
  int64_t pad_b = padding_dim > 1 ? padding[3] : 0;
  int64_t pad_front = padding_dim > 2 ? padding[4] : 0;
  int64_t pad_back  = padding_dim > 2 ? padding[5] : 0;

  int64_t nplane = input_.size(dim_slices);
  int64_t input_w = input_.size(dim_w);
  int64_t output_w  = input_w + pad_l + pad_r;
  int64_t input_h = padding_dim > 1 ? input_.size(dim_h) : 0;
  int64_t output_h = padding_dim > 1 ? input_h + pad_t + pad_b : 0;
  int64_t input_d = padding_dim > 2 ? input_.size(dim_d) : 0;
  int64_t output_d = padding_dim > 2 ? input_d + pad_front + pad_back : 0;

  Tensor grad_output, input = input_;

  if (!is_backward_pass) {
    TORCH_CHECK(pad_l < input_w && pad_r < input_w,
      "Argument #4: Padding size should be less than the corresponding "
      "input dimension, but got: padding (", pad_l, ", ", pad_r,
      ") at dimension ", dim_w, " of input ", ndims);

    if (padding_dim > 1) {
      TORCH_CHECK(pad_t < input_h && pad_b < input_h,
        "Argument #6: Padding size should be less than the corresponding "
        "input dimension, but got: padding (", pad_t, ", ", pad_b,
        ") at dimension ", dim_h, " of input ", ndims);
    }
    TORCH_CHECK(output_w >= 1 || output_h >= padding_dim - 1,
      "input (H: ", input_h, ", W: ", input_w, ") is too small. Calculated "
      "output H: ", output_h, " W: ", output_w);

    if (ndims == 1 + padding_dim) {
      if (padding_dim == 3)
        output.resize_({nplane, output_d, output_h, output_w});
      else if (padding_dim == 2)
        output.resize_({nplane, output_h, output_w});
      else
        output.resize_({nplane, output_w});
    } else {
      if (padding_dim == 3)
        output.resize_({nbatch, nplane, output_d, output_h, output_w});
      else if (padding_dim == 2)
        output.resize_({nbatch, nplane, output_h, output_w});
      else
        output.resize_({nbatch, nplane, output_w});
    }
    if (output.numel() == 0 || input_.numel() == 0)
      return output;
    input = input_.contiguous();
  } else {
    TORCH_CHECK(output_w == grad_output_.size(dim_w),
      "gradOutput width unexpected. Expected: ", output_w, ", Got: ", grad_output_.size(dim_w));
    if (padding_dim > 1) {
      TORCH_CHECK(output_h == grad_output_.size(dim_h),
        "gradOutput height unexpected. Expected: ", output_h, ", Got: ", grad_output_.size(dim_h));
    }
    grad_output = grad_output_.contiguous();
  }

  const int64_t input_dim = input.dim();
  MPSShape *leftPadding = nullptr, *rightPadding = nullptr;
  if (padding_dim == 3) {
    leftPadding = [NSArray arrayWithObjects:(const NSNumber*[]){ @(0), @(0), @(pad_front), @(pad_t), @(pad_l) } count:input_dim];
    rightPadding = [NSArray arrayWithObjects:(const NSNumber*[]){ @(0), @(0), @(pad_back), @(pad_b), @(pad_r) } count:input_dim];
  } else if (padding_dim == 2) {
    leftPadding = [NSArray arrayWithObjects:(const NSNumber*[]){ @(0), @(0), @(pad_t), @(pad_l) } count:input_dim];
    rightPadding = [NSArray arrayWithObjects:(const NSNumber*[]){ @(0), @(0), @(pad_b), @(pad_r) } count:input_dim];
  } else if (padding_dim == 1) {
    leftPadding = [NSArray arrayWithObjects:(const NSNumber*[]){ @(0), @(0), @(pad_l) } count:input_dim];
    rightPadding = [NSArray arrayWithObjects:(const NSNumber*[]){ @(0), @(0), @(pad_r) } count:input_dim];
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) { }
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor *gradOutputTensor = nil;
  };
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = op_name + getTensorsStringKey({input, grad_output}) +
                           ":L" + to_string(pad_l)     + ":R" + to_string(pad_r) +
                           ":T" + to_string(pad_t)     + ":B" + to_string(pad_b) +
                           ":F" + to_string(pad_front) + ":K" + to_string(pad_back);

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      cachedGraph = static_cast<CachedGraph*>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);
            newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
            if (!is_backward_pass) {
              newCachedGraph->outputTensor = [mpsGraph padTensor:newCachedGraph->inputTensor
                                                 withPaddingMode:mode
                                                     leftPadding:leftPadding
                                                    rightPadding:rightPadding
                                                   constantValue:0
                                                            name:nil];
            } else {
              newCachedGraph->gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
              newCachedGraph->outputTensor = [mpsGraph padGradientWithIncomingGradientTensor:newCachedGraph->gradOutputTensor
                                                                                sourceTensor:newCachedGraph->inputTensor
                                                                                 paddingMode:mode
                                                                                 leftPadding:leftPadding
                                                                                rightPadding:rightPadding
                                                                                        name:nil];
            }
        }
        return newCachedGraph;
      }));
    }
    Placeholder inputPlaceholder  = Placeholder(cachedGraph->inputTensor, input);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, output);

    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    if (is_backward_pass) {
        Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor, grad_output);
        feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();
    }
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
  }
  return output;
}
} // namespace mps

// 1D Reflection and Replication Padding
TORCH_IMPL_FUNC(reflection_pad1d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt, MPSGraphPaddingModeReflect, "reflection_pad1d_out_mps");
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input), input, padding, grad_output, MPSGraphPaddingModeReflect, "reflection_pad1d_backward_out_mps");
}

TORCH_IMPL_FUNC(replication_pad1d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt, MPSGraphPaddingModeClampToEdge, "replication_pad1d_out_mps");
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input), input, padding, grad_output, MPSGraphPaddingModeClampToEdge, "replication_pad1d_backward_out_mps");
}

// 2D Reflection and Replication Padding
Tensor& reflection_pad2d_out_mps(const Tensor& input, IntArrayRef padding, Tensor& output)
{
  return mps::pad_out_template(output, input, padding, c10::nullopt, MPSGraphPaddingModeReflect, __func__);
}

Tensor reflection_pad2d_mps(const Tensor& input, IntArrayRef padding)
{
  Tensor output = at::empty({0}, input.options());
  return mps::pad_out_template(output, input, padding, c10::nullopt, MPSGraphPaddingModeReflect, __func__);
}

Tensor& reflection_pad2d_backward_out_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeReflect, __func__);
}

Tensor reflection_pad2d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding)
{
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeReflect, __func__);
}

TORCH_IMPL_FUNC(replication_pad2d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt, MPSGraphPaddingModeClampToEdge, "replication_pad2d_out_mps");
}

Tensor& replication_pad2d_backward_out_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, __func__);
}

Tensor replication_pad2d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding)
{
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, __func__);
}

// 3D Reflection and Replication Padding
TORCH_IMPL_FUNC(reflection_pad3d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt, MPSGraphPaddingModeReflect, "reflection_pad3d_out_mps");
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input), input, padding, grad_output, MPSGraphPaddingModeReflect, "reflection_pad3d_backward_out_mps");
}

TORCH_IMPL_FUNC(replication_pad3d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt, MPSGraphPaddingModeClampToEdge, "replication_pad3d_out_mps");
}

Tensor& replication_pad3d_backward_out_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, __func__);
}

Tensor replication_pad3d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding)
{
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, __func__);
}

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

inline c10::MemoryFormat compute_output_memory_format(const TensorList &inputs) {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (auto &t : inputs) {
    auto f = t.suggest_memory_format();
    if (!format.has_value()) {
      format = f;
      continue;
    }
    if (format.value() == f) {
      continue;
    }
    bool contiguous = (format.value() == c10::MemoryFormat::Contiguous || f == c10::MemoryFormat::Contiguous || format.value() != f);
    if (contiguous) {
      return c10::MemoryFormat::Contiguous;
    }
  }
  return format.value();
}

//Tensor cat_mps(TensorList inputs, int64_t dimension) {
  //ScalarType high_type = result_type(inputs);
  //Tensor out = at::empty({0}, inputs.front().options().dtype(high_type));
  //at::native::cat_out_mps(inputs, dimension, out);
  //return out;
//}

TORCH_IMPL_FUNC(cat_out_mps)
      (ITensorListRef inputs,
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

  int idx = 0;
  for(const Tensor& t : materialized_inputs) {
    TORCH_CHECK(t.dim() > 0,
             "zero-dimensional tensor (at position ", idx, ") cannot be concatenated");
    idx++;
  }

  dimension = legacy_cat_wrap_dim(dimension, inputs);

  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible to cat empty tensors unless all the other tensors were
  // 1-dimensional, so we allowed these tensors to be "skipped".  We maintain
  // this behavior for backwards compatibility, but only for this specific size
  // (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  auto should_skip = [](const Tensor& t) {
    return t.dim() == 1 && at::native::size(t, 0) == 0;
  };

  const Tensor* notSkippedTensor = NULL; // non-owning reference
  int nDims = 0;

  // Check for type promotion
  TORCH_CHECK(
      canCast(result_type(inputs), out.scalar_type()),
      "torch.cat(): input types ",
      " can't be cast to the desired output type ",
      out.scalar_type());

  // Inputs cannot alias the output tensor
  idx = 0;
  for(const Tensor& t : materialized_inputs) {
    auto lap = at::get_overlap_status(out, t);
    TORCH_CHECK(
        lap != at::MemOverlapStatus::Partial &&
            lap != at::MemOverlapStatus::Full,
        "torch.cat(): unsupported operation: the input tensors cannot refer to any "
        "of the output memory locations. Found overlap in input "
        "tensor ",
        idx);
    idx++;
  }
  at::assert_no_internal_overlap(out);

  // Indices of tensors to be skipped because they're empty
  std::vector<int64_t> skipped_tensor_indices;
  // Tensors to be read
  std::vector<const Tensor*> input_tensors;
  int tensor_idx = 0;
  for(const Tensor& t : materialized_inputs) {
    if(t.numel() == 0 || should_skip(t)) {
      skipped_tensor_indices.push_back(tensor_idx);
      tensor_idx++;
      continue;
    }
    input_tensors.push_back(&t);
    nDims = t.dim();
    // TODO: Is this OK?
    notSkippedTensor = &t;
    tensor_idx++;
  }

  // If all inputs are empty tensors, return an empty tensor
  if (notSkippedTensor == NULL) {
    return;
  }

  TORCH_CHECK(
      inputs.size() > 0,
      "torch.cat(): invalid number of inputs ",
      inputs.size());
  TORCH_CHECK(dimension >= 0, "torch.cat(): invalid dimension ", dimension);

  for (const Tensor& t : inputs) {
    TORCH_CHECK(
        t.device() == notSkippedTensor->device(),
        "torch.cat(): all input tensors must be on the same device. Received ",
        t.device(),
        " and ",
        notSkippedTensor->device());
  }

  TORCH_CHECK(
      out.device() == notSkippedTensor->device(),
      "torch.cat(): all input tensors and out must be on the same device, but inputs are on ",
      notSkippedTensor->device(),
      " and out is on ",
      out.device());

  // TODO: memory_format is now an argument?
  // // TODO: Factor out `compute_output_memory_format`
  // c10::MemoryFormat memory_format = compute_output_memory_format(inputs);

  std::vector<int64_t> size(notSkippedTensor->sizes().vec());

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  idx = 0;
  for(const Tensor& tensor : materialized_inputs) {
    if (should_skip(tensor)) {
      continue;
    }
    // TODO: Factor out `check_shape_except_dim`
    check_shape_except_dim(*notSkippedTensor, tensor, dimension, idx);
    cat_dim_size += at::native::size(tensor, dimension);
    idx++;
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

  // Get stream
  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    // TODO: Free this when no longer needed globally
    MPSGraphTensor** inputMPSGraphTensors_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache *cache_ = MPSGraphCache::getInstance();

  // Make string out of skipped tensor indices
  string skipped_indices_string = "";
  for(int idx : skipped_tensor_indices)
    skipped_indices_string += (std::to_string(idx)+",");
  string input_types = "";
  for(const Tensor& tensor : materialized_inputs)
    input_types += (getMPSTypeString(tensor.scalar_type())+",");

  @autoreleasepool {
    string key = "cat_out_mps:" + getMPSTypeString(result_type(inputs))
                                + ":" + to_string(inputs.size())
                                + ":" + skipped_indices_string
                                + ":" + input_types
                                + ":" + to_string(dimension);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          // Initialize graph
          MPSGraph *mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          // Create placeholders
          auto len_tensor_array = inputs.size() - skipped_tensor_indices.size();
          MPSGraphTensor* inputMPSGraphTensors[len_tensor_array];
          MPSGraphTensor* castInputMPSGraphTensors[len_tensor_array];

          int graph_tensor_idx = 0;
          for(const Tensor* tensor : input_tensors) {
            inputMPSGraphTensors[graph_tensor_idx] = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(tensor->scalar_type()) );
            if(getMPSDataType(result_type(inputs)) == MPSDataTypeBool) {
              castInputMPSGraphTensors[graph_tensor_idx] = [mpsGraph castTensor:inputMPSGraphTensors[graph_tensor_idx]
                                                                           toType:MPSDataTypeFloat32
                                                                             name:[NSString stringWithFormat:@"castInput%@", [NSNumber numberWithInt:graph_tensor_idx]]];
            }
            else {
              if(tensor->scalar_type() != result_type(inputs))
                castInputMPSGraphTensors[graph_tensor_idx] = [mpsGraph castTensor:inputMPSGraphTensors[graph_tensor_idx]
                                                                           toType:getMPSDataType(result_type(inputs))
                                                                             name:[NSString stringWithFormat:@"castInput%@", [NSNumber numberWithInt:graph_tensor_idx]]];
              else
                castInputMPSGraphTensors[graph_tensor_idx] = inputMPSGraphTensors[graph_tensor_idx];
            }
            graph_tensor_idx++;
          }

          auto inputTensorsArray = [NSArray arrayWithObjects:castInputMPSGraphTensors
                                                       count:len_tensor_array];
          // Use concatTensors to concatenate
          MPSGraphTensor* outputTensor = [mpsGraph concatTensors:inputTensorsArray
                                                       dimension:dimension // Maybe convert this from int64_t -> int32
                                                            name:nil];

          newCachedGraph->inputMPSGraphTensors_ = (MPSGraphTensor**)malloc(len_tensor_array * sizeof(MPSGraphTensor*));

          for(int i = 0; i < len_tensor_array; i++)
            newCachedGraph->inputMPSGraphTensors_[i] = inputMPSGraphTensors[i];
          if(getMPSDataType(result_type(inputs)) == MPSDataTypeBool)
            outputTensor = [mpsGraph castTensor:outputTensor
                                         toType:MPSDataTypeBool
                                           name:@"outputTensor"];
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    std::vector<Placeholder> inputPlaceholders;
    int i = 0;
    int t_idx = 0;
    for(const Tensor& tensor : materialized_inputs) {
      if(std::find(skipped_tensor_indices.begin(), skipped_tensor_indices.end(), i) == skipped_tensor_indices.end()) {
        Placeholder currentInputPlaceholder = Placeholder(cachedGraph->inputMPSGraphTensors_[t_idx], tensor);
        inputPlaceholders.push_back(currentInputPlaceholder);
        t_idx++;
      }
      i++;
    }

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);

    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    for (int i = 0; i < inputPlaceholders.size(); i++) {
      feeds[(inputPlaceholders[i]).getMPSGraphTensor()] = (inputPlaceholders[i]).getMPSGraphTensorData();
    }
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

}

void upsample_backward_out_mps(const Tensor& grad_output,
                               IntArrayRef output_size,
                               IntArrayRef input_size,
                               c10::optional<double> scales_h,
                               c10::optional<double> scales_w,
                               const Tensor& grad_input,
                               MPSGraphResizeMode requested_mode,
                               bool requested_align_corners
                               )
{
    using namespace mps;
    int64_t input_dims = input_size.size();

    TORCH_CHECK((input_dims == 4),
            "NCHW tensor expected for input");

    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor *gradInputTensor = nil, *gradOutputTensor = nil;
    };
    MPSGraphCache* cache_ = MPSGraphCache::getInstance();
    /* sizes */
    int64_t output_height = output_size[0];
    int64_t output_width = output_size[1];

    int64_t input_n = input_size[0];
    int64_t input_c = input_size[1];
    int64_t input_height = input_size[2];
    int64_t input_width = input_size[3];

    @autoreleasepool {
      MPSShape* output_shape = getMPSShape(grad_output);
      string key = string("upsample_backward:") + mps::getMPSShapeString(output_shape) + ":" +
                             getMPSTypeString(grad_output.scalar_type()) +
                             ":oh" + to_string(output_height) + ":ow" + to_string(output_width) +
                             ":ih" + to_string(input_height) + ":iw" + to_string(input_width) +
                             ":mode" + to_string(requested_mode);

      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
      if(!cachedGraph) {
        cachedGraph = static_cast<CachedGraph*>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

          CachedGraph *newCachedGraph = nil;
          @autoreleasepool {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);

              newCachedGraph->gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_input.scalar_type()), output_shape);
              MPSGraphTensor * shapeTensor = [mpsGraph constantWithScalar:0
                                                                    shape:@[[NSNumber numberWithLong: input_n],
                                                                            [NSNumber numberWithLong: input_c],
                                                                            [NSNumber numberWithLong:input_height],
                                                                            [NSNumber numberWithLong:input_width]]
                                                                 dataType:getMPSDataType(grad_output.scalar_type())];

              newCachedGraph->gradInputTensor  = [mpsGraph resizeWithGradientTensor: newCachedGraph->gradOutputTensor
                                                                           input: shapeTensor
                                                                            mode: requested_mode
                                                                    centerResult: true
                                                                    alignCorners: requested_align_corners
                                                                        layout: MPSGraphTensorNamedDataLayoutNCHW
                                                                            name: nil];

          }
          return newCachedGraph;
        }));
      }
      Placeholder gradOutputPlaceholder  = Placeholder(cachedGraph->gradOutputTensor, grad_output);
      Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor, grad_input);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData(),
      };
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          gradInputPlaceholder.getMPSGraphTensor() : gradInputPlaceholder.getMPSGraphTensorData()
      };
      runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
    }
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_mps) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input)
{
    upsample_backward_out_mps(grad_output, output_size, input_size, scales_h, scales_w, grad_input, MPSGraphResizeNearest, false);
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_mps) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input)
{
    upsample_backward_out_mps(grad_output, output_size, input_size, scales_h, scales_w, grad_input, MPSGraphResizeNearest, false);
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_mps) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input)
{
    upsample_backward_out_mps(grad_output, output_size, input_size, scales_h, scales_w, grad_input, MPSGraphResizeBilinear, align_corners);
}

void upsample_out_mps(const Tensor& input,
                      IntArrayRef output_size,
                      c10::optional<double> scales_h,
                      c10::optional<double> scales_w,
                      const Tensor& output,
                      MPSGraphResizeMode requested_mode,
                      bool requested_align_corners)
{
    // Get stream
    using namespace mps;
    struct CachedGraph : public MPSCachedGraph {
        CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    };
    MPSGraphCache* cache_ = MPSGraphCache::getInstance();

    /* sizes */
    int64_t output_height = output_size[0];
    int64_t output_width = output_size[1];
    @autoreleasepool {
      MPSShape* input_shape = getMPSShape(input);
      NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
      string key = string("upsample_2d:") + mps::getMPSShapeString(input_shape) + ":" +
                             getMPSTypeString(input.scalar_type()) +
                             ":h" + to_string(output_height) + ":w" + to_string(output_width) +
                             ":mode" + to_string(requested_mode);

      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
      if(!cachedGraph) {
        cachedGraph = static_cast<CachedGraph*>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

          CachedGraph *newCachedGraph = nil;

          @autoreleasepool {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);

              newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), input_shape);
              newCachedGraph->outputTensor = [mpsGraph resizeTensor:newCachedGraph->inputTensor
                                                               size:@[ @(output_height), @(output_width)]
                                                               mode:requested_mode
                                                               centerResult: true
                                                               alignCorners: requested_align_corners
                                                               layout: MPSGraphTensorNamedDataLayoutNCHW
                                                               name:nil];
          }
          return newCachedGraph;
        }));
      }
      Placeholder inputPlaceholder  = Placeholder(cachedGraph->inputTensor, input);
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, output);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      };
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
      };
      runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
    }
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_mps) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output)
{
    // Note: this differs from the CPU implementation in the way
    // ties are resolved wrt to nearest mostly in cases where the scale
    // is not an integer.
    // Example:
    // For upsampling from (2, 5) to (2, 16)
    // MPS:
    // tensor([[[[0., 0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.],
    // [5., 5., 5., 5., 6., 6., 6., 7., 7., 7., 8., 8., 8., 9., 9., 9.]]]])
    // CPU:
    // tensor([[[[0., 0., 0., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 4., 4., 4.],
    // [5., 5., 5., 6., 6., 6., 7., 7., 7., 7., 8., 8., 8., 9., 9., 9.]]]])
    using namespace mps;
    upsample_out_mps(input, output_size, scales_h, scales_w, output, MPSGraphResizeNearest, false);
}


TORCH_IMPL_FUNC(upsample_nearest2d_out_mps) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output)
{
    // Note: this differs from the CPU implementation in the way
    // ties are resolved wrt to nearest mostly in cases where the scale
    // is not an integer.
    // Example:
    // For upsampling from (2, 5) to (2, 16)
    // MPS:
    // tensor([[[[0., 0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.],
    // [5., 5., 5., 5., 6., 6., 6., 7., 7., 7., 8., 8., 8., 9., 9., 9.]]]])
    // CPU:
    // tensor([[[[0., 0., 0., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 4., 4., 4.],
    // [5., 5., 5., 6., 6., 6., 7., 7., 7., 7., 8., 8., 8., 9., 9., 9.]]]])
    using namespace mps;
    upsample_out_mps(input, output_size, scales_h, scales_w, output, MPSGraphResizeNearest, false);
}

TORCH_IMPL_FUNC(upsample_bilinear2d_out_mps) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output)
{
    using namespace mps;
    upsample_out_mps(input, output_size, scales_h, scales_w, output, MPSGraphResizeBilinear, align_corners);
}

void upsample1d_out_mps(const Tensor& input,
                      IntArrayRef output_size,
                      c10::optional<double> scales,
                      const Tensor& output,
                      MPSGraphResizeMode requested_mode)
{
    // Get stream
    using namespace mps;
    using CachedGraph = MPSUnaryCachedGraph;
    MPSGraphCache* cache_ = MPSGraphCache::getInstance();

    /* sizes */
    int64_t out_size = output_size[0];
    @autoreleasepool {
      MPSShape* input_shape = getMPSShape(input);
      NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
      string key = string("upsample_1d:") + mps::getMPSShapeString(input_shape) + ":" +
                             getMPSTypeString(input.scalar_type()) +
                             ":size" + to_string(out_size) +
                             ":mode" + to_string(requested_mode);

      CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);
      if(!cachedGraph) {
        cachedGraph = static_cast<CachedGraph*>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

          CachedGraph *newCachedGraph = nil;

          @autoreleasepool {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);

              newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), input_shape);
              newCachedGraph->outputTensor_ = [mpsGraph resizeTensor:newCachedGraph->inputTensor_
                                                               size:@[ @(out_size), @(1)]
                                                               mode:requested_mode
                                                               centerResult: true
                                                               alignCorners: true
                                                               layout: MPSGraphTensorNamedDataLayoutCHW
                                                               name:nil];
          }
          return newCachedGraph;
        }));
      }
      Placeholder inputPlaceholder  = Placeholder(cachedGraph->inputTensor_, input);
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      };
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
      };
      runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
    }
}


TORCH_IMPL_FUNC(upsample_nearest1d_out_mps) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    const Tensor& output)
{
    using namespace mps;
    upsample1d_out_mps(input, output_size, scales, output, MPSGraphResizeNearest);
}





} // namespace native
} // namespace at
