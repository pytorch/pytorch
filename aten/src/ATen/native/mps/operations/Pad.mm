//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {
namespace mps {

// Pad operations (1D/2D/3D forward and backward)
Tensor& pad_out_template(Tensor &output, const Tensor &input_, IntArrayRef padding,
                         const c10::optional<Tensor>& grad_output_opt,
                         MPSGraphPaddingMode mode, double constantValue, const string op_name)
{
  const int padding_size = (int) padding.size();
  const int padding_dim = padding_size / 2; // either 1D, 2D, or 3D

  TORCH_CHECK(padding_size == 2 || padding_size == 4 || padding_size == 6,
              "invalid padding argument of size ", padding_size);

  const Tensor& grad_output_ = *(at::borrow_from_optional_tensor(grad_output_opt));
  const bool is_backward_pass = grad_output_.defined();

  int64_t nbatch = 1;
  int64_t ndims = input_.ndimension();
  // number of input dims with ConstantPad could be less than 2
  int dim_w = ndims > 1 ? padding_dim : 0;
  int dim_h = padding_dim - 1;
  int dim_d = padding_dim - 2;
  int dim_slices = 0;

  if (!is_backward_pass && ndims > 1) {
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
      else if (ndims > 1)
        output.resize_({nbatch, nplane, output_w});
      else
        output.resize_({output_w});
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

  std::vector<NSNumber*> leftPadVec(ndims, @(0));
  std::vector<NSNumber*> rightPadVec(ndims, @(0));
  leftPadVec [ndims - 1] = @(pad_l);
  rightPadVec[ndims - 1] = @(pad_r);
  if (padding_dim >= 2) {
    leftPadVec [ndims - 2] = @(pad_t);
    rightPadVec[ndims - 2] = @(pad_b);
  }
  if (padding_dim >= 3) {
    leftPadVec [ndims - 3] = @(pad_front);
    rightPadVec[ndims - 3] = @(pad_back);
  }
  MPSShape *leftPadding  = [NSArray arrayWithObjects:leftPadVec.data() count:ndims];
  MPSShape *rightPadding = [NSArray arrayWithObjects:rightPadVec.data() count:ndims];

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
                                                   constantValue:constantValue
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
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt,
                        MPSGraphPaddingModeReflect, 0.0, "reflection_pad1d_out_mps");
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input), input, padding, grad_output,
                        MPSGraphPaddingModeReflect, 0.0, "reflection_pad1d_backward_out_mps");
}

TORCH_IMPL_FUNC(replication_pad1d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt,
                        MPSGraphPaddingModeClampToEdge, 0.0, "replication_pad1d_out_mps");
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input), input, padding, grad_output,
                        MPSGraphPaddingModeClampToEdge, 0.0, "replication_pad1d_backward_out_mps");
}

// 2D Reflection and Replication Padding
Tensor& reflection_pad2d_out_mps(const Tensor& input, IntArrayRef padding, Tensor& output)
{
  return mps::pad_out_template(output, input, padding, c10::nullopt, MPSGraphPaddingModeReflect, 0.0, __func__);
}

Tensor reflection_pad2d_mps(const Tensor& input, IntArrayRef padding)
{
  Tensor output = at::empty({0}, input.options());
  return mps::pad_out_template(output, input, padding, c10::nullopt, MPSGraphPaddingModeReflect, 0.0, __func__);
}

Tensor& reflection_pad2d_backward_out_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeReflect, 0.0, __func__);
}

Tensor reflection_pad2d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding)
{
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeReflect, 0.0, __func__);
}

TORCH_IMPL_FUNC(replication_pad2d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt,
                        MPSGraphPaddingModeClampToEdge, 0.0, "replication_pad2d_out_mps");
}

Tensor& replication_pad2d_backward_out_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, 0.0, __func__);
}

Tensor replication_pad2d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding)
{
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, 0.0, __func__);
}

// 3D Reflection and Replication Padding
TORCH_IMPL_FUNC(reflection_pad3d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt,
                        MPSGraphPaddingModeReflect, 0.0, "reflection_pad3d_out_mps");
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input), input, padding, grad_output,
                        MPSGraphPaddingModeReflect, 0.0, "reflection_pad3d_backward_out_mps");
}

TORCH_IMPL_FUNC(replication_pad3d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output)
{
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, c10::nullopt,
                        MPSGraphPaddingModeClampToEdge, 0.0, "replication_pad3d_out_mps");
}

Tensor& replication_pad3d_backward_out_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, Tensor& grad_input)
{
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, 0.0, __func__);
}

Tensor replication_pad3d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding)
{
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, 0.0, __func__);
}

// backward pass is exlicitly handled in autograd by negating the "pad" argument
Tensor constant_pad_nd_mps(const Tensor& self, IntArrayRef pad, const Scalar& value)
{
  if (pad.size() > 6) {
    TORCH_WARN_ONCE("MPS: The constant padding of more than 3 dimensions is not currently supported natively. ",
                    "It uses View Ops default implementation to run. This may have performance implications.");
    return at::native::constant_pad_nd(self, pad, value);
  }
  Tensor output = at::empty({0}, self.options());
  return mps::pad_out_template(output, self, pad, c10::nullopt, MPSGraphPaddingModeConstant, value.toDouble(), __func__);
}

} // namespace native
} // namespace at
