//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/constant_pad_nd_native.h>
#include <ATen/ops/reflection_pad1d_backward_native.h>
#include <ATen/ops/reflection_pad1d_native.h>
#include <ATen/ops/reflection_pad2d_backward_native.h>
#include <ATen/ops/reflection_pad2d_native.h>
#include <ATen/ops/reflection_pad3d_backward_native.h>
#include <ATen/ops/reflection_pad3d_native.h>
#include <ATen/ops/replication_pad1d_backward_native.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad2d_backward_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad3d_backward_native.h>
#include <ATen/ops/replication_pad3d_native.h>
#endif

namespace at::native {
namespace mps {

// Pad operations (1D/2D/3D forward and backward)
static Tensor& pad_out_template(Tensor& output,
                                const Tensor& input_,
                                IntArrayRef padding,
                                const std::optional<Tensor>& grad_output_opt,
                                MPSGraphPaddingMode mode,
                                double constantValue,
                                const std::string& op_name) {
  using CachedGraph = MPSUnaryGradCachedGraph;
  const int padding_size = (int)padding.size();
  int padding_dim = padding_size / 2; // either 1D, 2D, or 3D

  TORCH_CHECK(
      padding_size == 2 || padding_size == 4 || padding_size == 6, "invalid padding argument of size ", padding_size);

  const Tensor& grad_output_ = *(at::borrow_from_optional_tensor(grad_output_opt));
  const bool is_backward_pass = grad_output_.defined();

  int64_t nbatch = 1;
  int64_t ndims = input_.ndimension();

  TORCH_CHECK(ndims >= (int64_t)padding_dim,
              "Length of pad should be no more than twice the number of "
              "dimensions of the input. Pad length is ",
              padding_size,
              "while the input has ",
              ndims,
              "dimensions.");

  // number of input dims with ConstantPad could be less than 2
  int dim_w = padding_dim;
  int dim_h = padding_dim - 1;
  int dim_d = padding_dim - 2;
  int dim_slices = 0;

  if (!is_backward_pass && mode != MPSGraphPaddingModeConstant && ndims > padding_dim) {
    bool valid_dims = input_.size(1) != 0 && input_.size(padding_dim) != 0;
    TORCH_CHECK((ndims == 1 + padding_dim && valid_dims) ||
                    (ndims == 2 + padding_dim && valid_dims && input_.size(1 + padding_dim) != 0),
                "3D or 4D (batch mode) tensor expected for input, but got: ",
                input_);
  }

  if (ndims == padding_dim) {
    dim_w--;
    dim_h--;
    dim_d--;
  } else if (ndims > padding_dim + 1) {
    const int dim_diff = (int)ndims - padding_dim - 1;
    // this virtually inflates the padding with zeros if ndims > padding_dim + 2
    padding_dim += dim_diff - 1;
    dim_w += dim_diff;
    dim_h += dim_diff;
    dim_d += dim_diff;
    dim_slices++;
    nbatch = input_.size(0);
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding_size > 2 ? padding[2] : 0;
  int64_t pad_b = padding_size > 2 ? padding[3] : 0;
  int64_t pad_front = padding_size > 4 ? padding[4] : 0;
  int64_t pad_back = padding_size > 4 ? padding[5] : 0;

  int64_t nplane = input_.size(dim_slices);
  int64_t input_w = input_.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;
  int64_t input_h = padding_dim > 1 ? input_.size(dim_h) : 0;
  int64_t output_h = padding_dim > 1 ? input_h + pad_t + pad_b : 0;
  int64_t input_d = padding_dim > 2 ? input_.size(dim_d) : 0;
  int64_t output_d = padding_dim > 2 ? input_d + pad_front + pad_back : 0;

  Tensor grad_output, input = input_;

  if (!is_backward_pass) {
    TORCH_CHECK(output_w >= 1 || output_h >= padding_dim - 1,
                "input (H: ",
                input_h,
                ", W: ",
                input_w,
                ") is too small. Calculated "
                "output H: ",
                output_h,
                " W: ",
                output_w);

    std::vector<int64_t> outputSizes;
    if (mode == MPSGraphPaddingModeConstant) {
      // support arbitrary input dimensions for constant pad.
      auto input_sizes = input_.sizes();
      auto ori_padding_dim = padding_size / 2;
      auto l_diff = ndims - ori_padding_dim;

      for (size_t i = 0; i < (size_t)l_diff; i++) {
        outputSizes.emplace_back(input_sizes[i]);
      }
      for (const auto i : c10::irange((size_t)ori_padding_dim)) {
        auto pad_idx = padding.size() - ((i + 1) * 2);
        auto new_dim = input_sizes[l_diff + i] + padding[pad_idx] + padding[pad_idx + 1];
        outputSizes.emplace_back(new_dim);
      }
    } else {
      // these checks are only relevant for reflection padding (code taken from ReflectionPad.cpp)
      if (mode == MPSGraphPaddingModeReflect) {
        TORCH_CHECK(pad_l < input_w && pad_r < input_w,
                    "Argument #4: Padding size should be less than the corresponding "
                    "input dimension, but got: padding (",
                    pad_l,
                    ", ",
                    pad_r,
                    ") at dimension ",
                    dim_w,
                    " of input ",
                    input_.sizes());

        if (padding_dim > 1) {
          TORCH_CHECK(pad_t < input_h && pad_b < input_h,
                      "Argument #6: Padding size should be less than the corresponding "
                      "input dimension, but got: padding (",
                      pad_t,
                      ", ",
                      pad_b,
                      ") at dimension ",
                      dim_h,
                      " of input ",
                      input_.sizes());
        }
        if (padding_dim > 2) {
          TORCH_CHECK(pad_front < input_d && pad_back < input_d,
                      "Argument #8: Padding size should be less than the corresponding "
                      "input dimension, but got: padding (",
                      pad_front,
                      ", ",
                      pad_back,
                      ") at dimension ",
                      dim_d,
                      " of input ",
                      input_.sizes());
        }
      }
      outputSizes.insert(outputSizes.begin(), output_w);
      if (padding_dim >= 2)
        outputSizes.insert(outputSizes.begin(), output_h);
      if (padding_dim >= 3)
        outputSizes.insert(outputSizes.begin(), output_d);
      if (ndims >= 1 + padding_dim)
        outputSizes.insert(outputSizes.begin(), nplane);
      if (ndims >= 2 + padding_dim)
        outputSizes.insert(outputSizes.begin(), nbatch);
    }

    output.resize_(outputSizes);

    if (output.numel() == 0) {
      return output;
    }
    if (input_.numel() == 0) {
      output.fill_(constantValue);
      return output;
    }
    input = input_.contiguous();
  } else {
    TORCH_CHECK(output_w == grad_output_.size(dim_w),
                "gradOutput width unexpected. Expected: ",
                output_w,
                ", Got: ",
                grad_output_.size(dim_w));
    if (padding_dim > 1) {
      TORCH_CHECK(output_h == grad_output_.size(dim_h),
                  "gradOutput height unexpected. Expected: ",
                  output_h,
                  ", Got: ",
                  grad_output_.size(dim_h));
    }
    output.resize_as_(input);
    if (output.numel() == 0 || grad_output_.numel() == 0)
      return output;
    grad_output = grad_output_.contiguous();
  }

  const uint32_t dims_mask = (1U << ndims) - 1;
  uint32_t startMask = dims_mask, endMask = dims_mask;
  std::vector<NSNumber*> leftPadVec(ndims, @(0));
  std::vector<NSNumber*> rightPadVec(ndims, @(0));
  std::vector<NSNumber*> startsVec(ndims, @(0));
  std::vector<NSNumber*> endsVec(ndims, @(0));
  std::vector<NSNumber*> stridesVec(ndims, @(1));

  for (int64_t pdim = 0; pdim < padding_size / 2; pdim++) {
    const int64_t leftIdx = pdim * 2;
    const int64_t rightIdx = pdim * 2 + 1;
    const int64_t padIdx = ndims - pdim - 1;

    leftPadVec[padIdx] = @(padding[leftIdx]);
    rightPadVec[padIdx] = @(padding[rightIdx]);
    // workaround for negative padding issue in backward pass
    if (is_backward_pass) {
      if (padding[leftIdx] < 0) {
        leftPadVec[padIdx] = @(0);
        startsVec[padIdx] = @(-padding[leftIdx]);
        startMask &= ~(1U << padIdx);
      }
      if (padding[rightIdx] < 0) {
        rightPadVec[padIdx] = @(0);
        endsVec[padIdx] = @(input.size(padIdx) + padding[rightIdx]);
        endMask &= ~(1U << padIdx);
      }
    }
  }
  MPSShape* leftPadding = [NSArray arrayWithObjects:leftPadVec.data() count:ndims];
  MPSShape* rightPadding = [NSArray arrayWithObjects:rightPadVec.data() count:ndims];

  MPSDataType dataType = getMPSScalarType(input.scalar_type());
  // workaround for Bool type assert with Constant padding
  if (input.scalar_type() == kBool) {
    dataType = MPSDataTypeInt8;
  }

  @autoreleasepool {
    std::string key = op_name + getTensorsStringKey({input, grad_output, output}) + ":[" + getArrayRefString(padding) +
        "]:" + std::to_string(constantValue);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, dataType, getMPSShape(input));
      const bool needsSlice = startMask != dims_mask || endMask != dims_mask;

      if (!is_backward_pass) {
        MPSGraphTensor* padTensor = [mpsGraph padTensor:newCachedGraph->inputTensor_
                                        withPaddingMode:mode
                                            leftPadding:leftPadding
                                           rightPadding:rightPadding
                                          constantValue:constantValue
                                                   name:nil];
        // workaround for the right padding bug in Monterey
        if (needsSlice) {
          newCachedGraph->gradInputTensor_ =
              [mpsGraph sliceTensor:padTensor
                             starts:[NSArray arrayWithObjects:startsVec.data() count:ndims]
                               ends:[NSArray arrayWithObjects:endsVec.data() count:ndims]
                            strides:[NSArray arrayWithObjects:stridesVec.data() count:ndims]
                          startMask:startMask
                            endMask:endMask
                        squeezeMask:0
                               name:nil];
        } else {
          newCachedGraph->gradInputTensor_ = padTensor;
        }
      } else {
        newCachedGraph->gradOutputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, dataType, getMPSShape(grad_output));
        MPSGraphTensor* padGradTensor =
            [mpsGraph padGradientWithIncomingGradientTensor:newCachedGraph->gradOutputTensor_
                                               sourceTensor:newCachedGraph->inputTensor_
                                                paddingMode:mode
                                                leftPadding:leftPadding
                                               rightPadding:rightPadding
                                                       name:nil];
        // workaround for negative padding issue with padGradientWithIncomingGradientTensor()
        if (needsSlice) {
          for (auto i : c10::irange(ndims)) {
            auto start = [startsVec[i] intValue];
            auto input_size = input.size(i);
            // TODO: It should be possible to make this case work. Currently
            // MPSGraph can crash if start >= input_size, so we raise an error
            // to prevent the crash.
            TORCH_INTERNAL_ASSERT(start == 0 || start < input_size);
          }
          newCachedGraph->gradInputTensor_ =
              [mpsGraph sliceGradientTensor:padGradTensor
                           fwdInShapeTensor:[mpsGraph shapeOfTensor:newCachedGraph->inputTensor_ name:nil]
                                     starts:[NSArray arrayWithObjects:startsVec.data() count:ndims]
                                       ends:[NSArray arrayWithObjects:endsVec.data() count:ndims]
                                    strides:[NSArray arrayWithObjects:stridesVec.data() count:ndims]
                                  startMask:startMask
                                    endMask:endMask
                                squeezeMask:0
                                       name:nil];
        } else {
          newCachedGraph->gradInputTensor_ = padGradTensor;
        }
      }
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input, nullptr, true, dataType);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, output, nullptr, true, dataType);
    Placeholder gradOutputPlaceholder = !is_backward_pass
        ? Placeholder()
        : Placeholder(cachedGraph->gradOutputTensor_, grad_output, nullptr, true, dataType);

    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    if (is_backward_pass) {
      feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();
    }
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return output;
}
} // namespace mps

// 1D Reflection and Replication Padding
TORCH_IMPL_FUNC(reflection_pad1d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output),
                        input,
                        padding,
                        std::nullopt,
                        MPSGraphPaddingModeReflect,
                        0.0,
                        "reflection_pad1d_out_mps");
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input),
                        input,
                        padding,
                        grad_output,
                        MPSGraphPaddingModeReflect,
                        0.0,
                        "reflection_pad1d_backward_out_mps");
}

TORCH_IMPL_FUNC(replication_pad1d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output),
                        input,
                        padding,
                        std::nullopt,
                        MPSGraphPaddingModeClampToEdge,
                        0.0,
                        "replication_pad1d_out_mps");
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input),
                        input,
                        padding,
                        grad_output,
                        MPSGraphPaddingModeClampToEdge,
                        0.0,
                        "replication_pad1d_backward_out_mps");
}

// 2D Reflection and Replication Padding
Tensor& reflection_pad2d_out_mps(const Tensor& input, IntArrayRef padding, Tensor& output) {
  return mps::pad_out_template(output, input, padding, std::nullopt, MPSGraphPaddingModeReflect, 0.0, __func__);
}

Tensor reflection_pad2d_mps(const Tensor& input, IntArrayRef padding) {
  Tensor output = at::empty({0}, input.options());
  return mps::pad_out_template(output, input, padding, std::nullopt, MPSGraphPaddingModeReflect, 0.0, __func__);
}

Tensor& reflection_pad2d_backward_out_mps(const Tensor& grad_output,
                                          const Tensor& input,
                                          IntArrayRef padding,
                                          Tensor& grad_input) {
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeReflect, 0.0, __func__);
}

Tensor reflection_pad2d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeReflect, 0.0, __func__);
}

TORCH_IMPL_FUNC(replication_pad2d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output),
                        input,
                        padding,
                        std::nullopt,
                        MPSGraphPaddingModeClampToEdge,
                        0.0,
                        "replication_pad2d_out_mps");
}

Tensor& replication_pad2d_backward_out_mps(const Tensor& grad_output,
                                           const Tensor& input,
                                           IntArrayRef padding,
                                           Tensor& grad_input) {
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, 0.0, __func__);
}

Tensor replication_pad2d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, 0.0, __func__);
}

// 3D Reflection and Replication Padding
TORCH_IMPL_FUNC(reflection_pad3d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output),
                        input,
                        padding,
                        std::nullopt,
                        MPSGraphPaddingModeReflect,
                        0.0,
                        "reflection_pad3d_out_mps");
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  grad_input.resize_as_(input).zero_();
  mps::pad_out_template(const_cast<Tensor&>(grad_input),
                        input,
                        padding,
                        grad_output,
                        MPSGraphPaddingModeReflect,
                        0.0,
                        "reflection_pad3d_backward_out_mps");
}

TORCH_IMPL_FUNC(replication_pad3d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output),
                        input,
                        padding,
                        std::nullopt,
                        MPSGraphPaddingModeClampToEdge,
                        0.0,
                        "replication_pad3d_out_mps");
}

Tensor& replication_pad3d_backward_out_mps(const Tensor& grad_output,
                                           const Tensor& input,
                                           IntArrayRef padding,
                                           Tensor& grad_input) {
  grad_input.resize_as_(input).zero_();
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, 0.0, __func__);
}

Tensor replication_pad3d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, MPSGraphPaddingModeClampToEdge, 0.0, __func__);
}

// backward pass is explicitly handled in autograd by negating the "pad" argument
Tensor constant_pad_nd_mps(const Tensor& self, IntArrayRef pad, const Scalar& value) {
  if (pad.empty()) {
    return self.clone();
  }
  if (pad.size() > 6) {
    TORCH_WARN_ONCE("MPS: The constant padding of more than 3 dimensions is not currently supported natively. ",
                    "It uses View Ops default implementation to run. This may have performance implications.");
    return at::native::constant_pad_nd(self, pad, value);
  }
  Tensor output = at::empty({0}, self.options());
  return mps::pad_out_template(
      output, self, pad, std::nullopt, MPSGraphPaddingModeConstant, value.toDouble(), __func__);
}

} // namespace at::native
