//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/metal/common.h>

#include <algorithm>
#include <limits>
#include <numeric>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/constant_pad_nd_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
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

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ReplicationPad_metallib.h>
#endif

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

static MTLSize pad_threadgroup(id<MTLComputePipelineState> pso, NSUInteger gx, NSUInteger gy, NSUInteger gz) {
  const auto maxTPG = [pso maxTotalThreadsPerThreadgroup];
  const auto tg_x = std::min<NSUInteger>(maxTPG, gx);
  const auto tg_y = std::min<NSUInteger>(maxTPG / tg_x, gy);
  const auto tg_z = std::min<NSUInteger>(maxTPG / (tg_x * tg_y), gz);
  return MTLSizeMake(tg_x, tg_y, tg_z);
}

static void replication_pad1d_kernel_mps(const Tensor& input_, IntArrayRef padding, const Tensor& output) {
  if (output.numel() == 0 || input_.numel() == 0) {
    return;
  }
  auto input = input_.contiguous();
  const bool output_needs_copy = !output.is_contiguous();
  auto output_buf = output_needs_copy ? at::empty(output.sizes(), output.options()) : output;
  auto output_c = output_buf;
  if (input.dim() == 2) {
    input = input.unsqueeze(0);
    output_c = output_c.unsqueeze(0);
  }
  TORCH_INTERNAL_ASSERT(input.dim() == 3 && output_c.dim() == 3);

  const auto nbatch = c10::checked_convert<int32_t>(input.size(0), "int32_t");
  const auto nplane = c10::checked_convert<int32_t>(input.size(1), "int32_t");
  const auto input_W = c10::checked_convert<int32_t>(input.size(2), "int32_t");
  const auto output_W = c10::checked_convert<int32_t>(output_c.size(2), "int32_t");
  const std::array<int32_t, 4> sizes_pad = {input_W,
                                            output_W,
                                            c10::checked_convert<int32_t>(padding[0], "int32_t"),
                                            c10::checked_convert<int32_t>(padding[1], "int32_t")};

  auto pso = lib.getPipelineStateForFunc("replication_pad1d_forward_" + scalarToMetalTypeString(input));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "replication_pad1d_forward", {input, output_c}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, input, output_c, sizes_pad);
      [encoder dispatchThreads:MTLSizeMake(output_W, nplane, nbatch)
          threadsPerThreadgroup:pad_threadgroup(pso, output_W, nplane, nbatch)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  if (output_needs_copy) {
    output.copy_(output_buf);
  }
}

static void replication_pad1d_backward_kernel_mps(const Tensor& grad_output_,
                                                  const Tensor& input,
                                                  IntArrayRef padding,
                                                  const Tensor& grad_input) {
  if (grad_input.numel() == 0 || grad_output_.numel() == 0) {
    return;
  }
  auto grad_output = grad_output_.contiguous();
  const bool grad_input_needs_copy = !grad_input.is_contiguous();
  auto grad_input_buf = grad_input_needs_copy ? at::empty(grad_input.sizes(), grad_input.options()) : grad_input;
  auto grad_input_c = grad_input_buf;
  if (input.dim() == 2) {
    grad_output = grad_output.unsqueeze(0);
    grad_input_c = grad_input_c.unsqueeze(0);
  }
  TORCH_INTERNAL_ASSERT(grad_output.dim() == 3 && grad_input_c.dim() == 3);

  const auto nbatch = c10::checked_convert<int32_t>(grad_input_c.size(0), "int32_t");
  const auto nplane = c10::checked_convert<int32_t>(grad_input_c.size(1), "int32_t");
  const auto input_W = c10::checked_convert<int32_t>(grad_input_c.size(2), "int32_t");
  const auto output_W = c10::checked_convert<int32_t>(grad_output.size(2), "int32_t");
  const std::array<int32_t, 4> sizes_pad = {input_W,
                                            output_W,
                                            c10::checked_convert<int32_t>(padding[0], "int32_t"),
                                            c10::checked_convert<int32_t>(padding[1], "int32_t")};

  auto pso = lib.getPipelineStateForFunc("replication_pad1d_backward_" + scalarToMetalTypeString(grad_input_c));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "replication_pad1d_backward", {grad_output, grad_input_c}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, grad_output, grad_input_c, sizes_pad);
      [encoder dispatchThreads:MTLSizeMake(input_W, nplane, nbatch)
          threadsPerThreadgroup:pad_threadgroup(pso, input_W, nplane, nbatch)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  if (grad_input_needs_copy) {
    grad_input.copy_(grad_input_buf);
  }
}

template <typename index_t>
static void constant_pad_nd_strided_kernel_mps(const Tensor& input,
                                               const Tensor& output,
                                               const Scalar& value,
                                               const std::vector<index_t>& output_sizes,
                                               const std::vector<index_t>& input_sizes,
                                               const std::vector<index_t>& input_strides,
                                               const std::vector<index_t>& output_strides,
                                               const std::vector<index_t>& left_pad,
                                               NSUInteger grid_x,
                                               NSUInteger grid_y) {
  const auto metal_ndim = static_cast<uint32_t>(output.dim());
  const auto index_type = sizeof(index_t) == sizeof(uint32_t) ? "u32" : "u64";
  auto pso = lib.getPipelineStateForFunc("constant_pad_nd_strided_" + std::string(index_type) + "_" +
                                         scalarToMetalTypeString(input));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "constant_pad_nd", {input, output}, stream);
      auto encoder = stream->commandEncoder();
      auto fill_value = getMPSScalar(value, input.scalar_type());
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder,
                  input,
                  output,
                  output_sizes,
                  input_sizes,
                  input_strides,
                  output_strides,
                  left_pad,
                  metal_ndim,
                  fill_value);
      mtl_dispatch2DJob(encoder, pso, grid_x, grid_y);
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
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
  mps::replication_pad1d_kernel_mps(input, padding, output);
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  mps::replication_pad1d_backward_kernel_mps(grad_output, input, padding, grad_input);
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
  TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size());

  const auto ndim = self.dim();
  const auto padding_dim = static_cast<int64_t>(pad.size() / 2);
  TORCH_CHECK(ndim >= padding_dim,
              "Length of pad should be no more than twice the number of dimensions of the input. Pad length is ",
              pad.size(),
              " while the input has ",
              ndim,
              " dimensions.");

  bool all_pads_non_positive = true;
  auto cropped = self;
  for (const auto dim : c10::irange(ndim - padding_dim, ndim)) {
    const auto pad_idx = 2 * (ndim - dim - 1);
    if (pad[pad_idx] < 0) {
      cropped = cropped.narrow(dim, -pad[pad_idx], cropped.size(dim) + pad[pad_idx]);
    } else if (pad[pad_idx] != 0) {
      all_pads_non_positive = false;
    }
    if (pad[pad_idx + 1] < 0) {
      cropped = cropped.narrow(dim, 0, cropped.size(dim) + pad[pad_idx + 1]);
    } else if (pad[pad_idx + 1] != 0) {
      all_pads_non_positive = false;
    }
  }

  if (all_pads_non_positive && cropped.is_contiguous()) {
    return cropped.clone();
  }

  if (self.is_quantized() || self.is_conj() || self.is_neg() || ndim > c10::metal::max_ndim) {
    return at::native::constant_pad_nd(self, pad, value);
  }

  std::vector<int64_t> output_sizes;
  if (all_pads_non_positive) {
    output_sizes = cropped.sizes().vec();
  } else {
    output_sizes = self.sizes().vec();
    for (const auto dim : c10::irange(ndim - padding_dim, ndim)) {
      const auto pad_idx = 2 * (ndim - dim - 1);
      const auto output_size = self.size(dim) + pad[pad_idx] + pad[pad_idx + 1];
      TORCH_CHECK(output_size >= 0,
                  "The input size ",
                  self.size(dim),
                  ", plus negative padding ",
                  pad[pad_idx],
                  " and ",
                  pad[pad_idx + 1],
                  " resulted in a negative output size, which is invalid. Check dimension ",
                  dim,
                  " of your input.");
      output_sizes[dim] = output_size;
    }
  }

  Tensor output;
  if (all_pads_non_positive && cropped.is_non_overlapping_and_dense()) {
    output = at::empty_strided(cropped.sizes(), cropped.strides(), cropped.options());
  } else if (all_pads_non_positive) {
    output = at::empty_like(cropped);
  } else {
    output = at::empty(output_sizes, self.options().memory_format(self.suggest_memory_format()));
  }
  if (output.numel() == 0) {
    return output;
  }

  const Scalar kernel_value = all_pads_non_positive ? Scalar(0) : value;
  constexpr auto uint_max = std::numeric_limits<uint32_t>::max();
  const auto type = mps::scalarToMetalTypeString(self);
  const auto stream = getCurrentMPSStream();
  const bool use_dense = padding_dim <= 3 && cropped.is_contiguous() && output.is_contiguous();

  if (use_dense) {
    const auto in_w = cropped.size(-1);
    const auto in_h = padding_dim >= 2 ? cropped.size(-2) : 1;
    const auto in_d = padding_dim >= 3 ? cropped.size(-3) : 1;
    const auto out_w = output.size(-1);
    const auto out_h = padding_dim >= 2 ? output.size(-2) : 1;
    const auto out_d = padding_dim >= 3 ? output.size(-3) : 1;
    const auto grid_x = c10::metal::ceil_div(out_w, static_cast<int64_t>(c10::metal::ILP_PER_THREAD));
    const auto grid_z = output.numel() / (out_w * out_h);
    const auto left_w = padding_dim >= 1 ? std::max<int64_t>(pad[0], 0) : 0;
    const auto left_h = padding_dim >= 2 ? std::max<int64_t>(pad[2], 0) : 0;
    const auto left_d = padding_dim >= 3 ? std::max<int64_t>(pad[4], 0) : 0;
    const std::array<int64_t, 9> params64 = {in_w, in_h, in_d, out_w, out_h, out_d, left_w, left_h, left_d};
    const bool params_fit_uint = std::all_of(
        params64.begin(), params64.end(), [](int64_t param) { return param <= std::numeric_limits<uint32_t>::max(); });

    if (params_fit_uint && grid_x <= uint_max && out_h <= uint_max && grid_z <= uint_max) {
      std::array<uint32_t, 9> params;
      std::transform(
          params64.begin(), params64.end(), params.begin(), [](int64_t param) { return static_cast<uint32_t>(param); });
      auto pso = mps::lib.getPipelineStateForFunc("constant_pad_nd_dense_" + type);
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          getMPSProfiler().beginProfileKernel(pso, "constant_pad_nd", {cropped, output}, stream);
          auto encoder = stream->commandEncoder();
          auto fill_value = mps::getMPSScalar(kernel_value, self.scalar_type());
          [encoder setComputePipelineState:pso];
          mps::mtl_setArgs(encoder, cropped, output, params, fill_value);
          [encoder dispatchThreads:MTLSizeMake(grid_x, out_h, grid_z)
              threadsPerThreadgroup:mps::pad_threadgroup(pso, grid_x, out_h, grid_z)];
          getMPSProfiler().endProfileKernel(pso, stream);
        }
      });
      return output;
    }
  }

  std::vector<int64_t> dim_order(ndim);
  std::iota(dim_order.begin(), dim_order.end(), 0);
  std::stable_sort(dim_order.begin(), dim_order.end(), [&](int64_t lhs, int64_t rhs) {
    return output.stride(lhs) < output.stride(rhs);
  });

  std::vector<int64_t> ordered_output_sizes(ndim);
  std::vector<int64_t> ordered_input_sizes(ndim);
  std::vector<int64_t> ordered_output_strides(ndim);
  std::vector<int64_t> ordered_input_strides(ndim);
  std::vector<int64_t> ordered_left_pad(ndim);
  for (const auto ordered_dim : c10::irange(ndim)) {
    const auto dim = dim_order[ordered_dim];
    ordered_output_sizes[ordered_dim] = output.size(dim);
    ordered_input_sizes[ordered_dim] = cropped.size(dim);
    ordered_output_strides[ordered_dim] = output.stride(dim);
    ordered_input_strides[ordered_dim] = cropped.stride(dim);
    if (dim >= ndim - padding_dim) {
      const auto pad_idx = 2 * (ndim - dim - 1);
      ordered_left_pad[ordered_dim] = std::max<int64_t>(pad[pad_idx], 0);
    }
  }

  const auto inner = ordered_output_sizes[0];
  const auto grid_x = c10::metal::ceil_div(inner, static_cast<int64_t>(c10::metal::ILP_PER_THREAD));
  const auto grid_y = output.numel() / inner;
  if (grid_x > uint_max || grid_y > uint_max) {
    return at::native::constant_pad_nd(self, pad, value);
  }

  const auto values_fit_uint = [](const std::vector<int64_t>& values) {
    return std::all_of(
        values.begin(), values.end(), [](int64_t value) { return value <= std::numeric_limits<uint32_t>::max(); });
  };
  const bool use_32bit_index = mps::offsetsFitIn<uint32_t>(cropped, output) && values_fit_uint(ordered_output_sizes) &&
      values_fit_uint(ordered_input_sizes) && values_fit_uint(ordered_output_strides) &&
      values_fit_uint(ordered_input_strides) && values_fit_uint(ordered_left_pad);

  if (use_32bit_index) {
    const auto to_uint = [](const std::vector<int64_t>& values) {
      std::vector<uint32_t> result(values.size());
      std::transform(
          values.begin(), values.end(), result.begin(), [](int64_t value) { return static_cast<uint32_t>(value); });
      return result;
    };
    mps::constant_pad_nd_strided_kernel_mps(cropped,
                                            output,
                                            kernel_value,
                                            to_uint(ordered_output_sizes),
                                            to_uint(ordered_input_sizes),
                                            to_uint(ordered_input_strides),
                                            to_uint(ordered_output_strides),
                                            to_uint(ordered_left_pad),
                                            grid_x,
                                            grid_y);
  } else {
    const auto to_uint64 = [](const std::vector<int64_t>& values) {
      return std::vector<uint64_t>(values.begin(), values.end());
    };
    mps::constant_pad_nd_strided_kernel_mps(cropped,
                                            output,
                                            kernel_value,
                                            to_uint64(ordered_output_sizes),
                                            to_uint64(ordered_input_sizes),
                                            to_uint64(ordered_input_strides),
                                            to_uint64(ordered_output_strides),
                                            to_uint64(ordered_left_pad),
                                            grid_x,
                                            grid_y);
  }
  return output;
}

} // namespace at::native
