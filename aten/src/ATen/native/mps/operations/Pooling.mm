//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool2d_backward.h>
#include <ATen/ops/avg_pool2d_backward_native.h>
#include <ATen/ops/avg_pool2d_native.h>
#include <ATen/ops/max_pool2d_with_indices_backward_native.h>
#include <ATen/ops/max_pool2d_with_indices_native.h>
#endif

namespace at::native {
namespace mps {

struct PoolingCachedGraph : public MPSCachedGraph {
  PoolingCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor = nil;
  MPSGraphTensor* outputTensor = nil;
  MPSGraphTensor* indicesTensor = nil;
  MPSGraphTensor* gradOutputTensor = nil;
  MPSGraphTensor* divisorTensor = nil;
};

typedef MPSGraphTensor* (^PoolingOpBlock)(PoolingCachedGraph&, MPSGraphPooling2DOpDescriptor*);
#define PoolingOpFn(graph, desc) MPSGraphTensor*(mps::PoolingCachedGraph & graph, MPSGraphPooling2DOpDescriptor * desc)

// Pooling ops (1D/2D forward and backward Max and Average pooling)
static void pool2d_template(const Tensor& input,
                            const Tensor& output,
                            const c10::optional<Tensor>& indices_opt,
                            const c10::optional<Tensor>& grad_output_opt,
                            IntArrayRef kernel_size,
                            IntArrayRef stride,
                            IntArrayRef padding,
                            IntArrayRef dilation,
                            bool ceil_mode,
                            bool count_include_pad,
                            const c10::optional<int64_t> divisor_override,
                            PoolingOpBlock poolingBlock,
                            const c10::string& op_name) {
  if (!is_macos_13_or_newer()) {
    TORCH_CHECK(input.scalar_type() != ScalarType::Long,
                "MPS: ",
                op_name,
                " op with int64 input is supported natively starting from macOS 13.0.");
  }
  const int64_t ndims = input.ndimension();
  const Tensor& grad_output = *(at::borrow_from_optional_tensor(grad_output_opt));
  const Tensor& indices = *(at::borrow_from_optional_tensor(indices_opt));
  const bool is_backward_pass = grad_output.defined();
  const bool has_indices = indices.defined();
  const bool has_divisor = divisor_override.has_value() && divisor_override.value() != 0;
  const auto suggested_memory_format = input.suggest_memory_format();
  // for max_pool2d_with_indices() we cannot pass ChannelsLast (i.e., NHWC) to 'desc.dataLayout' in MPSGraph.
  // Because the returned indices will be selected based on NHWC memory layout which will
  // be incompatible with the PyTorch's global NCHW layout.
  const auto memory_format = has_indices ? MemoryFormat::Contiguous : suggested_memory_format;

  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
              op_name,
              ": kernel_size must either be a single int, or a tuple of two ints")
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
              op_name,
              ": stride must either be omitted, a single int, or a tuple of two ints")
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
              op_name,
              ": padding must be either be a single int, or a tuple of two ints");
  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
              op_name,
              ": dilation must be either a single int, or a tuple of two ints");

  if (suggested_memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(ndims == 4, "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (suggested_memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((ndims == 3 || ndims == 4), "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    AT_ERROR("Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }

  int padH = safe_downcast<int, int64_t>(padding[0]);
  int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);
  const int64_t nbatch = ndims == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(input,
                     kH,
                     kW,
                     dH,
                     dW,
                     padH,
                     padW,
                     dilationH,
                     dilationW,
                     nInputPlane,
                     inputHeight,
                     inputWidth,
                     outputHeight,
                     outputWidth,
                     memory_format);

  if (input.numel() == 0) {
    return;
  }

  auto output_memory_format = output.suggest_memory_format();
  // the output and indices are 'empty', so we could avoid unnecessary gatherView on empty tensors
  // by simply restriding them (instead of calling the costly Contiguous()).
  if (indices.suggest_memory_format() == MemoryFormat::ChannelsLast) {
    indices.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::Contiguous);
  }
  if (output.numel() == 0) {
    std::vector<int64_t> outputSizes{nInputPlane, outputHeight, outputWidth};
    if (ndims == 4) {
      outputSizes.insert(outputSizes.begin(), nbatch);
    }
    output.resize_(outputSizes);
  } else if (output_memory_format == MemoryFormat::ChannelsLast) {
    output.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::Contiguous);
    output_memory_format = MemoryFormat::Contiguous;
  }

  if (output.numel() == 0 || (is_backward_pass && grad_output.numel() == 0)) {
    return;
  }
  // workaround for issue #103039644: mismatching MPS vs. CPU results
  // when both ceil_mode and count_include_pad are True
  if (count_include_pad && ceil_mode) {
    padH = padW = 0;
  }
  @autoreleasepool {
    string key = op_name + getTensorsStringKey({input, indices, grad_output}) + ":K[" + getArrayRefString(kernel_size) +
        "]:S[" + getArrayRefString(stride) + "]:P[" + getArrayRefString(padding) + "]:D[" +
        getArrayRefString(dilation) + "]" + (ceil_mode ? ":ceil" : "") + (count_include_pad ? ":include_pad" : "") +
        (has_divisor ? ":divisor" : "") + ":" +
        (suggested_memory_format == MemoryFormat::ChannelsLast ? "NHWC" : "NCHW");

    MPSShape* inputShape = getMPSShape(input, memory_format);
    MPSShape* gradOutputShape = is_backward_pass ? getMPSShape(grad_output, memory_format) : nullptr;

    auto cachedGraph = LookUpOrCreateCachedGraph<PoolingCachedGraph>(key, [&](auto* mpsGraph, auto* newCachedGraph) {
      MPSGraphPooling2DOpDescriptor* desc = [MPSGraphPooling2DOpDescriptor
          descriptorWithKernelWidth:kW
                       kernelHeight:kH
                          strideInX:dW
                          strideInY:dH
                    dilationRateInX:dilationW
                    dilationRateInY:dilationH
                        paddingLeft:padW
                       paddingRight:ceil_mode ? padW * dW : padW
                         paddingTop:padH
                      paddingBottom:ceil_mode ? padH * dH : padH
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:memory_format == MemoryFormat::ChannelsLast ? MPSGraphTensorNamedDataLayoutNHWC
                                                                                : MPSGraphTensorNamedDataLayoutNCHW];
      desc.ceilMode = (padW == 0 && padH == 0) ? ceil_mode : false;
      if (has_indices) {
        desc.returnIndicesMode = MPSGraphPoolingReturnIndicesGlobalFlatten2D;
        desc.returnIndicesDataType = MPSDataTypeInt32;
      }
      newCachedGraph->inputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input.scalar_type()), inputShape);
      if (is_backward_pass) {
        newCachedGraph->gradOutputTensor =
            mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output.scalar_type()), gradOutputShape);
      }
      if (has_divisor) {
        newCachedGraph->divisorTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, @[ @1 ]);
      }
      MPSGraphTensor* outputTensor = poolingBlock(*newCachedGraph, desc);
      // with desc.dataLayout = NHWC (i.e., ChannelsLast), the results need to be converted back to NCHW
      newCachedGraph->outputTensor =
          memory_format == MemoryFormat::ChannelsLast ? convertNHWCtoNCHW(mpsGraph, outputTensor) : outputTensor;
    });

    MPSStream* mpsStream = getCurrentMPSStream();
    // in case of ChannelsLast we don't perform gather() in placeholder to avoid implicit conversion to NCHW
    Placeholder inputPlaceholder =
        Placeholder(cachedGraph->inputTensor, input, inputShape, memory_format != MemoryFormat::ChannelsLast);
    Placeholder gradOutputPlaceholder = !is_backward_pass
        ? Placeholder()
        : Placeholder(
              cachedGraph->gradOutputTensor, grad_output, gradOutputShape, memory_format != MemoryFormat::ChannelsLast);
    Placeholder indicesPlaceholder = has_indices ? Placeholder(cachedGraph->indicesTensor, indices) : Placeholder();
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, output);
    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    NSMutableDictionary* results = [[NSMutableDictionary new] autorelease];

    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    results[outputPlaceholder.getMPSGraphTensor()] = outputPlaceholder.getMPSGraphTensorData();

    if (cachedGraph->gradOutputTensor) {
      feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();
    }
    if (cachedGraph->indicesTensor) {
      if (is_backward_pass) {
        feeds[indicesPlaceholder.getMPSGraphTensor()] = indicesPlaceholder.getMPSGraphTensorData();
      } else {
        results[indicesPlaceholder.getMPSGraphTensor()] = indicesPlaceholder.getMPSGraphTensorData();
      }
    }
    MPSScalar divisor_scalar;
    if (cachedGraph->divisorTensor) {
      const float divisor = float(kH * kW) / (float)divisor_override.value();
      divisor_scalar = getMPSScalar(divisor, ScalarType::Float);
      feeds[cachedGraph->divisorTensor] = getMPSGraphTensorFromScalar(mpsStream, divisor_scalar);
    }

    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);

    if (output_memory_format != suggested_memory_format) {
      const_cast<Tensor&>(output) = output.to(suggested_memory_format);
    }
  }
}

static void avg_pool2d_template(const Tensor& input,
                                const Tensor& output,
                                const c10::optional<Tensor>& grad_output_opt,
                                IntArrayRef kernel_size,
                                IntArrayRef stride,
                                IntArrayRef padding,
                                IntArrayRef dilation,
                                bool ceil_mode,
                                bool count_include_pad,
                                const c10::optional<int64_t> divisor_override,
                                const c10::string& op_name) {
  const Tensor& grad_output = *(at::borrow_from_optional_tensor(grad_output_opt));
  const bool is_backward_pass = grad_output.defined();
  const bool use_divisor = divisor_override.has_value() && divisor_override.value() != 0;

  // custom divisor isn't supported natively in avgPooling2DWithSourceTensor().
  // For Float input type, we work around it by multiplying divisor after avgPooling2D.
  // However, for Long type, the accumulated error when multiplying the divisor
  // would produce results that mismatch CPU results.
  if (use_divisor && input.scalar_type() == ScalarType::Long) {
    TORCH_WARN_ONCE("MPS: passing divisor to Average Pooling op with int64 input is ",
                    "not supported on MPS backend. ",
                    "Falling back on CPU. This may have performance implications.");
    if (!is_backward_pass) {
      const_cast<Tensor&>(output) =
          at::avg_pool2d(input.to("cpu"), kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
              .clone()
              .to("mps");
    } else {
      const_cast<Tensor&>(output) = at::avg_pool2d_backward(grad_output.to("cpu"),
                                                            input.to("cpu"),
                                                            kernel_size,
                                                            stride,
                                                            padding,
                                                            ceil_mode,
                                                            count_include_pad,
                                                            divisor_override)
                                        .clone()
                                        .to("mps");
    }
    return;
  }

  mps::PoolingOpBlock pooling_op_block = ^PoolingOpFn(cachedGraph, desc) {
    MPSGraph* mpsGraph = cachedGraph.graph();
    const int64_t ndims = input.ndimension();
    MPSShape* paddingShape = nil;
    MPSGraphTensor* paddedTensor = cachedGraph.inputTensor;

    // workaround for issue #103039644: mismatching MPS vs. CPU results
    // when both ceilMode and includeZeroPadToAverage are True
    const bool explicit_padding = count_include_pad && ceil_mode;
    if (explicit_padding) {
      std::vector<NSNumber*> padVec(ndims, @(0));
      padVec[ndims - 1] = @(padding.size() == 1 ? padding[0] : padding[1]);
      padVec[ndims - 2] = @(ndims > 3 ? padding[0] : 0);
      paddingShape = [NSArray arrayWithObjects:padVec.data() count:ndims];
      paddedTensor = [mpsGraph padTensor:cachedGraph.inputTensor
                         withPaddingMode:MPSGraphPaddingModeZero
                             leftPadding:paddingShape
                            rightPadding:paddingShape
                           constantValue:0.0
                                    name:nil];
      paddedTensor = [mpsGraph identityWithTensor:paddedTensor name:nil];
    } else {
      desc.includeZeroPadToAverage = count_include_pad;
    }
    if (use_divisor) {
      desc.includeZeroPadToAverage = YES;
    }

    if (!is_backward_pass) {
      MPSGraphTensor* avgPoolTensor = [mpsGraph avgPooling2DWithSourceTensor:paddedTensor descriptor:desc name:nil];
      if (cachedGraph.divisorTensor) {
        // workaround: custom divisor isn't supported by MPS backend, so we scale manually
        return [mpsGraph multiplicationWithPrimaryTensor:avgPoolTensor
                                         secondaryTensor:cachedGraph.divisorTensor
                                                    name:nil];
      } else {
        return avgPoolTensor;
      }
    } else { // backward pass
      MPSGraphTensor* scaledGradTensor = cachedGraph.gradOutputTensor;
      if (cachedGraph.divisorTensor) {
        scaledGradTensor = [mpsGraph multiplicationWithPrimaryTensor:cachedGraph.gradOutputTensor
                                                     secondaryTensor:cachedGraph.divisorTensor
                                                                name:nil];
      }
      MPSGraphTensor* avgPoolTensor = [mpsGraph avgPooling2DGradientWithGradientTensor:scaledGradTensor
                                                                          sourceTensor:paddedTensor
                                                                            descriptor:desc
                                                                                  name:nil];
      if (explicit_padding) {
        return [mpsGraph padGradientWithIncomingGradientTensor:avgPoolTensor
                                                  sourceTensor:cachedGraph.inputTensor
                                                   paddingMode:MPSGraphPaddingModeZero
                                                   leftPadding:paddingShape
                                                  rightPadding:paddingShape
                                                          name:nil];

      } else {
        return avgPoolTensor;
      }
    }
  };

  pool2d_template(input,
                  output,
                  c10::nullopt,
                  grad_output_opt,
                  kernel_size,
                  stride,
                  padding,
                  {1, 1},
                  ceil_mode,
                  count_include_pad,
                  divisor_override,
                  pooling_op_block,
                  op_name);
}

} // namespace mps

Tensor mps_max_pool2d(const Tensor& input,
                      IntArrayRef kernel_size,
                      IntArrayRef stride,
                      IntArrayRef padding,
                      IntArrayRef dilation,
                      bool ceil_mode) {
  Tensor output = at::empty({0}, input.options(), MemoryFormat::Contiguous);
  mps::PoolingOpBlock pooling_op_block = ^PoolingOpFn(cachedGraph, desc) {
    MPSGraph* mpsGraph = cachedGraph.graph();
    return [mpsGraph maxPooling2DWithSourceTensor:cachedGraph.inputTensor descriptor:desc name:nil];
  };
  mps::pool2d_template(input,
                       output,
                       c10::nullopt,
                       c10::nullopt,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       ceil_mode,
                       false,
                       c10::nullopt,
                       pooling_op_block,
                       "max_pool2d");

  return output;
}

Tensor mps_max_pool2d_backward(const Tensor& grad_output,
                               const Tensor& input,
                               IntArrayRef kernel_size,
                               IntArrayRef stride,
                               IntArrayRef padding,
                               IntArrayRef dilation,
                               bool ceil_mode) {
  Tensor grad_input = at::empty(input.sizes(), input.options(), MemoryFormat::Contiguous);
  mps::PoolingOpBlock pooling_op_block = ^PoolingOpFn(cachedGraph, desc) {
    MPSGraph* mpsGraph = cachedGraph.graph();
    return [mpsGraph maxPooling2DGradientWithGradientTensor:cachedGraph.gradOutputTensor
                                               sourceTensor:cachedGraph.inputTensor
                                                 descriptor:desc
                                                       name:nil];
  };
  mps::pool2d_template(input,
                       grad_input,
                       c10::nullopt,
                       grad_output,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       ceil_mode,
                       false,
                       c10::nullopt,
                       pooling_op_block,
                       "max_pool2d_backward");

  return grad_input;
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_out_mps)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& output,
 const Tensor& indices) {
  auto indices_memory_format = indices.suggest_memory_format();

  mps::PoolingOpBlock pooling_op_block = ^PoolingOpFn(cachedGraph, desc) {
    MPSGraph* mpsGraph = cachedGraph.graph();
    NSArray<MPSGraphTensor*>* poolOutputs = [mpsGraph maxPooling2DReturnIndicesWithSourceTensor:cachedGraph.inputTensor
                                                                                     descriptor:desc
                                                                                           name:nil];
    cachedGraph.indicesTensor = mps::castMPSTensor(mpsGraph, poolOutputs[1], ScalarType::Long);
    return poolOutputs[0];
  };
  mps::pool2d_template(input,
                       output,
                       indices,
                       c10::nullopt,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       ceil_mode,
                       false,
                       c10::nullopt,
                       pooling_op_block,
                       "max_pool2d_indices");

  if (indices_memory_format == MemoryFormat::ChannelsLast) {
    const_cast<Tensor&>(indices) = indices.to(MemoryFormat::ChannelsLast);
  }
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_mps)
(const Tensor& grad_output,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& indices,
 const Tensor& grad_input) {
  mps::PoolingOpBlock pooling_op_block = ^PoolingOpFn(cachedGraph, desc) {
    MPSGraph* mpsGraph = cachedGraph.graph();
    return [mpsGraph maxPooling2DGradientWithGradientTensor:cachedGraph.gradOutputTensor
                                               sourceTensor:cachedGraph.inputTensor
                                                 descriptor:desc
                                                       name:nil];
  };
  mps::pool2d_template(input,
                       grad_input,
                       indices,
                       grad_output,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       ceil_mode,
                       false,
                       c10::nullopt,
                       pooling_op_block,
                       "max_pool2d_indices_backward");
}

TORCH_IMPL_FUNC(avg_pool2d_out_mps)
(const Tensor& input,
 int64_t kH,
 int64_t kW,
 int64_t dH,
 int64_t dW,
 int64_t padH,
 int64_t padW,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override,
 const Tensor& output) {
  mps::avg_pool2d_template(input,
                           output,
                           c10::nullopt,
                           {kH, kW},
                           {dH, dW},
                           {padH, padW},
                           {1, 1},
                           ceil_mode,
                           count_include_pad,
                           divisor_override,
                           "avg_pool2d");
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_mps)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override,
 const Tensor& gradInput) {
  mps::avg_pool2d_template(input,
                           gradInput,
                           gradOutput,
                           kernel_size,
                           stride,
                           padding,
                           {1, 1},
                           ceil_mode,
                           count_include_pad,
                           divisor_override,
                           "avg_pool2d_backward");
}

} // namespace at::native
