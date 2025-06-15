//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/avg_pool3d_backward_native.h>
#include <ATen/ops/avg_pool3d_native.h>

// Using Metal's built-in packed_int4 type for more efficient parameter passing

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool2d_backward.h>
#include <ATen/ops/avg_pool2d_backward_native.h>
#include <ATen/ops/avg_pool2d_native.h>
#include <ATen/ops/avg_pool3d.h>
#include <ATen/ops/avg_pool3d_backward.h>
#include <ATen/ops/avg_pool3d_backward_native.h>
#include <ATen/ops/avg_pool3d_native.h>
#include <ATen/ops/max_pool2d_backward_native.h>
#include <ATen/ops/max_pool2d_native.h>
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
                            const std::optional<Tensor>& indices_opt,
                            const std::optional<Tensor>& grad_output_opt,
                            IntArrayRef kernel_size,
                            IntArrayRef stride,
                            IntArrayRef padding,
                            IntArrayRef dilation,
                            bool ceil_mode,
                            bool count_include_pad,
                            const std::optional<int64_t> divisor_override,
                            PoolingOpBlock poolingBlock,
                            const std::string& op_name) {
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
              ": padding must either be a single int, or a tuple of two ints");
  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
              op_name,
              ": dilation must be either a single int, or a tuple of two ints");

  if (suggested_memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(ndims == 4, "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (suggested_memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((ndims == 3 || ndims == 4), "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
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
    std::string key = op_name + getTensorsStringKey({input, indices, grad_output}) + ":K[" +
        getArrayRefString(kernel_size) + "]:S[" + getArrayRefString(stride) + "]:P[" + getArrayRefString(padding) +
        "]:D[" + getArrayRefString(dilation) + "]" + (ceil_mode ? ":ceil" : "") +
        (count_include_pad ? ":include_pad" : "") + (has_divisor ? ":divisor" : "") + ":" +
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

    // MPS TODO: Using strided API causes invalid indices to be generated if the original format is NHWC
    //           Output is still correct, but indices are not matching. Disable it for now and use the old
    //           gather path to solve the strides.
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor,
                                               input,
                                               inputShape,
                                               memory_format != MemoryFormat::ChannelsLast,
                                               MPSDataTypeInvalid,
                                               /*useMPSStridedAPI=*/false);
    Placeholder gradOutputPlaceholder = !is_backward_pass ? Placeholder()
                                                          : Placeholder(cachedGraph->gradOutputTensor,
                                                                        grad_output,
                                                                        gradOutputShape,
                                                                        memory_format != MemoryFormat::ChannelsLast,
                                                                        MPSDataTypeInvalid,
                                                                        /*useMPSStridedAPI=*/false);
    Placeholder indicesPlaceholder = has_indices
        ? Placeholder(
              cachedGraph->indicesTensor, indices, nullptr, true, MPSDataTypeInvalid, /*useMPSStridedAPI=*/false)
        : Placeholder();
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor, output, nullptr, false, MPSDataTypeInvalid, false);
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
                                const std::optional<Tensor>& grad_output_opt,
                                IntArrayRef kernel_size,
                                IntArrayRef stride,
                                IntArrayRef padding,
                                IntArrayRef dilation,
                                bool ceil_mode,
                                bool count_include_pad,
                                const std::optional<int64_t> divisor_override,
                                const std::string& op_name) {
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
      output.copy_(at::avg_pool2d(
          input.to("cpu"), kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override));
    } else {
      output.copy_(at::avg_pool2d_backward(grad_output.to("cpu"),
                                           input.to("cpu"),
                                           kernel_size,
                                           stride,
                                           padding,
                                           ceil_mode,
                                           count_include_pad,
                                           divisor_override));
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
        return
            [mpsGraph multiplicationWithPrimaryTensor:avgPoolTensor
                                      secondaryTensor:mps::castMPSTensor(
                                                          mpsGraph, cachedGraph.divisorTensor, [avgPoolTensor dataType])
                                                 name:nil];
      } else {
        return avgPoolTensor;
      }
    } else { // backward pass
      MPSGraphTensor* scaledGradTensor = cachedGraph.gradOutputTensor;
      if (cachedGraph.divisorTensor) {
        scaledGradTensor = [mpsGraph
            multiplicationWithPrimaryTensor:cachedGraph.gradOutputTensor
                            secondaryTensor:mps::castMPSTensor(
                                                mpsGraph, cachedGraph.divisorTensor, [scaledGradTensor dataType])
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
                  std::nullopt,
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
                       std::nullopt,
                       std::nullopt,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       ceil_mode,
                       false,
                       std::nullopt,
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
                       std::nullopt,
                       grad_output,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       ceil_mode,
                       false,
                       std::nullopt,
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
                       std::nullopt,
                       kernel_size,
                       stride,
                       padding,
                       dilation,
                       ceil_mode,
                       false,
                       std::nullopt,
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
                       std::nullopt,
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
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  mps::avg_pool2d_template(input,
                           output,
                           std::nullopt,
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
 std::optional<int64_t> divisor_override,
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

// 3D Average Pooling implementation using custom Metal shader
static void avg_pool3d_template(const Tensor& input,
                                const Tensor& output,
                                const std::optional<Tensor>& grad_output_opt,
                                IntArrayRef kernel_size,
                                IntArrayRef stride,
                                IntArrayRef padding,
                                IntArrayRef dilation,
                                bool ceil_mode,
                                bool count_include_pad,
                                const std::optional<int64_t> divisor_override,
                                const c10::string& op_name) {
  const Tensor& grad_output = *(at::borrow_from_optional_tensor(grad_output_opt));
  const bool is_backward_pass = grad_output.defined();
  const bool use_divisor = divisor_override.has_value() && divisor_override.value() != 0;

  // Check macOS version compatibility
  TORCH_CHECK(is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS),
              "avg_pool3d is only supported on MPS for MacOS_13_2 or newer");

  // Handle special cases
  if (use_divisor && input.scalar_type() == ScalarType::Long) {
    TORCH_WARN_ONCE("MPS: passing divisor to Average Pooling op with int64 input is ",
                    "not supported on MPS backend. ",
                    "Falling back on CPU. This may have performance implications.");
    if (!is_backward_pass) {
      output.copy_(at::avg_pool3d(
          input.to("cpu"), kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override));
    } else {
      output.copy_(at::avg_pool3d_backward(grad_output.to("cpu"),
                                           input.to("cpu"),
                                           kernel_size,
                                           stride,
                                           padding,
                                           ceil_mode,
                                           count_include_pad,
                                           divisor_override));
    }
    return;
  }

  // Check input dimensions
  TORCH_CHECK(input.dim() == 5, "avg_pool3d: Expected 5D tensor as input, got ", input.dim(), "D tensor");

  // Extract dimensions
  const int64_t nbatch = input.size(0);
  const int64_t channels = input.size(1);
  const int64_t input_depth = input.size(2);
  const int64_t input_height = input.size(3);
  const int64_t input_width = input.size(4);

  // Extract pooling parameters
  const int64_t kD = kernel_size[0];
  const int64_t kH = kernel_size.size() > 1 ? kernel_size[1] : kD;
  const int64_t kW = kernel_size.size() > 2 ? kernel_size[2] : kH;

  const int64_t dD = stride.empty() ? kD : stride[0];
  const int64_t dH = stride.empty() ? kH : (stride.size() > 1 ? stride[1] : dD);
  const int64_t dW = stride.empty() ? kW : (stride.size() > 2 ? stride[2] : dH);

  const int64_t pD = padding.empty() ? 0 : padding[0];
  const int64_t pH = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pD);
  const int64_t pW = padding.empty() ? 0 : (padding.size() > 2 ? padding[2] : pH);

  const int64_t dilationD = dilation.empty() ? 1 : dilation[0];
  const int64_t dilationH = dilation.empty() ? 1 : (dilation.size() > 1 ? dilation[1] : dilationD);
  const int64_t dilationW = dilation.empty() ? 1 : (dilation.size() > 2 ? dilation[2] : dilationH);

  // Check dilation
  TORCH_CHECK(dilationD == 1 && dilationH == 1 && dilationW == 1,
              "avg_pool3d: Dilation > 1 not supported in MPS backend");

  // Calculate output dimensions
  const int64_t output_depth = pooling_output_shape<int64_t>(input_depth, kD, pD, dD, dilationD, ceil_mode);
  const int64_t output_height = pooling_output_shape<int64_t>(input_height, kH, pH, dH, dilationH, ceil_mode);
  const int64_t output_width = pooling_output_shape<int64_t>(input_width, kW, pW, dW, dilationW, ceil_mode);

  // Early return for empty tensors
  if (input.numel() == 0 || output.numel() == 0 || (is_backward_pass && grad_output.numel() == 0)) {
    return;
  }

  // Get MPS stream
  MPSStream* mpsStream = getCurrentMPSStream();

  // Note: We only synchronize when necessary, not at the beginning of every operation
  // This improves performance by allowing more operations to be batched together

  // Get MPS data types
  MPSDataType inputDataType = mps::getMPSDataType(input.scalar_type());
  MPSDataType outputDataType = mps::getMPSDataType(output.scalar_type());

  // Create command buffer
  id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();

  // Create compute command encoder
  id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
  TORCH_CHECK(computeEncoder != nil, "Failed to create compute command encoder");

  // Get compute pipeline state
  NSString* kernelName = nil;
  if (input.scalar_type() == at::ScalarType::Float) {
    kernelName = is_backward_pass ? @"avg_pool3d_backward_float" : @"avg_pool3d_float";
  } else if (input.scalar_type() == at::ScalarType::Half) {
    kernelName = is_backward_pass ? @"avg_pool3d_backward_half" : @"avg_pool3d_half";
  } else {
    TORCH_CHECK(false, "Unsupported data type for avg_pool3d on MPS: ", input.scalar_type());
  }

  // Get compute pipeline state from bundled library
  static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
  id<MTLComputePipelineState> pipelineState = lib.getPipelineStateForFunc([kernelName UTF8String]);
  [computeEncoder setComputePipelineState:pipelineState];

  // Set buffers
  if (!is_backward_pass) {
    // Forward pass
    id<MTLBuffer> inputBuffer = mps::getMTLBufferStorage(input);
    id<MTLBuffer> outputBuffer = mps::getMTLBufferStorage(output);

    [computeEncoder setBuffer:outputBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:inputBuffer offset:0 atIndex:1];
  } else {
    // Backward pass
    id<MTLBuffer> gradInputBuffer = mps::getMTLBufferStorage(output); // grad_input
    id<MTLBuffer> gradOutputBuffer = mps::getMTLBufferStorage(grad_output);

    [computeEncoder setBuffer:gradInputBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:gradOutputBuffer offset:0 atIndex:1];
  }

  // Set parameters
  int batch_size = static_cast<int>(nbatch);
  int channels_val = static_cast<int>(channels);

  // Create packed_int4 vectors for more efficient parameter passing
  packed_int4 input_dims = {static_cast<int>(input_depth),
                            static_cast<int>(input_height),
                            static_cast<int>(input_width),
                            static_cast<int>(output_depth)};

  packed_int4 output_dims = {
      static_cast<int>(output_height), static_cast<int>(output_width), static_cast<int>(kD), static_cast<int>(kH)};

  packed_int4 kernel_dims = {static_cast<int>(kW), static_cast<int>(dD), static_cast<int>(dH), static_cast<int>(dW)};

  packed_int4 padding_dims = {
      static_cast<int>(pD), static_cast<int>(pH), static_cast<int>(pW), count_include_pad ? 1 : 0};

  int divisor_override_val = divisor_override.has_value() ? static_cast<int>(divisor_override.value()) : 0;

  [computeEncoder setBytes:&batch_size length:sizeof(int) atIndex:2];
  [computeEncoder setBytes:&channels_val length:sizeof(int) atIndex:3];
  [computeEncoder setBytes:&input_dims length:sizeof(packed_int4) atIndex:4];
  [computeEncoder setBytes:&output_dims length:sizeof(packed_int4) atIndex:5];
  [computeEncoder setBytes:&kernel_dims length:sizeof(packed_int4) atIndex:6];
  [computeEncoder setBytes:&padding_dims length:sizeof(packed_int4) atIndex:7];
  [computeEncoder setBytes:&divisor_override_val length:sizeof(int) atIndex:8];

  // Calculate grid and threadgroup sizes
  MTLSize gridSize, threadgroupSize;
  if (!is_backward_pass) {
    // Forward pass
    int total_output_size = nbatch * channels * output_depth * output_height * output_width;
    gridSize = MTLSizeMake(total_output_size, 1, 1);
  } else {
    // Backward pass
    int total_input_size = nbatch * channels * input_depth * input_height * input_width;
    gridSize = MTLSizeMake(total_input_size, 1, 1);
  }

  // Calculate optimal threadgroup size
  threadgroupSize = MTLSizeMake(
      std::min(static_cast<NSUInteger>(pipelineState.maxTotalThreadsPerThreadgroup), static_cast<NSUInteger>(256)),
      1,
      1);

  // Dispatch threads
  [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

  // End encoding
  [computeEncoder endEncoding];

  // Only commit the command buffer without waiting for completion
  // This allows better parallelism with other operations
  // Note: We're not synchronizing here to improve performance
}

TORCH_IMPL_FUNC(avg_pool3d_out_mps)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  // Check input dimensions
  TORCH_CHECK(input.dim() == 5, "avg_pool3d: Expected 5D tensor as input, got ", input.dim(), "D tensor");

  // Extract dimensions
  const int64_t nbatch = input.size(0);
  const int64_t channels = input.size(1);
  const int64_t input_depth = input.size(2);
  const int64_t input_height = input.size(3);
  const int64_t input_width = input.size(4);

  // Extract pooling parameters
  const int64_t kD = kernel_size[0];
  const int64_t kH = kernel_size.size() > 1 ? kernel_size[1] : kD;
  const int64_t kW = kernel_size.size() > 2 ? kernel_size[2] : kH;

  const int64_t dD = stride.empty() ? kD : stride[0];
  const int64_t dH = stride.empty() ? kH : (stride.size() > 1 ? stride[1] : dD);
  const int64_t dW = stride.empty() ? kW : (stride.size() > 2 ? stride[2] : dH);

  const int64_t pD = padding.empty() ? 0 : padding[0];
  const int64_t pH = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pD);
  const int64_t pW = padding.empty() ? 0 : (padding.size() > 2 ? padding[2] : pH);

  // Calculate output dimensions
  const int64_t output_depth = pooling_output_shape<int64_t>(input_depth, kD, pD, dD, 1, ceil_mode);
  const int64_t output_height = pooling_output_shape<int64_t>(input_height, kH, pH, dH, 1, ceil_mode);
  const int64_t output_width = pooling_output_shape<int64_t>(input_width, kW, pW, dW, 1, ceil_mode);

  // Check output dimensions
  TORCH_CHECK(output.dim() == 5, "avg_pool3d: Expected 5D tensor as output, got ", output.dim(), "D tensor");
  TORCH_CHECK(output.size(0) == nbatch, "avg_pool3d: Output batch size doesn't match input batch size");
  TORCH_CHECK(output.size(1) == channels, "avg_pool3d: Output channels doesn't match input channels");

  // Check data type
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float || input.scalar_type() == at::ScalarType::Half,
              "avg_pool3d: MPS implementation only supports Float and Half data types, got ",
              input.scalar_type());
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "avg_pool3d: Input and output data types must match");

  // Create a new stream for non-contiguous tensors to avoid command buffer conflicts
  if (!input.is_contiguous()) {
    // Create a contiguous copy of the input tensor
    Tensor contiguous_input = input.contiguous();

    // Use the contiguous input tensor
    avg_pool3d_template(contiguous_input,
                        output,
                        std::nullopt,
                        kernel_size,
                        stride,
                        padding,
                        {1, 1, 1},
                        ceil_mode,
                        count_include_pad,
                        divisor_override,
                        "avg_pool3d");
  } else {
    // Use the original input tensor directly
    avg_pool3d_template(input,
                        output,
                        std::nullopt,
                        kernel_size,
                        stride,
                        padding,
                        {1, 1, 1},
                        ceil_mode,
                        count_include_pad,
                        divisor_override,
                        "avg_pool3d");
  }
}

TORCH_IMPL_FUNC(avg_pool3d_backward_out_mps)
(const Tensor& grad_output,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& grad_input) {
  // Check input dimensions
  TORCH_CHECK(input.dim() == 5, "avg_pool3d_backward: Expected 5D tensor as input, got ", input.dim(), "D tensor");
  TORCH_CHECK(grad_output.dim() == 5,
              "avg_pool3d_backward: Expected 5D tensor as grad_output, got ",
              grad_output.dim(),
              "D tensor");

  // Extract dimensions
  const int64_t nbatch = input.size(0);
  const int64_t channels = input.size(1);
  const int64_t input_depth = input.size(2);
  const int64_t input_height = input.size(3);
  const int64_t input_width = input.size(4);

  // Check grad_input dimensions
  TORCH_CHECK(grad_input.dim() == 5,
              "avg_pool3d_backward: Expected 5D tensor as grad_input, got ",
              grad_input.dim(),
              "D tensor");
  TORCH_CHECK(grad_input.size(0) == nbatch,
              "avg_pool3d_backward: grad_input batch size doesn't match input batch size");
  TORCH_CHECK(grad_input.size(1) == channels, "avg_pool3d_backward: grad_input channels doesn't match input channels");
  TORCH_CHECK(grad_input.size(2) == input_depth, "avg_pool3d_backward: grad_input depth doesn't match input depth");
  TORCH_CHECK(grad_input.size(3) == input_height, "avg_pool3d_backward: grad_input height doesn't match input height");
  TORCH_CHECK(grad_input.size(4) == input_width, "avg_pool3d_backward: grad_input width doesn't match input width");

  // Check data type
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float || input.scalar_type() == at::ScalarType::Half,
              "avg_pool3d_backward: MPS implementation only supports Float and Half data types, got ",
              input.scalar_type());
  TORCH_CHECK(input.scalar_type() == grad_output.scalar_type() && input.scalar_type() == grad_input.scalar_type(),
              "avg_pool3d_backward: Input, grad_output, and grad_input data types must match");

  // Handle non-contiguous tensors
  if (!input.is_contiguous() || !grad_output.is_contiguous()) {
    // Create contiguous copies of the input and grad_output tensors
    Tensor contiguous_input = input.is_contiguous() ? input : input.contiguous();
    Tensor contiguous_grad_output = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    // Use the contiguous tensors
    avg_pool3d_template(contiguous_input,
                        grad_input,
                        contiguous_grad_output,
                        kernel_size,
                        stride,
                        padding,
                        {1, 1, 1},
                        ceil_mode,
                        count_include_pad,
                        divisor_override,
                        "avg_pool3d_backward");
  } else {
    // Use the original tensors directly
    avg_pool3d_template(input,
                        grad_input,
                        grad_output,
                        kernel_size,
                        stride,
                        padding,
                        {1, 1, 1},
                        ceil_mode,
                        count_include_pad,
                        divisor_override,
                        "avg_pool3d_backward");
  }
}

} // namespace at::native
