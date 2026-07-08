//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ConvUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/_mps_convolution_native.h>
#include <ATen/ops/_mps_convolution_transpose_native.h>
#include <ATen/ops/mps_convolution_backward_native.h>
#include <ATen/ops/mps_convolution_transpose_backward_native.h>
#include <fmt/format.h>

namespace at::native {

// `memory_format` selects NDHWC vs NCDHW; `use_dhwio` selects DHWIO vs OIDHW
// (caller must insert the matching in-graph weight transpose).
static void fill_conv3d_desc(MPSGraphConvolution3DOpDescriptor* descriptor_,
                             NSUInteger strideInX,
                             NSUInteger strideInY,
                             NSUInteger strideInZ,
                             NSUInteger dilationRateInX,
                             NSUInteger dilationRateInY,
                             NSUInteger dilationRateInZ,
                             NSUInteger paddingHorizontal,
                             NSUInteger paddingVertical,
                             NSUInteger paddingDepth,
                             c10::MemoryFormat memory_format,
                             bool use_dhwio,
                             NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.strideInZ = strideInZ;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;
  descriptor_.dilationRateInZ = dilationRateInZ;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;
  descriptor_.paddingFront = paddingDepth;
  descriptor_.paddingBack = paddingDepth;

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::ChannelsLast3d) ? MPSGraphTensorNamedDataLayoutNDHWC
                                                                               : MPSGraphTensorNamedDataLayoutNCDHW;
  descriptor_.weightsLayout = use_dhwio ? MPSGraphTensorNamedDataLayoutDHWIO : MPSGraphTensorNamedDataLayoutOIDHW;

  descriptor_.groups = groups; // not yet tested in Xcode/C++
}

// Exact-stride match: a sliced view of CL3d has CL-like strides but isn't
// NHWC-packed; the raw-buffer NDHWC path would misread it (#180984).
static bool is_packed_channels_last_3d(const Tensor& t) {
  return t.dim() == 5 &&
      t.suggest_memory_format(/*channels_last_strides_exact_match=*/true) == at::MemoryFormat::ChannelsLast3d;
}

// DHWIO costs one in-graph weight transpose per call; only worth it when
// Cin/groups is large enough and the kernel is not factorized.
static bool conv3d_dhwio_is_beneficial(IntArrayRef weight_size) {
  constexpr int64_t kMinCinPerGroup = 4; // skip first-layer Cin=3, depthwise Cin/g=1.
  constexpr int64_t kMinKernelDim = 2; // skip 1x3x3, 3x1x1, 1x1x1.
  return weight_size.size() == 5 && weight_size[1] >= kMinCinPerGroup && weight_size[2] >= kMinKernelDim &&
      weight_size[3] >= kMinKernelDim && weight_size[4] >= kMinKernelDim;
}

// Force the tensor's stride pattern to match `desc_layout`; MPSGraph's 3D
// conv path takes a slow strided route otherwise. 4D tensors pass through.
static Tensor materialize_for_conv(const Tensor& t, c10::MemoryFormat desc_layout) {
  if (desc_layout == at::MemoryFormat::ChannelsLast3d) {
    return t.contiguous(at::MemoryFormat::ChannelsLast3d);
  }
  if (t.dim() == 5) {
    return t.contiguous();
  }
  return t;
}

// CL3d needs the NDArray path for explicit NDHWC ordering; NCDHW takes the
// tensor-direct Placeholder. Caller must materialize_for_conv first.
static at::native::mps::Placeholder make_conv_placeholder(MPSGraphTensor* graphTensor,
                                                          const at::Tensor& t,
                                                          c10::MemoryFormat desc_layout) {
  if (desc_layout == at::MemoryFormat::Contiguous) {
    return at::native::mps::Placeholder(graphTensor, t);
  }
  return at::native::mps::Placeholder(graphTensor,
                                      at::native::mps::getMPSNDArray(t, at::native::mps::getMPSShape(t, desc_layout)));
}

static void fill_depthwise_conv_desc(MPSGraphDepthwiseConvolution3DOpDescriptor* descriptor_,
                                     NSUInteger strideInX,
                                     NSUInteger strideInY,
                                     NSUInteger dilationRateInX,
                                     NSUInteger dilationRateInY,
                                     NSUInteger paddingHorizontal,
                                     NSUInteger paddingVertical) {
  descriptor_.strides =
      @[ @1, [[NSNumber alloc] initWithInteger:strideInY], [[NSNumber alloc] initWithInteger:strideInX] ];
  descriptor_.dilationRates =
      @[ @1, [[NSNumber alloc] initWithInteger:dilationRateInY], [[NSNumber alloc] initWithInteger:dilationRateInX] ];

  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;
  descriptor_.paddingValues = @[
    @0,
    @0,
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingHorizontal],
    [[NSNumber alloc] initWithInteger:paddingHorizontal]
  ];
  descriptor_.channelDimensionIndex = -3LL;
}

// Create convolution descriptor
static void fill_conv_desc(MPSGraphConvolution2DOpDescriptor* descriptor_,
                           NSUInteger strideInX,
                           NSUInteger strideInY,
                           NSUInteger dilationRateInX,
                           NSUInteger dilationRateInY,
                           NSUInteger paddingHorizontal,
                           NSUInteger paddingVertical,
                           c10::MemoryFormat memory_format,
                           NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::Contiguous) ? MPSGraphTensorNamedDataLayoutNCHW
                                                                           : MPSGraphTensorNamedDataLayoutNHWC;

  // PyTorch always uses OIHW memory layout for weights
  descriptor_.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
  descriptor_.groups = groups;
}

static Tensor _mps_convolution_impl(const Tensor& input_t,
                                    const Tensor& weight_t,
                                    const std::optional<Tensor>& bias_opt,
                                    IntArrayRef padding,
                                    IntArrayRef stride,
                                    IntArrayRef dilation,
                                    int64_t groups,
                                    std::optional<IntArrayRef> input_shape) {
  constexpr auto kChannelsLast = MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = MemoryFormat::Contiguous;
  const bool is_macos_15_plus = is_macos_at_least(MacOSVersion::MACOS_15_0);

  const bool is3DConv = input_t.dim() == 5;
  const auto memory_format = input_t.suggest_memory_format(/*channels_last_strides_exact_match=*/true);
  const bool is_cl_input =
      is_macos_15_plus && (is3DConv ? is_packed_channels_last_3d(input_t) : memory_format == kChannelsLast);
  const auto input_suggested_layout = is_cl_input ? (is3DConv ? kChannelsLast3d : kChannelsLast) : kContiguous;
  const bool use_dhwio = is3DConv && is_cl_input && conv3d_dhwio_is_beneficial(weight_t.sizes());
  // Allocate output in the user-requested layout regardless of fast-path gate.
  const bool is_channels_last = mps_conv_use_channels_last(input_t, weight_t);
  const bool bias_defined = bias_opt ? bias_opt->defined() : false;

  TORCH_CHECK(isFloatingType(input_t.scalar_type()), "Convolution is supported only for Floating types");

  using namespace at::native::mps;
  CheckedFrom c = "mps_convolution";
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto output_t =
      at::empty(input_shape.has_value() ? input_shape.value()
                                        : conv_output_size(input->sizes(), weight->sizes(), padding, stride, dilation),
                input->scalar_type(),
                std::nullopt,
                kMPS,
                std::nullopt,
                is_channels_last ? (is3DConv ? kChannelsLast3d : kChannelsLast) : kContiguous);
  if (output_t.numel() == 0) {
    return output_t;
  }
  TensorArg output{output_t, "result", 0};

  // TODO: Remove me when MacOS-14 is no longer supported
  std::optional<Tensor> output_c;
  if (!is_macos_15_plus && is_channels_last) {
    output_c = at::empty_like(output_t, output_t.options().memory_format(kContiguous));
  }

  if (!is_macos_at_least(MacOSVersion::MACOS_15_1)) {
    // On macOS < 15.1, MPS convolution kernel does not support output channels > 2^16
    for (auto elem : output_t.sizes()) {
      TORCH_CHECK_NOT_IMPLEMENTED(elem <= (1 << 16), "Output channels > 65536 not supported at the MPS device. ");
    }
  }

  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    IntArrayRef bias_shape;
    if (bias_defined)
      bias_shape = bias_opt.value().sizes();

    std::string bias_shape_key;
    if (bias_defined) {
      bias_shape_key = std::to_string(bias_shape[0]);
    } else {
      bias_shape_key = "nobias";
    }

    std::string key = fmt::format("mps_{}convolution:{}:{}:{}:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  is_cl_input,
                                  use_dhwio,
                                  mps::getTensorsStringKey({input_t, weight_t}),
                                  bias_defined,
                                  bias_shape_key);

    auto inputShape = mps::getMPSShape(input_t, input_suggested_layout);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      bool isDepthwiseConv =
          (groups > 1 && weight_t.size(1) == 1) && input_t.dim() >= 4 && weight_t.dim() >= 4 && !is_channels_last;

      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t), inputShape);
      auto weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);
      MPSGraphTensor* outputTensor = nil;
      if (is3DConv) {
        auto conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         input_suggested_layout,
                         use_dhwio,
                         groups);

        MPSGraphTensor* conv3dWeightTensor = use_dhwio
            ? [mpsGraph transposeTensor:weightTensor permutation:@[ @2, @3, @4, @1, @0 ] name:nil]
            : weightTensor;
        outputTensor = [mpsGraph convolution3DWithSourceTensor:inputTensor
                                                 weightsTensor:conv3dWeightTensor
                                                    descriptor:conv3dDescriptor_
                                                          name:nil];
      } else if (isDepthwiseConv) {
        auto depthWiseConv3dDescriptor_ = [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);

        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        outputTensor = [mpsGraph depthwiseConvolution3DWithSourceTensor:inputTensor
                                                          weightsTensor:weightTransposeTensor
                                                             descriptor:depthWiseConv3dDescriptor_
                                                                   name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       input_suggested_layout,
                       groups);

        outputTensor = [mpsGraph convolution2DWithSourceTensor:inputTensor
                                                 weightsTensor:weightTensor
                                                    descriptor:conv2dDescriptor_
                                                          name:nil];
      }

      MPSGraphTensor* biasTensor = nil;
      if (bias_defined) {
        biasTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(bias_opt.value()));
        outputTensor = [mpsGraph additionWithPrimaryTensor:outputTensor secondaryTensor:biasTensor name:nil];
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->biasTensor_ = biasTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    const auto input_for_graph =
        output_c ? input_t.contiguous() : materialize_for_conv(input_t, input_suggested_layout);
    auto inputPlaceholder = make_conv_placeholder(cachedGraph->inputTensor_, input_for_graph, input_suggested_layout);
    auto outputPlaceholder = output_c
        ? Placeholder(cachedGraph->outputTensor_, *output_c)
        : make_conv_placeholder(cachedGraph->outputTensor_, output_t, input_suggested_layout);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, output_c ? weight_t.contiguous() : weight_t);
    auto biasPlaceholder = Placeholder();
    // Reshape the bias to be broadcastable with output of conv2d or conv3d
    if (bias_defined) {
      const int64_t C = bias_shape[0];
      std::vector<int64_t> bias_view;
      if (is3DConv) {
        bias_view = input_suggested_layout == kChannelsLast3d ? std::vector<int64_t>{1, 1, 1, 1, C}
                                                              : std::vector<int64_t>{1, C, 1, 1, 1};
      } else {
        bias_view = input_suggested_layout == kChannelsLast ? std::vector<int64_t>{1, 1, 1, C}
                                                            : std::vector<int64_t>{1, C, 1, 1};
      }
      biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias_opt->view(bias_view));
    }

    auto feeds = [[[NSMutableDictionary alloc] initWithCapacity:3] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    if (bias_defined) {
      feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (output_c) {
    output_t.copy_(*output_c);
  }

  return output_t;
}

Tensor _mps_convolution(const Tensor& input_t,
                        const Tensor& weight_t,
                        const std::optional<Tensor>& bias_opt,
                        IntArrayRef padding,
                        IntArrayRef stride,
                        IntArrayRef dilation,
                        int64_t groups) {
  return _mps_convolution_impl(input_t, weight_t, bias_opt, padding, stride, dilation, groups, std::nullopt);
}

static Tensor mps_convolution_backward_input(IntArrayRef input_size,
                                             const Tensor& grad_output_t,
                                             const Tensor& weight_t,
                                             IntArrayRef padding,
                                             IntArrayRef stride,
                                             IntArrayRef dilation,
                                             int64_t groups,
                                             bool bias_defined) {
  using namespace at::native::mps;
  using namespace mps;
  bool is3DConv = grad_output_t.dim() == 5;
  if (!is_macos_at_least(MacOSVersion::MACOS_15_1)) {
    // On macOS < 15.1, MPS convolution kernel does not support output channels > 2^16
    for (auto elem : grad_output_t.sizes()) {
      TORCH_CHECK_NOT_IMPLEMENTED(elem <= (1 << 16), "Output channels > 65536 not supported at the MPS device. ");
    }
  }

  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_input";
  TensorArg grad_output{grad_output_t, "grad_output", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});
  constexpr auto kChannelsLast = at::MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = at::MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = at::MemoryFormat::Contiguous;
  const bool is_macos_15_plus = is_macos_at_least(MacOSVersion::MACOS_15_0);
  // Backward uses NDHWC+DHWIO only when the full fast path is beneficial; for
  // factorized kernels / small Cin / depthwise the NCDHW+OIDHW fallback wins.
  const bool use_dhwio = is3DConv && is_macos_15_plus && is_packed_channels_last_3d(grad_output_t) &&
      conv3d_dhwio_is_beneficial(weight_t.sizes());
  const auto desc_layout = use_dhwio ? kChannelsLast3d : kContiguous;
  // Allocate grad_input in the user-requested layout. The fast path writes
  // directly; the NCDHW fallback writes via a contig scratch + copy below.
  const bool is_channels_last = mps_conv_use_channels_last(grad_output_t, weight_t);
  auto grad_input_t =
      at::empty(input_size,
                grad_output_t.options(),
                is_channels_last ? std::optional(is3DConv ? kChannelsLast3d : kChannelsLast) : std::nullopt);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{grad_input_t, "result", 0};
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // Contig scratch when graph emits NCDHW but grad_input is CL3d -- covers
  // the macOS-14 fallback and the 3D NCDHW fallback on macOS 15+.
  std::optional<Tensor> grad_input_c;
  const bool needs_contig_scratch = is_channels_last && (!is_macos_15_plus || (is3DConv && !use_dhwio));
  if (needs_contig_scratch) {
    grad_input_c = at::empty_like(grad_input_t, grad_input_t.options().memory_format(MemoryFormat::Contiguous));
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
  };

  // Add backward with input
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();
    MPSShape* mps_input_shape = getMPSShape(input_size, desc_layout);
    std::string key = fmt::format("mps_{}_convolution_backward_input:{}:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  is_channels_last,
                                  use_dhwio,
                                  getTensorsStringKey({grad_output_t, weight_t}));
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto gradOutputShape = getMPSShape(grad_output_t, desc_layout);
      auto gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output_t), gradOutputShape);
      auto weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);

      MPSGraphTensor* gradInputTensor;
      MPSShape* weightOutputShape = mps::getMPSShape(weight_t);
      // Depthwise conv is input feature channels = groups. So I in OIHW has to be 1.
      bool isDepthwiseConv = ((groups > 1 && (weightOutputShape[1].intValue == 1)) && grad_output_t.ndimension() >= 4 &&
                              weightOutputShape.count >= 4 && !is_channels_last);

      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         desc_layout,
                         use_dhwio,
                         groups);
        MPSGraphTensor* convWeightTensor = use_dhwio
            ? [mpsGraph transposeTensor:weightTensor permutation:@[ @2, @3, @4, @1, @0 ] name:nil]
            : weightTensor;
        gradInputTensor = [mpsGraph convolution3DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                          weightsTensor:convWeightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv3dDescriptor_
                                                                                   name:nil];
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        gradInputTensor =
            [mpsGraph depthwiseConvolution3DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                     weightsTensor:weightTransposeTensor
                                                                       outputShape:mps_input_shape
                                                                        descriptor:depthWiseConv3dDescriptor_
                                                                              name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       at::MemoryFormat::Contiguous,
                       groups);

        gradInputTensor = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                          weightsTensor:weightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                   name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    const auto grad_out_for_graph =
        grad_input_c ? grad_output_t.contiguous() : materialize_for_conv(grad_output_t, desc_layout);
    auto gradOutputPlaceholder = make_conv_placeholder(cachedGraph->gradOutputTensor_, grad_out_for_graph, desc_layout);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, grad_input_c ? weight_t.contiguous() : weight_t);
    auto outputPlaceholder = grad_input_c
        ? Placeholder(cachedGraph->gradInputTensor_, *grad_input_c)
        : make_conv_placeholder(cachedGraph->gradInputTensor_, grad_input_t, desc_layout);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, weightsPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  if (grad_input_c) {
    grad_input_t.copy_(*grad_input_c);
  }
  return grad_input_t;
}

static Tensor mps_convolution_backward_weights(IntArrayRef weight_size,
                                               const Tensor& grad_output_t,
                                               const Tensor& input_t,
                                               IntArrayRef padding,
                                               IntArrayRef stride,
                                               IntArrayRef dilation,
                                               int64_t groups,
                                               bool bias_defined) {
  using namespace at::native::mps;
  using namespace mps;
  const bool is3DConv = input_t.dim() == 5;
  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_weights";
  constexpr auto kChannelsLast = at::MemoryFormat::ChannelsLast;
  constexpr auto kChannelsLast3d = at::MemoryFormat::ChannelsLast3d;
  constexpr auto kContiguous = at::MemoryFormat::Contiguous;
  const bool is_macos_15_plus = is_macos_at_least(MacOSVersion::MACOS_15_0);
  // Half-precision WG regresses on NDHWC+DHWIO; force NCDHW+OIDHW.
  const bool half_precision_wg =
      grad_output_t.scalar_type() == at::kBFloat16 || grad_output_t.scalar_type() == at::kHalf;
  // Require BOTH inputs CL3d-packed; otherwise we'd permute the non-packed one each call.
  const bool use_dhwio = is3DConv && is_macos_15_plus && !half_precision_wg && is_packed_channels_last_3d(input_t) &&
      is_packed_channels_last_3d(grad_output_t) && conv3d_dhwio_is_beneficial(weight_size);
  const auto desc_layout = use_dhwio ? kChannelsLast3d : kContiguous;
  // grad_weight allocation: 2D follows the standard CL convention; 3D always
  // stays contiguous OIDHW (the graph already transposes DHWIO -> OIDHW).
  const bool allocate_grad_weight_cl = mps_conv_use_channels_last(input_t, grad_output_t) && !is3DConv;

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_output{grad_output_t, "grad_output", 1};
  TensorArg input{input_t, "input", 2};

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t = at::empty(
      weight_size, grad_output_t.options(), allocate_grad_weight_cl ? std::optional(kChannelsLast) : std::nullopt);

  TensorArg grad_weight{grad_weight_t, "result", 0};

  convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gradWeightTensor_ = nil;
  };

  // TODO: Remove me when MacOS-14 is no longer supported
  std::optional<Tensor> grad_weight_c;
  if (!is_macos_at_least(MacOSVersion::MACOS_15_0) && allocate_grad_weight_cl) {
    grad_weight_c = at::empty_like(grad_weight_t, grad_weight_t.options().memory_format(MemoryFormat::Contiguous));
  }

  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();

    // Under DHWIO the graph emits weight grad in DHWIO order; the op output
    // shape must match, and we transpose back to OIDHW after.
    MPSShape* mps_weight_shape = use_dhwio
        ? @[ @(weight_size[2]), @(weight_size[3]), @(weight_size[4]), @(weight_size[1]), @(weight_size[0]) ]
        : getMPSShape(weight_size);
    std::string key = fmt::format("mps_{}convolution_backward_weights:{}:{}:{}:{}:{}:{}:{}",
                                  is3DConv ? "3d_" : "",
                                  getArrayRefString(stride),
                                  getArrayRefString(dilation),
                                  getArrayRefString(padding),
                                  groups,
                                  allocate_grad_weight_cl,
                                  use_dhwio,
                                  getTensorsStringKey({grad_output_t, input_t, grad_weight_t}));
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* inputShape = getMPSShape(input_t, desc_layout);
      MPSShape* gradOutputShape = getMPSShape(grad_output_t, desc_layout);
      // For the non-CL path the depthwise heuristic inspects the OIHW weight shape.
      MPSShape* weight_shape_OIDHW = getMPSShape(weight_size);
      bool isDepthwiseConv = ((groups > 1 && (weight_shape_OIDHW[1].intValue == 1)) && inputShape.count >= 4 &&
                              weight_shape_OIDHW.count >= 4);

      MPSGraphTensor* gradOutputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output_t), gradOutputShape);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t), inputShape);

      MPSGraphTensor* gradWeightTensor;
      if (is3DConv) {
        MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(conv3dDescriptor_,
                         stride[2],
                         stride[1],
                         stride[0],
                         dilation[2],
                         dilation[1],
                         dilation[0],
                         padding[2],
                         padding[1],
                         padding[0],
                         desc_layout,
                         use_dhwio,
                         groups);
        gradWeightTensor = [mpsGraph convolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv3dDescriptor_
                                                                                       name:nil];
        if (use_dhwio) {
          gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensor permutation:@[ @4, @3, @0, @1, @2 ] name:nil];
        }
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(
            depthWiseConv3dDescriptor_, stride[1], stride[0], dilation[1], dilation[0], padding[1], padding[0]);
        NSNumber* outputFeatChannelDim = mps_weight_shape[0];
        MPSShape* weightShapeTranspose = @[ @1, outputFeatChannelDim, mps_weight_shape[2], mps_weight_shape[3] ];
        MPSGraphTensor* gradWeightTensorTranspose =
            [mpsGraph depthwiseConvolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                         sourceTensor:inputTensor
                                                                          outputShape:weightShapeTranspose
                                                                           descriptor:depthWiseConv3dDescriptor_
                                                                                 name:nil];
        gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensorTranspose dimension:-3 withDimension:-4 name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       at::MemoryFormat::Contiguous,
                       groups);

        gradWeightTensor = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                       name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradWeightTensor_ = gradWeightTensor;
    });

    const auto grad_out_for_graph =
        grad_weight_c ? grad_output_t.contiguous() : materialize_for_conv(grad_output_t, desc_layout);
    const auto input_for_graph = grad_weight_c ? input_t.contiguous() : materialize_for_conv(input_t, desc_layout);
    auto gradOutputPlaceholder = make_conv_placeholder(cachedGraph->gradOutputTensor_, grad_out_for_graph, desc_layout);
    auto inputPlaceholder = make_conv_placeholder(cachedGraph->inputTensor_, input_for_graph, desc_layout);
    auto outputPlaceholder =
        Placeholder(cachedGraph->gradWeightTensor_, grad_weight_c ? *grad_weight_c : grad_weight_t);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (grad_weight_c) {
    grad_weight_t.copy_(*grad_weight_c);
  }
  return grad_weight_t;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mps_convolution_backward(const at::Tensor& input,
                                                                        const at::Tensor& grad_output,
                                                                        const at::Tensor& weight,
                                                                        IntArrayRef padding,
                                                                        IntArrayRef stride,
                                                                        IntArrayRef dilation,
                                                                        int64_t groups,
                                                                        std::array<bool, 3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    if (output_mask[0]) {
      grad_input = mps_convolution_backward_input(
          input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
    }
    if (output_mask[1]) {
      grad_weight = mps_convolution_backward_weights(
          weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

static Tensor mps_convolution_transpose_forward(const Tensor& grad_output,
                                                const Tensor& weight,
                                                IntArrayRef padding,
                                                IntArrayRef output_padding,
                                                IntArrayRef stride,
                                                IntArrayRef dilation,
                                                int64_t groups) {
  auto input_size =
      conv_input_size(grad_output.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
  return mps_convolution_backward_input(input_size, grad_output, weight, padding, stride, dilation, groups, false);
}

Tensor _mps_convolution_transpose(const Tensor& input_t,
                                  const Tensor& weight_t,
                                  IntArrayRef padding,
                                  IntArrayRef output_padding,
                                  IntArrayRef stride,
                                  IntArrayRef dilation,
                                  int64_t groups) {
  bool is_unsupported_3d_dtype =
      (input_t.dim() == 5 && (input_t.scalar_type() == kHalf || input_t.scalar_type() == kBFloat16));
  TORCH_CHECK(!is_unsupported_3d_dtype, "ConvTranspose 3D with BF16 or FP16 types is not supported on MPS");

  auto output_t =
      mps_convolution_transpose_forward(input_t, weight_t, padding, output_padding, stride, dilation, groups);
  return output_t;
}

static Tensor mps_convolution_transpose_backward_input(const Tensor& grad_output_t,
                                                       const Tensor& weight_t,
                                                       IntArrayRef padding,
                                                       IntArrayRef stride,
                                                       IntArrayRef dilation,
                                                       int64_t groups,
                                                       IntArrayRef input_shape) {
  return _mps_convolution_impl(grad_output_t, weight_t, std::nullopt, padding, stride, dilation, groups, input_shape);
}

static Tensor mps_convolution_transpose_backward_weight(IntArrayRef weight_size,
                                                        const Tensor& grad_output_t,
                                                        const Tensor& input_t,
                                                        IntArrayRef padding,
                                                        IntArrayRef stride,
                                                        IntArrayRef dilation,
                                                        int64_t groups) {
  return mps_convolution_backward_weights(
      weight_size, input_t, grad_output_t, padding, stride, dilation, groups, false);
}

std::tuple<Tensor, Tensor> mps_convolution_transpose_backward(const Tensor& input,
                                                              const Tensor& grad_output,
                                                              const Tensor& weight,
                                                              IntArrayRef padding,
                                                              IntArrayRef output_padding,
                                                              IntArrayRef stride,
                                                              IntArrayRef dilation,
                                                              int64_t groups,
                                                              std::array<bool, 2> output_mask) {
  Tensor grad_input, grad_weight;
  if (output_mask[0]) {
    grad_input =
        mps_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, input.sizes());
  }
  if (output_mask[1]) {
    grad_weight = mps_convolution_transpose_backward_weight(
        weight.sizes(), grad_output, input, padding, stride, dilation, groups);
  }

  return std::tuple<Tensor, Tensor>{grad_input, grad_weight};
}

} // namespace at::native
