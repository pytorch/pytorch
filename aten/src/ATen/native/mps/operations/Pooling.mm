//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/NamedTensorUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Pooling.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool2d_backward.h>
#include <ATen/ops/avg_pool2d_backward_native.h>
#include <ATen/ops/avg_pool2d_native.h>
#include <ATen/ops/avg_pool3d_native.h>
#include <ATen/ops/max_pool2d_backward_native.h>
#include <ATen/ops/max_pool2d_native.h>
#include <ATen/ops/max_pool2d_with_indices_backward_native.h>
#include <ATen/ops/max_pool2d_with_indices_native.h>
#include <ATen/ops/max_pool3d_with_indices_backward_native.h>
#include <ATen/ops/max_pool3d_with_indices_native.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Pooling_metallib.h>
#endif

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

static std::vector<int32_t> copy_and_maybe_expand(IntArrayRef a, int32_t pooling_dims) {
  std::vector<int32_t> b(pooling_dims);
  for (const auto dim : c10::irange(pooling_dims)) {
    b[dim] = safe_downcast<int32_t, int64_t>(a[a.size() == 1 ? 0 : dim]);
  }
  return b;
}

using PoolSizes = std::tuple<int32_t,
                             std::vector<int64_t>,
                             std::vector<int32_t>,
                             std::vector<int32_t>,
                             std::vector<int32_t>,
                             std::optional<std::vector<int32_t>>>;

static PoolSizes process_pool_sizes(const Tensor& input,
                                    IntArrayRef kernel_size,
                                    IntArrayRef stride,
                                    IntArrayRef padding,
                                    std::optional<IntArrayRef> dilation_opt,
                                    bool ceil_mode,
                                    const int32_t pooling_dims,
                                    const std::string& op_name) {
  TORCH_INTERNAL_ASSERT(pooling_dims == 1 || pooling_dims == 2 || pooling_dims == 3);

  const int32_t dims = input.dim();

  TORCH_CHECK(dims == pooling_dims + 1 || dims == pooling_dims + 2,
              op_name,
              ": non-empty ",
              pooling_dims + 1,
              "D or ",
              pooling_dims + 2,
              "D (batch mode) tensor expected for input");

  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == pooling_dims,
              op_name,
              ": kernel_size must either be a single int, or a tuple of ",
              pooling_dims,
              " ints");

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
              op_name,
              ": stride must either be omitted, a single int, or a tuple of ",
              pooling_dims,
              " ints");

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
              op_name,
              ": padding must either be a single int, or a tuple of ",
              pooling_dims,
              " ints");

  if (dilation_opt.has_value()) {
    auto dilation = dilation_opt.value();
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == pooling_dims,
                op_name,
                ": dilation must be either a single int, or a tuple of ",
                pooling_dims,
                " ints");
  }

  int32_t leading_dims = input.dim() - pooling_dims;

  const auto kernel_size_expanded = copy_and_maybe_expand(kernel_size, pooling_dims);
  const auto stride_expanded = copy_and_maybe_expand(stride.empty() ? kernel_size : stride, pooling_dims);
  const auto padding_expanded = copy_and_maybe_expand(padding, pooling_dims);
  const auto dilation_expanded = dilation_opt.has_value() ? copy_and_maybe_expand(dilation_opt.value(), pooling_dims)
                                                          : std::vector<int32_t>(pooling_dims, 1);

  for (const auto dim : c10::irange(pooling_dims)) {
    TORCH_CHECK(padding_expanded[dim] >= 0, op_name, ": pad must be non-negative");
    TORCH_CHECK(padding_expanded[dim] * 2 <= kernel_size_expanded[dim],
                op_name,
                ": pad should be at most half of effective kernel size");
  }

  for (const auto dim : c10::irange(static_cast<int>(leading_dims == 2), dims)) {
    TORCH_CHECK(input.size(dim) > 0, op_name, ": Expected input's non-batch dimensions to have positive length");
  }

  // According to the documentation, the output size of each pooling dimension
  // follows this basic formula:
  // (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

  std::vector<int64_t> output_pooling_size(pooling_dims);

  for (const auto dim : c10::irange(pooling_dims)) {
    int64_t out_size = (input.size(leading_dims + dim) + 2 * padding_expanded[dim] -
                        dilation_expanded[dim] * (kernel_size_expanded[dim] - 1)) -
        1;

    if (ceil_mode) {
      out_size += stride_expanded[dim] - 1;
    }

    out_size = out_size / stride_expanded[dim] + 1;

    if (ceil_mode) {
      if (((out_size - 1) * stride_expanded[dim]) >= (input.size(leading_dims + dim) + padding_expanded[dim])) {
        out_size -= 1;
      }
    }
    output_pooling_size[dim] = out_size;
  }

  std::vector<int64_t> output_size(dims);
  for (const auto dim : c10::irange(leading_dims)) {
    output_size[dim] = input.size(dim);
  }
  for (const auto dim : c10::irange(pooling_dims)) {
    output_size[leading_dims + dim] = output_pooling_size[dim];
  }

  return PoolSizes(dims,
                   output_size,
                   kernel_size_expanded,
                   stride_expanded,
                   padding_expanded,
                   dilation_opt.has_value() ? std::make_optional(dilation_expanded) : std::nullopt);
}

static void max_pool_with_indices_out_mps_template(const Tensor& output,
                                                   const std::optional<Tensor>& indices_opt,
                                                   const Tensor& input,
                                                   IntArrayRef _kernel_size,
                                                   IntArrayRef _stride,
                                                   IntArrayRef _padding,
                                                   IntArrayRef _dilation,
                                                   bool ceil_mode,
                                                   const int32_t pooling_dims,
                                                   const std::string& op_name) {
  auto [dims, output_size, kernel_size, stride, padding, dilation_opt] =
      process_pool_sizes(input, _kernel_size, _stride, _padding, _dilation, ceil_mode, pooling_dims, op_name);
  TORCH_INTERNAL_ASSERT(dilation_opt.has_value());
  auto dilation = dilation_opt.value();
  const Tensor& indices = *(at::borrow_from_optional_tensor(indices_opt));
  const bool return_indices = indices.defined();

  const auto memory_format = input.suggest_memory_format();
  output.resize_(output_size, memory_format);
  if (return_indices) {
    indices.resize_(output_size, memory_format);
  }

  auto iter = TensorIteratorConfig().add_output(output).resize_outputs(false).check_all_same_dtype(false).build();

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const auto numThreads = iter.numel();
  TORCH_INTERNAL_ASSERT(numThreads == output.numel());

  PoolingParams<5> params;

  params.dims = dims;
  params.pooling_dims = pooling_dims;
  params.return_indices = return_indices;

  for (const auto dim : c10::irange(dims)) {
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = safe_downcast<int32_t, int64_t>(input.stride(dim));
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = safe_downcast<int32_t, int64_t>(output.stride(dim));
    if (return_indices) {
      params.indices_sizes[dim] = safe_downcast<int32_t, int64_t>(indices.size(dim));
      params.indices_strides[dim] = safe_downcast<int32_t, int64_t>(indices.stride(dim));
    }
  }

  memcpy(params.kernel_size.data(), kernel_size.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.stride.data(), stride.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.padding.data(), padding.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.dilation.data(), dilation.data(), pooling_dims * sizeof(int32_t));

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto maxPoolPSO = lib.getPipelineStateForFunc("max_pool_" + scalarToMetalTypeString(input));

      getMPSProfiler().beginProfileKernel(maxPoolPSO, op_name, {input});
      [computeEncoder setComputePipelineState:maxPoolPSO];
      mtl_setArgs(
          computeEncoder, input, output, return_indices ? std::optional<Tensor>(indices) : std::nullopt, params);

      mtl_dispatch1DJob(computeEncoder, maxPoolPSO, numThreads);
      getMPSProfiler().endProfileKernel(maxPoolPSO);
    }
  });
}

static void max_pool_with_indices_backward_out_mps_template(Tensor& grad_input,
                                                            const Tensor& indices,
                                                            const Tensor& input,
                                                            const Tensor& grad_output,
                                                            IntArrayRef _kernel_size,
                                                            IntArrayRef _stride,
                                                            IntArrayRef _padding,
                                                            IntArrayRef _dilation,
                                                            bool ceil_mode,
                                                            const int32_t pooling_dims,
                                                            const std::string& op_name) {
  auto [dims, output_size, kernel_size, stride, padding, dilation_opt] =
      process_pool_sizes(input, _kernel_size, _stride, _padding, _dilation, ceil_mode, pooling_dims, op_name);

  const auto memory_format = input.suggest_memory_format();
  grad_input.resize_(input.sizes(), memory_format);
  grad_input.fill_(0);

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const auto numThreads = grad_output.numel();
  PoolingBackwardParams<5> params;

  params.dims = dims;
  params.pooling_dims = pooling_dims;

  for (const auto dim : c10::irange(dims)) {
    params.grad_input_sizes[dim] = safe_downcast<int32_t, int64_t>(grad_input.size(dim));
    params.grad_input_strides[dim] = safe_downcast<int32_t, int64_t>(grad_input.stride(dim));
    params.grad_output_sizes[dim] = safe_downcast<int32_t, int64_t>(grad_output.size(dim));
    params.grad_output_strides[dim] = safe_downcast<int32_t, int64_t>(grad_output.stride(dim));
    params.indices_strides[dim] = safe_downcast<int32_t, int64_t>(indices.stride(dim));
  }

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto maxPoolPSO = lib.getPipelineStateForFunc("max_pool_backward_" + scalarToMetalTypeString(input));

      getMPSProfiler().beginProfileKernel(maxPoolPSO, op_name, {input});
      [computeEncoder setComputePipelineState:maxPoolPSO];
      mtl_setArgs(computeEncoder, grad_input, grad_output, indices, params);

      mtl_dispatch1DJob(computeEncoder, maxPoolPSO, numThreads);
      getMPSProfiler().endProfileKernel(maxPoolPSO);
    }
  });
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

static void avg_pool_out_mps_template(const Tensor& output,
                                      const Tensor& input,
                                      IntArrayRef _kernel_size,
                                      IntArrayRef _stride,
                                      IntArrayRef _padding,
                                      bool ceil_mode,
                                      bool count_include_pad,
                                      std::optional<int64_t> divisor_override,
                                      const int32_t pooling_dims,
                                      const std::string& op_name) {
  auto [dims, output_size, kernel_size, stride, padding, _] =
      process_pool_sizes(input, _kernel_size, _stride, _padding, std::nullopt, ceil_mode, pooling_dims, op_name);

  const auto memory_format = input.suggest_memory_format();
  output.resize_(output_size, memory_format);

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const auto numThreads = output.numel();

  AvgPoolingParams<5> params;

  params.dims = dims;
  params.pooling_dims = pooling_dims;
  params.count_include_pad = count_include_pad;
  params.has_divisor_override = divisor_override.has_value();
  if (divisor_override.has_value()) {
    params.divisor_override = safe_downcast<int32_t, int64_t>(divisor_override.value());
  }

  for (const auto dim : c10::irange(dims)) {
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = safe_downcast<int32_t, int64_t>(input.stride(dim));
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = safe_downcast<int32_t, int64_t>(output.stride(dim));
  }

  memcpy(params.kernel_size.data(), kernel_size.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.stride.data(), stride.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.padding.data(), padding.data(), pooling_dims * sizeof(int32_t));

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto PSO = lib.getPipelineStateForFunc("avg_pool_" + scalarToMetalTypeString(input));

      getMPSProfiler().beginProfileKernel(PSO, op_name, {input});
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, input, output, params);

      mtl_dispatch1DJob(computeEncoder, PSO, numThreads);
      getMPSProfiler().endProfileKernel(PSO);
    }
  });
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

std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out_mps(const Tensor& input,
                                                             IntArrayRef kernel_size,
                                                             IntArrayRef stride,
                                                             IntArrayRef padding,
                                                             IntArrayRef dilation,
                                                             bool ceil_mode,
                                                             Tensor& output,
                                                             Tensor& indices) {
  mps::max_pool_with_indices_out_mps_template(output,
                                              indices,
                                              input,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              ceil_mode,
                                              /*pooling_dims=*/3,
                                              "max_pool3d");
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices_mps(const Tensor& input,
                                                       IntArrayRef kernel_size,
                                                       IntArrayRef stride,
                                                       IntArrayRef padding,
                                                       IntArrayRef dilation,
                                                       bool ceil_mode) {
  NoNamesGuard guard;

  Tensor output = at::empty({0}, input.options(), MemoryFormat::Contiguous);
  Tensor indices = at::empty({0}, input.options().dtype(kLong), MemoryFormat::Contiguous);
  mps::max_pool_with_indices_out_mps_template(output,
                                              indices,
                                              input,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              ceil_mode,
                                              /*pooling_dims=*/3,
                                              "max_pool3d");

  guard.reset();
  namedinference::propagate_names(output, input);
  namedinference::propagate_names(indices, input);

  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& max_pool3d_with_indices_backward_out_mps(const Tensor& grad_output,
                                                 const Tensor& input,
                                                 IntArrayRef kernel_size,
                                                 IntArrayRef stride,
                                                 IntArrayRef padding,
                                                 IntArrayRef dilation,
                                                 bool ceil_mode,
                                                 const Tensor& indices,
                                                 Tensor& grad_input) {
  mps::max_pool_with_indices_backward_out_mps_template(grad_input,
                                                       indices,
                                                       input,
                                                       grad_output,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       dilation,
                                                       ceil_mode,
                                                       /*pooling_dims=*/3,
                                                       "max_pool3d_backward");
  return grad_input;
}

Tensor max_pool3d_with_indices_backward_mps(const Tensor& grad_output,
                                            const Tensor& input,
                                            IntArrayRef kernel_size,
                                            IntArrayRef stride,
                                            IntArrayRef padding,
                                            IntArrayRef dilation,
                                            bool ceil_mode,
                                            const Tensor& indices) {
  auto grad_input = at::empty({0}, input.options());
  mps::max_pool_with_indices_backward_out_mps_template(grad_input,
                                                       indices,
                                                       input,
                                                       grad_output,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       dilation,
                                                       ceil_mode,
                                                       /*pooling_dims=*/3,
                                                       "max_pool3d_backward");
  return grad_input;
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

TORCH_IMPL_FUNC(avg_pool3d_out_mps)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  mps::avg_pool_out_mps_template(output,
                                 input,
                                 kernel_size,
                                 stride,
                                 padding,
                                 ceil_mode,
                                 count_include_pad,
                                 divisor_override,
                                 /*pooling_dims=*/3,
                                 "avg_pool3d");
}

} // namespace at::native
