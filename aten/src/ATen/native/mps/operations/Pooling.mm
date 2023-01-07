//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/TensorUtils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

namespace at {
namespace native {

// Create pooling descriptor
void fill_pool_desc(MPSGraphPooling2DOpDescriptor* desc,
                    NSUInteger kW, NSUInteger kH,
                    NSUInteger dW, NSUInteger dH,
                    NSUInteger dilationW, NSUInteger dilationH,
                    NSUInteger padW, NSUInteger padH,
                    bool ceil_mode, c10::MemoryFormat memory_format) {
  desc.kernelWidth = kW;
  desc.kernelHeight = kH;
  desc.strideInX = dW;
  desc.strideInY = dH;
  desc.dilationRateInX = dilationW;
  desc.dilationRateInY = dilationH;
  desc.paddingLeft = padW;
  desc.paddingRight = padW;
  desc.paddingTop = padH;
  desc.paddingBottom = padH;
  desc.ceilMode = ceil_mode;
  desc.paddingStyle = MPSGraphPaddingStyleExplicit;
  switch(memory_format) {
    case at::MemoryFormat::Contiguous:
      desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
      break;
    case at::MemoryFormat::ChannelsLast:
      desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
      break;
    default:
      assert(0 && "Check should have been done earlier\n");
  }
}

Tensor _mps_max_pool2d(
    const Tensor& input_t,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input_t.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_t.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((input_t.ndimension() == 3 || input_t.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nbatch = input_t.ndimension() == 4 ? input_t.size(-4) : 1;
  const int64_t nInputPlane = input_t.size(-3);
  const int64_t inputHeight = input_t.size(-2);
  const int64_t inputWidth = input_t.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(
    input_t,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth, memory_format);

  namespace native_mps = at::native::mps;
  using CachedGraph = native_mps::MPSUnaryCachedGraph;

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  Tensor output_t;

  if (input_t.ndimension() == 3) {
    output_t = at::native::empty_mps(
                  {nInputPlane, outputHeight, outputWidth},
                  input_t.scalar_type(),
                  c10::nullopt,
                  kMPS,
                  c10::nullopt,
                  memory_format);
  } else {
    output_t = at::native::empty_mps(
                  {nbatch, nInputPlane, outputHeight, outputWidth},
                  input_t.scalar_type(),
                  c10::nullopt,
                  kMPS,
                  c10::nullopt,
                  memory_format);
  }

  if (output_t.numel() == 0) {
    return output_t;
  }

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    string key = "mps_max_pool2d:" + to_string(kW) + ":" + to_string(kH) + ":" +
                                     to_string(dW) + ":" + to_string(dH) + ":" +
                                     to_string(dilationW) + ":" + to_string(dilationH) + ":" +
                                     to_string(padW) + ":" + to_string(padH) + ":" +
                                     to_string(ceil_mode) + ":" + mem_format_key +
                                     mps::getTensorsStringKey({input_t});
    CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor new] autorelease];
          fill_pool_desc(desc, kW, kH, dW, dH, dilationW, dilationH, padW, padH, ceil_mode, memory_format);

          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input_t);
          MPSGraphTensor* outputTensor = [mpsGraph maxPooling2DWithSourceTensor:inputTensor
                                                                     descriptor:desc
                                                                           name:nil];
          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = native_mps::Placeholder(cachedGraph->outputTensor_, output_t);

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output_t;
}

Tensor mps_max_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& input_t,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input_t.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_t.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((input_t.ndimension() == 3 || input_t.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }

  namespace native_mps = at::native::mps;

  // Derive from MPSCachedGraph
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *gradOutputTensor_ = nil;
    MPSGraphTensor *gradInputTensor_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  Tensor grad_input;
  grad_input = at::native::empty_mps(
                input_t.sizes(),
                input_t.scalar_type(),
                c10::nullopt,
                kMPS,
                c10::nullopt,
                memory_format);

  if (grad_input.numel() == 0) {
    return grad_input;
  }

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    string key = "mps_max_pool2d_backward:" + to_string(kW) + ":" + to_string(kH) + ":" +
                                              to_string(dW) + ":" + to_string(dH) + ":" +
                                              to_string(dilationW) + ":" + to_string(dilationH) + ":" +
                                              to_string(padW) + ":" + to_string(padH) + ":" +
                                              to_string(ceil_mode) + ":" + mem_format_key +
                                              mps::getTensorsStringKey({input_t, grad_output});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor new] autorelease];
          fill_pool_desc(desc, kW, kH, dW, dH, dilationW, dilationH, padW, padH, ceil_mode, memory_format);

          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input_t);
          MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
          MPSGraphTensor* gradInputTensor = [mpsGraph maxPooling2DGradientWithGradientTensor:gradOutputTensor
                                                                                sourceTensor:inputTensor
                                                                                  descriptor:desc
                                                                                        name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->gradInputTensor_ = gradInputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, input_t);
    auto gradOutputPlaceholder = native_mps::Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    auto gradInputPlaceholder = native_mps::Placeholder(cachedGraph->gradInputTensor_, grad_input);

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      gradInputPlaceholder.getMPSGraphTensor() : gradInputPlaceholder.getMPSGraphTensorData()
    };

    native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return grad_input;
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_out_mps)(
    const Tensor& input_t,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& output_t,
    const Tensor& indices) {

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input_t.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_t.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((input_t.ndimension() == 3 || input_t.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nInputPlane = input_t.size(-3);
  const int64_t inputHeight = input_t.size(-2);
  const int64_t inputWidth = input_t.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(
    input_t,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth, memory_format);

  namespace native_mps = at::native::mps;

  // Derive from MPSCachedGraph
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* indicesTensor_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  if (output_t.numel() == 0) {
    return;
  }

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    string key = "max_pool2d_with_indices_out_mps:" + to_string(kW) + ":" + to_string(kH) + ":" +
                                                      to_string(dW) + ":" + to_string(dH) + ":" +
                                                      to_string(dilationW) + ":" + to_string(dilationH) + ":" +
                                                      to_string(padW) + ":" + to_string(padH) + ":" +
                                                      to_string(ceil_mode) + ":" + mem_format_key +
                                                      mps::getTensorsStringKey({input_t}) + ":" +
                                                      native_mps::getMPSTypeString(indices.scalar_type());
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor new] autorelease];
          fill_pool_desc(desc, kW, kH, dW, dH, dilationW, dilationH, padW, padH, ceil_mode, memory_format);
          desc.returnIndicesMode = MPSGraphPoolingReturnIndicesGlobalFlatten2D;
          desc.returnIndicesDataType = MPSDataTypeInt32;

          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input_t);
          NSArray<MPSGraphTensor*>* poolOutputs = [mpsGraph maxPooling2DReturnIndicesWithSourceTensor:inputTensor
                                                                                           descriptor:desc
                                                                                                 name:nil];

            MPSGraphTensor* indicesTensor = poolOutputs[1];
            if(mps::getMPSDataType(indices.scalar_type()) == MPSDataTypeInt64) {
                indicesTensor = [mpsGraph castTensor:indicesTensor
                                               toType:MPSDataTypeInt64
                                                 name:@"castToI64"];
            }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = poolOutputs[0];
          newCachedGraph->indicesTensor_ = indicesTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = native_mps::Placeholder(cachedGraph->outputTensor_, output_t);
    auto indicesPlaceholder = native_mps::Placeholder(cachedGraph->indicesTensor_, indices);

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData(),
      indicesPlaceholder.getMPSGraphTensor() : indicesPlaceholder.getMPSGraphTensorData()
    };

    native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

}

TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_mps)
(const Tensor& grad_output,
const Tensor& input_t,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode,
const Tensor& indices,
const Tensor& grad_input) {

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input_t.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_t.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((input_t.ndimension() == 3 || input_t.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }

  namespace native_mps = at::native::mps;

  // Derive from MPSCachedGraph
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *gradOutputTensor_ = nil;
    MPSGraphTensor *gradInputTensor_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  if (grad_input.numel() == 0) {
    return;
  }

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    string key = "max_pool2d_with_indices_backward_out_mps:" + to_string(kW) + ":" + to_string(kH) + ":" +
                                               to_string(dW) + ":" + to_string(dH) + ":" +
                                               to_string(dilationW) + ":" + to_string(dilationH) + ":" +
                                               to_string(padW) + ":" + to_string(padH) + ":" +
                                               to_string(ceil_mode) + ":" + mem_format_key +
                                               mps::getTensorsStringKey({input_t, grad_output});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor new] autorelease];
          fill_pool_desc(desc, kW, kH, dW, dH, dilationW, dilationH, padW, padH, ceil_mode, memory_format);

          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input_t);
          MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
          MPSGraphTensor* gradInputTensor = [mpsGraph maxPooling2DGradientWithGradientTensor:gradOutputTensor
                                                                                sourceTensor:inputTensor
                                                                                  descriptor:desc
                                                                                        name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->gradInputTensor_ = gradInputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, input_t);
    auto gradOutputPlaceholder = native_mps::Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    auto gradInputPlaceholder = native_mps::Placeholder(cachedGraph->gradInputTensor_, grad_input);

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      gradInputPlaceholder.getMPSGraphTensor() : gradInputPlaceholder.getMPSGraphTensorData()
    };

    native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

TORCH_IMPL_FUNC(avg_pool2d_out_mps) (
   const Tensor& input_,
   int64_t kH_,
   int64_t kW_,
   int64_t dH_,
   int64_t dW_,
   int64_t padH_,
   int64_t padW_,
   bool ceil_mode,
   bool count_include_pad,
   c10::optional<int64_t> divisor_override,
   const Tensor& output) {
  namespace native_mps = at::native::mps;

  TensorArg output_arg{ output, "output", 1 };
  TensorArg input_arg{ input_, "input_", 2 };

  checkAllSameGPU("avg_pool2d_out_cuda", {output_arg, input_arg});

  const int kH = safe_downcast<int, int64_t>(kH_);
  const int kW = safe_downcast<int, int64_t>(kW_);

  const int dH = safe_downcast<int, int64_t>(dH_);
  const int dW = safe_downcast<int, int64_t>(dW_);

  const int padH = safe_downcast<int, int64_t>(padH_);
  const int padW = safe_downcast<int, int64_t>(padW_);

  /* sizes */

  const auto memory_format = input_.suggest_memory_format();

  Tensor input = input_.contiguous(memory_format);

  const int32_t count = safe_downcast<int32_t, int64_t>(output.numel());

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value = use_divisor ? divisor_override.value() : 0;

  if (count != 0) {
    // Derive from MPSCachedGraph
    struct CachedGraph : public native_mps::MPSCachedGraph
    {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor* inputTensor_ = nil;
      MPSGraphTensor* outputTensor_ = nil;
      MPSGraphTensor* indicesTensor_ = nil;
    };

    native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

    auto stream = at::mps::getCurrentMPSStream();

    @autoreleasepool {
      string mem_format_key;
      switch(memory_format) {
        case at::MemoryFormat::Contiguous:
          mem_format_key = "Contiguous";
          break;
        case at::MemoryFormat::ChannelsLast:
          mem_format_key = "ChannelsLast";
          break;
        default:
          assert(0 && "Check should have been done earlier\n");
      }

      string key = "mps_avg_pool2d:" + to_string(kW) + ":" + to_string(kH) + ":" +
                                       to_string(dW) + ":" + to_string(dH) + ":" +
                                       to_string(padW) + ":" + to_string(padH) + ":" +
                                       to_string(ceil_mode) + ":" + mem_format_key + ":" +
                                       to_string(divisor_override_value) +
                                       mps::getTensorsStringKey({input});
      CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);

      if(!cachedGraph) {
        native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {
          CachedGraph *newCachedGraph = nil;

          @autoreleasepool {
            MPSGraph* mpsGraph = native_mps::make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);

            MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor new] autorelease];
            fill_pool_desc(desc, kW, kH, dW, dH, 1, 1, padW, padH, ceil_mode, memory_format);

            MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input);
            MPSGraphTensor* outputTensor = [mpsGraph avgPooling2DWithSourceTensor:inputTensor
                                                                       descriptor:desc
                                                                             name:nil];
            newCachedGraph->inputTensor_ = inputTensor;
            newCachedGraph->outputTensor_ = outputTensor;
          }
          return newCachedGraph;
        });
        cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
      }

      auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, input);
      auto outputPlaceholder = native_mps::Placeholder(cachedGraph->outputTensor_, output);

      NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
        inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      };

      NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
        outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
      };

      native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }
  }
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_mps) (
  const Tensor& gradOutput_,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override,
  const Tensor& gradInput
) {
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameGPU("avg_pool2d_backward_out_cuda",
                  {gradInput_arg, gradOutput_arg, input_arg});
  namespace native_mps = at::native::mps;

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const auto memory_format = input_.suggest_memory_format();
  const Tensor input = input_.contiguous(memory_format);
  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);


  const int32_t count = safe_downcast<int32_t, int64_t>(input.numel());
  if (count == 0) {
    return;
  }

  namespace native_mps = at::native::mps;

  // Derive from MPSCachedGraph
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *gradOutputTensor_ = nil;
    MPSGraphTensor *gradInputTensor_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  if (gradInput.numel() == 0) {
    return;
  }

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    string key = "avg_pool2d_backward_out_mps:" + to_string(kW) + ":" + to_string(kH) + ":" +
                                               to_string(dW) + ":" + to_string(dH) + ":" +
                                               to_string(outputWidth) + ":" + to_string(outputHeight) + ":" +
                                               to_string(padW) + ":" + to_string(padH) + ":" +
                                               to_string(ceil_mode) + ":" + mem_format_key +
                                               mps::getTensorsStringKey({input, gradOutput});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor new] autorelease];
          fill_pool_desc(desc, kW, kH, dW, dH, 1, 1, padW, padH, ceil_mode, memory_format);

          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input);
          MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, gradOutput);
          MPSGraphTensor *gradInputTensor = [mpsGraph avgPooling2DGradientWithGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                descriptor : desc
                                                                                       name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->gradInputTensor_ = gradInputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, input);
    auto gradOutputPlaceholder = native_mps::Placeholder(cachedGraph->gradOutputTensor_, gradOutput);
    auto gradInputPlaceholder = native_mps::Placeholder(cachedGraph->gradInputTensor_, gradInput);

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      gradInputPlaceholder.getMPSGraphTensor() : gradInputPlaceholder.getMPSGraphTensorData()
    };

    native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

} // namespace native
} // namespace at
