//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#endif

namespace at::native {

static void get_shapes(MPSShape* input_shape_readonly,
                       NSMutableArray<NSNumber*>*& input_shape,
                       int num_input_dims,
                       c10::MemoryFormat memory_format) {
  // Modify the shape
  if (memory_format == at::MemoryFormat::Contiguous) {
    for (int i = 0; i < num_input_dims; i++)
      input_shape[i] = input_shape_readonly[i];
  } else { // ChannelsLast
    auto num_channels = input_shape_readonly[1];
    input_shape[0] = input_shape_readonly[0];
    for (int i = 1; i < num_input_dims - 1; i++)
      input_shape[i] = input_shape_readonly[i + 1];
    input_shape[num_input_dims - 1] = num_channels;
  }
}

// Note - Currently only supported for 4D image tensors

TORCH_IMPL_FUNC(softmax_mps_out)
(const Tensor& input_, const int64_t dim, const bool half_to_float, const Tensor& output) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on MPS");
  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);

  if (input_.numel() == 0) {
    return;
  }

  Tensor input;
  if (input_.dim() == 0) {
    input = input_.view(1);
  } else
    input = input_;

  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < input.dim(), "Softmax:dim must be non-negative and less than input dimensions");

  const auto memory_format = input.suggest_memory_format();
  // TORCH_CHECK(input.suggest_memory_format() == output.suggest_memory_format(), "Input and output memory format should
  // match")

  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string mem_format_key = get_mem_format_string(memory_format);
    MPSShape* input_shape_readonly = mps::getMPSShape(input);
    int num_input_dims = [input_shape_readonly count];
    // Check - Channels last implies 4d
    TORCH_CHECK(memory_format != at::MemoryFormat::ChannelsLast || num_input_dims == 4,
                "ChannelsLast implies 4d tensor")
    // Input shape changes based on memory format
    NSMutableArray<NSNumber*>* input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

    get_shapes(input_shape_readonly, input_shape, num_input_dims, memory_format);

    // Change dim
    if (memory_format == at::MemoryFormat::ChannelsLast && dim_ > 0 && !is_macOS_15_0_or_newer) {
      switch (dim_) {
        case 1:
          dim_ = 3;
          break;
        case 2:
          dim_ = 1;
          break;
        case 3:
          dim_ = 2;
          break;
        default:
          assert(0 && "Invalid dim\n");
      }
    }

    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    string key = "softmax_mps_out" + getTensorsStringKey(input, true, /*exclude_shape*/ true) + ":" + mem_format_key +
        ":" + std::to_string(dim_);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()));

      // passing selector of softMaxWithTensor on the mpsGraph object
      MPSGraphTensor* outputTensor = [mpsGraph softMaxWithTensor:inputTensor axis:(NSInteger)dim_ name:nil];

      // Output needs to be contiguous format
      if (memory_format == at::MemoryFormat::ChannelsLast && !is_macOS_15_0_or_newer) {
        auto N = input_shape[0];
        auto H = input_shape[1];
        auto W = input_shape[2];
        auto C = input_shape[3];

        outputTensor = [mpsGraph reshapeTensor:outputTensor
                                     withShape:@[ N, ([NSNumber numberWithInt:[H intValue] * [W intValue]]), C ]
                                          name:nil];
        outputTensor = [mpsGraph transposeTensor:outputTensor dimension:1 withDimension:2 name:nil];
        outputTensor = [mpsGraph reshapeTensor:outputTensor withShape:@[ N, C, H, W ] name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder =
        Placeholder(cachedGraph->inputTensor_, input, is_macOS_15_0_or_newer ? nil : input_shape);
    // This must be the Contiguous shape
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(softmax_backward_mps_out)
(const Tensor& grad_, const Tensor& output_, int64_t dim, ScalarType input_dtype, const Tensor& grad_input) {
  if (output_.numel() == 0) {
    return;
  }

  Tensor grad;
  if (grad_.dim() == 0) {
    grad = grad_.view(1);
  } else
    grad = grad_;

  Tensor output;
  if (output_.dim() == 0) {
    output = output_.view(1);
  } else
    output = output_;

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < grad.dim(), "Grad:dim must be non-negative and less than input dimensions");

  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* grad_shape = mps::getMPSShape(grad);
    NSString* ns_shape_key = [[grad_shape valueForKey:@"description"] componentsJoinedByString:@","];

    string key = "softmax_backward_mps_out:" + getMPSTypeString(output) + ":" + [ns_shape_key UTF8String] + ":" +
        std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* softmaxTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(output), grad_shape);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad), grad_shape);

      MPSGraphTensor* mulTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                            secondaryTensor:gradOutputTensor
                                                                       name:nil];
      MPSGraphTensor* mulSumTensor = [mpsGraph reductionSumWithTensor:mulTensor axis:(NSInteger)dim_ name:nil];
      MPSGraphTensor* gradSubTensor = [mpsGraph subtractionWithPrimaryTensor:gradOutputTensor
                                                             secondaryTensor:mulSumTensor
                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                                  secondaryTensor:gradSubTensor
                                                                             name:nil];

      newCachedGraph->outputTensor_ = softmaxTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder softmaxPlaceholder = Placeholder(cachedGraph->outputTensor_, output, grad_shape);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad, grad_shape);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(softmaxPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

} // namespace at::native
