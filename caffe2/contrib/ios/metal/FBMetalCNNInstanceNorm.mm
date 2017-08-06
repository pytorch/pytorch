// Copyright 2004-present Facebook. All Rights Reserved.

#import "FBMetalCNNInstanceNorm.h"
#import "MetalShaderUtilities.h"


std::string FBMetalInstanceNormConstantValues::to_string() {
  std::ostringstream ss;
  ss << ":" << input_width << ":" << input_height << ":" << input_channels << ":" << prelu_size;
  return ss.str();
}

@interface FBMetalCNNInstanceNormBase () {
  struct {
    ushort channel;
    ushort width;
    ushort height;
  } configuration;
}
@end

namespace {
  extern const char *metalCode;
}

@implementation FBMetalCNNInstanceNormBase

static constexpr size_t kThreadGroupSize_x = 4;
static constexpr size_t kThreadGroupSize_y = 8;

static NSArray* kernelNames = @[
                                @"cnn_instance_norm_avg_kern",
                                @"cnn_instance_norm_stdev_kern",
                                @"cnn_instance_norm_last_kern",
                                @"cnn_instance_norm_prelu_kern"
                                ];

+ (instancetype)filterWithContext:(MetalContext*)context
                     functionName:(NSString*)functionName
                   constantValues:(FBMetalInstanceNormConstantValues*)constantValues
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                          channel:(NSUInteger)channel {
  return [[self alloc] initWithContext:context
                          functionName:functionName
                        constantValues:constantValues
                                 width:width
                                height:height
                               channel:channel];
}

- (instancetype)initWithContext:(MetalContext*)context
                   functionName:(NSString*)functionName
                 constantValues:(FBMetalInstanceNormConstantValues*)constantValues
                          width:(NSUInteger)width
                         height:(NSUInteger)height
                        channel:(NSUInteger)channel {
  if ((self = [super initWithFunctionName:functionName
                              libraryName:@"InstanceNorm"
                            librarySource:[NSString stringWithCString:metalCode encoding:NSUTF8StringEncoding]
                                  context:context
                           constantValues:constantValues])) {
    _functionName = functionName;

    configuration.channel = channel;
    configuration.width   = width;
    configuration.height  = height;

    super.outputTextureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                       width:width
                                                                                      height:height
                                                                                   mipmapped:NO];
  }
  return self;
}

- (MTLSize)threadsPerThreadgroup {
  if (_functionName == kernelNames[0] || _functionName == kernelNames[1]) {
    NSUInteger maxTotalThreadsPerThreadgroup = [self.pipeline maxTotalThreadsPerThreadgroup];
    return MTLSizeMake(maxTotalThreadsPerThreadgroup, 1, 1);
  } else {
    return MTLSizeMake(kThreadGroupSize_x, kThreadGroupSize_y, 1);
  }
}

- (MTLSize)threadgroupsPerGrid {
  if (_functionName == kernelNames[0] || _functionName == kernelNames[1]) {
    return MTLSizeMake(configuration.channel, 1, 1);
  } else {
    MTLSize threadsPerThreadgroup = [self threadsPerThreadgroup];
    return MTLSizeMake(
        (self.outputTextureDescriptor.width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
        (self.outputTextureDescriptor.height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
        1);
  }
}

- (NSString*)replaceConstantValues:(FBMetalConstantValues*)constantValues librarySource:(NSString*)librarySource {
  FBMetalInstanceNormConstantValues* convolutionConstantValues = (FBMetalInstanceNormConstantValues*)constantValues;
  std::string source                                           = [librarySource UTF8String];

  REPLACE_CONSTANT(source, convolutionConstantValues->input_width * convolutionConstantValues->input_height, 0);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_width, 1);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_height, 2);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_channels, 3);
  REPLACE_CONSTANT(source, convolutionConstantValues->prelu_size, 4);

  return [NSString stringWithUTF8String:source.c_str()];
}

- (void)configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
                              weightBufferOffset:(NSInteger)weightBufferOffset
                              outputBufferOffset:(NSInteger)outputBufferOffset {
  [commandEncoder setBuffer:_scaleBuffer offset:0 atIndex:0];
  [commandEncoder setBuffer:_biasBuffer offset:0 atIndex:1];
  [commandEncoder setBuffer:_dataBuffer offset:0 atIndex:2];
  [commandEncoder setBuffer:_outputBuffer offset:0 atIndex:3];
  [commandEncoder setBuffer:_avgBuffer offset:0 atIndex:4];
  [commandEncoder setBuffer:_stdevBuffer offset:0 atIndex:5];
  [commandEncoder setBuffer:_epsilonBuffer offset:0 atIndex:6];
  if (_preluBuffer != nil) {
    [commandEncoder setBuffer:_preluBuffer offset:0 atIndex:7];
  }

  if (_functionName == kernelNames[0] || _functionName == kernelNames[1]) {
    MTLSize threads                   = [self threadsPerThreadgroup];
    const int threadGroupMemoryLength = threads.width * sizeof(float);
    [commandEncoder setThreadgroupMemoryLength:threadGroupMemoryLength atIndex:0];
  }
}
@end

@implementation FBMetalCNNInstanceNorm {
  NSMutableArray<FBMetalCNNInstanceNormBase*>* instanceNorm;
  NSMutableArray<NSString*>* functionNames;
}

+ (instancetype)filterWithContext:(MetalContext*)context
                   constantValues:(FBMetalInstanceNormConstantValues*)constantValues
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                          channel:(NSUInteger)channel
                        withPRelu:(BOOL)withPRelu {
  return [[self alloc] initWithContext:context
                        constantValues:constantValues
                                 width:width
                                height:height
                               channel:channel
                             withPRelu:withPRelu];
}

- (instancetype)initWithContext:(MetalContext*)context
                 constantValues:(FBMetalInstanceNormConstantValues*)constantValues
                          width:(NSUInteger)width
                         height:(NSUInteger)height
                        channel:(NSUInteger)channel
                      withPRelu:(BOOL)withPRelu{
  self.context = context;
  if (withPRelu) {
    functionNames = [[NSMutableArray alloc]  initWithObjects:kernelNames[0], kernelNames[1], kernelNames[3], nil];
  } else {
    functionNames = [[NSMutableArray alloc]  initWithObjects:kernelNames[0], kernelNames[1], kernelNames[2], nil];
  }
  instanceNorm = [NSMutableArray<FBMetalCNNInstanceNormBase*> arrayWithCapacity:functionNames.count];
  for (int i = 0; i < functionNames.count; i++) {
    FBMetalCNNInstanceNormBase* instanceNormChunk = [FBMetalCNNInstanceNormBase filterWithContext:context
                                                                                     functionName:functionNames[i]
                                                                                   constantValues:constantValues
                                                                                            width:width
                                                                                           height:height
                                                                                          channel:channel];

    instanceNorm[i] = instanceNormChunk;
  }
  return self;
}

- (void)loadEpsilon:(const float)epsilon {
  int length = 1;
  if (self.epsilonBuffer == nil || [self.epsilonBuffer length] < sizeof(float) * length) {
    self.epsilonBuffer =
    [self.context.device newBufferWithLength:sizeof(float) * length options:MTLResourceOptionCPUCacheModeDefault];
  }
  if (self.epsilonBuffer) {
    float* bias_data = (float*)[self.epsilonBuffer contents];
    bias_data[0]     = epsilon;
  }
}

- (void)applyFilter:(void (^)(NSError*))completionHandler {
  for (int i = 0; i < functionNames.count; i++) {
    FBMetalCNNInstanceNormBase* instanceNormChunk = instanceNorm[i];
    instanceNormChunk.avgBuffer                   = self.avgBuffer;
    instanceNormChunk.stdevBuffer                 = self.stdevBuffer;
    instanceNormChunk.dataBuffer                  = self.dataBuffer;
    instanceNormChunk.outputBuffer                = self.outputBuffer;
    instanceNormChunk.scaleBuffer                 = self.scaleBuffer;
    instanceNormChunk.biasBuffer                  = self.biasBuffer;
    instanceNormChunk.preluBuffer                 = self.preluBuffer;
    instanceNormChunk.epsilonBuffer               = self.epsilonBuffer;

    [instanceNorm[i] applyFilter:completionHandler];
  }
}

@end

namespace {
const char *metalCode = R"Metal(
//  Copyright 2004-present Facebook. All Rights Reserved.

#include <metal_stdlib>

using namespace metal;

constant constexpr int input_size                     [[ function_constant(0) ]];
constant constexpr ushort input_size_x                [[ function_constant(1) ]];
constant constexpr ushort input_size_y                [[ function_constant(2) ]];
constant constexpr ushort channels                    [[ function_constant(3) ]];
constant constexpr ushort prelu_size                  [[ function_constant(4) ]];

typedef float data_buffer_type;

typedef struct {
  data_buffer_type data[channels * input_size];
} input_output_data;

typedef struct {
  half data[channels];
} stats_data;

typedef struct {
  half data[channels];
} channel_data;

typedef struct {
  float data[1];
} epsilon_data;

kernel void cnn_instance_norm_avg_kern(
                                       constant input_output_data &inputBuffer    [[ buffer(2) ]],
                                       device stats_data &avgBuffer               [[ buffer(4) ]],
                                       threadgroup float *per_thread_sum          [[ threadgroup(0) ]],
                                       ushort channel                             [[ threadgroup_position_in_grid ]],
                                       ushort num_threads                         [[ threads_per_threadgroup ]],
                                       ushort tid                                 [[ thread_index_in_threadgroup ]]
                                       ) {
  const int chunk_size = (input_size + num_threads - 1) / num_threads;

  constant float* input  = &inputBuffer.data[channel * input_size + tid * chunk_size];

  const int max_index = min(chunk_size, input_size - tid * chunk_size);

  float4 sum = 0;
  int i = 0;
  for (; i < max_index-16; i+=16) {
    sum += ((constant float4 *) input)[i/4];
    sum += ((constant float4 *) input)[i/4+1];
    sum += ((constant float4 *) input)[i/4+2];
    sum += ((constant float4 *) input)[i/4+3];
  }
  for (; i < max_index-8; i+=8) {
    sum += ((constant float4 *) input)[i/4];
    sum += ((constant float4 *) input)[i/4+1];
  }
  for (; i < max_index-4; i+=4) {
    sum += ((constant float4 *) input)[i/4];
  }
  for (; i < max_index; i++) {
    sum[0] += input[i];
  }
  per_thread_sum[tid] = sum[0] + sum[1] + sum[2] + sum[3];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int last_bit_size = 32;
  const int last_bit_section = num_threads/last_bit_size;

  if (tid < last_bit_size) {
    sum = 0;
    for (int t = 0; t < last_bit_section/4; t++) {
      sum += ((threadgroup float4 *) per_thread_sum)[tid * last_bit_section/4 + t];
    }
    per_thread_sum[tid] = sum[0] + sum[1] + sum[2] + sum[3];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    sum = 0;
    for (int t = 0; t < last_bit_size/4; t++) {
      sum += ((threadgroup float4 *) per_thread_sum)[t];
    }
    avgBuffer.data[channel] = (sum[0] + sum[1] + sum[2] + sum[3]) / input_size;
  }
}

kernel void cnn_instance_norm_stdev_kern(
                                         constant channel_data &scaleBuffer         [[ buffer(0) ]],
                                         constant channel_data &biasBuffer          [[ buffer(1) ]],
                                         constant input_output_data &inputBuffer    [[ buffer(2) ]],
                                         device stats_data &avgBuffer               [[ buffer(4) ]],
                                         device stats_data &stdevBuffer             [[ buffer(5) ]],
                                         constant epsilon_data &epsilon             [[ buffer(6) ]],
                                         threadgroup float *per_thread_sq_norm      [[ threadgroup(0) ]],
                                         ushort channel                             [[ threadgroup_position_in_grid ]],
                                         ushort num_threads                         [[ threads_per_threadgroup ]],
                                         ushort tid                                 [[ thread_index_in_threadgroup ]]
                                         ) {
  const int chunk_size = (input_size + num_threads - 1) / num_threads;

  constant float* input  = &inputBuffer.data[channel * input_size + tid * chunk_size];

  const int max_index = min(chunk_size, input_size - tid * chunk_size);

  float4 sum = 0;
  float mean = avgBuffer.data[channel];

  int i = 0;
  for (; i < max_index-16; i+=16) {
    float4 delta = ((constant float4 *) input)[i/4] - mean;
    sum += delta * delta;
    delta = ((constant float4 *) input)[i/4+1] - mean;
    sum += delta * delta;
    delta = ((constant float4 *) input)[i/4+2] - mean;
    sum += delta * delta;
    delta = ((constant float4 *) input)[i/4+3] - mean;
    sum += delta * delta;
  }
  for (; i < max_index-8; i+=8) {
    float4 delta = ((constant float4 *) input)[i/4] - mean;
    sum += delta * delta;
    delta = ((constant float4 *) input)[i/4+1] - mean;
    sum += delta * delta;
  }
  for (; i < max_index-4; i+=4) {
    float4 delta = ((constant float4 *) input)[i/4] - mean;
    sum += delta * delta;
  }
  for (; i < max_index; i++) {
    half delta = input[i] - mean;
    sum[0] += delta * delta;
  }
  per_thread_sq_norm[tid] = sum[0] + sum[1] + sum[2] + sum[3];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int last_bit_size = 32;
  const int last_bit_section = num_threads/last_bit_size;

  if (tid < last_bit_size) {
    sum = 0;
    for (int t = 0; t < last_bit_section/4; t++) {
      sum += ((threadgroup float4 *) per_thread_sq_norm)[tid * last_bit_section/4 + t];
    }
    per_thread_sq_norm[tid] = sum[0] + sum[1] + sum[2] + sum[3];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    sum = 0;
    for (int t = 0; t < last_bit_size/4; t++) {
      sum += ((threadgroup float4 *) per_thread_sq_norm)[t];
    }

    float inv_stdev = 1.0h / sqrt((sum[0] + sum[1] + sum[2] + sum[3]) / input_size + epsilon.data[0]);
    float scale = inv_stdev * scaleBuffer.data[channel];
    float shift = biasBuffer.data[channel] - mean * scale;

    avgBuffer.data[channel] = scale; //scale
    stdevBuffer.data[channel] = shift; // shift
  }
}

kernel void cnn_instance_norm_last_kern(
                                        constant input_output_data &inputBuffer           [[ buffer(2) ]],
                                        device input_output_data &outputBuffer            [[ buffer(3) ]],
                                        constant stats_data &avgBuffer                    [[ buffer(4) ]], //scale
                                        constant stats_data &stdevBuffer                  [[ buffer(5) ]], //shift
                                        uint2 gid                                         [[ thread_position_in_grid ]]) {
  if (gid.x < input_size_x && gid.y < input_size_y) {
    for (int c = 0; c < channels; c++) {
      int idx = c * input_size + gid.y * input_size_x + gid.x;
      outputBuffer.data[idx] = half(inputBuffer.data[idx]) * avgBuffer.data[c] + stdevBuffer.data[c];
    }
  }
}

kernel void cnn_instance_norm_prelu_kern(
                                         constant input_output_data &inputBuffer           [[ buffer(2) ]],
                                         device input_output_data &outputBuffer            [[ buffer(3) ]],
                                         constant stats_data &avgBuffer                    [[ buffer(4) ]], //scale
                                         constant stats_data &stdevBuffer                  [[ buffer(5) ]], //shift
                                         constant channel_data &preluBuffer                [[ buffer(7) ]], //prelu weights
                                         uint2 gid                                         [[ thread_position_in_grid ]]) {
  if (gid.x < input_size_x && gid.y < input_size_y) {
    for (int c = 0; c < channels; c++) {
      int idx = c * input_size + gid.y * input_size_x + gid.x;
      half weight = preluBuffer.data[prelu_size > 1 ? c : 0];
      half value = half(inputBuffer.data[idx]) * avgBuffer.data[c] + stdevBuffer.data[c];
      outputBuffer.data[idx] = value > 0 ? value : value * weight;
    }
  }
}
)Metal";
}
