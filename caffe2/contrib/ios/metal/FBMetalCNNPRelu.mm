// Copyright 2004-present Facebook. All Rights Reserved.

#import "FBMetalCNNPRelu.h"
#import "MetalShaderUtilities.h"

#include "caffe2/core/logging.h"

std::string FBMetalPReluConstantValues::to_string() {
  std::ostringstream ss;

  ss  << ":" <<
  input_width << ":" <<
  input_height << ":" <<
  weight_length;

  return ss.str();
}

@interface FBMetalCNNPRelu() {
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

@implementation FBMetalCNNPRelu

static constexpr size_t kThreadGroupSize_x = 4;
static constexpr size_t kThreadGroupSize_y = 8;

+ (instancetype)filterWithContext:(MetalContext*)context
                   constantValues:(FBMetalPReluConstantValues*)constantValues
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                          channel:(NSUInteger)channel {
  return [[self alloc] initWithContext:context constantValues:constantValues width:width height:height channel:channel];
}

- (instancetype)initWithContext:(MetalContext*)context
                 constantValues:(FBMetalPReluConstantValues*)constantValues
                          width:(NSUInteger)width
                         height:(NSUInteger)height
                        channel:(NSUInteger)channel {
  if ((self = [super initWithFunctionName:@"cnn_prelu_kern"
                              libraryName:@"PRelu"
                            librarySource:[NSString stringWithCString:metalCode encoding:NSUTF8StringEncoding]
                                  context:context
                           constantValues:constantValues])) {
    configuration.channel = channel;
    configuration.width = width;
    configuration.height = height;

    super.outputTextureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                       width:width
                                                                                      height:height
                                                                                   mipmapped:NO];
  }
  return self;
}

- (MTLSize) threadsPerThreadgroup {
  return MTLSizeMake(kThreadGroupSize_x, kThreadGroupSize_y, 1);
}

- (NSString*)replaceConstantValues:(FBMetalConstantValues *)constantValues
                     librarySource:(NSString*)librarySource {
  FBMetalPReluConstantValues* convolutionConstantValues = (FBMetalPReluConstantValues *) constantValues;
  std::string source = [librarySource UTF8String];

  REPLACE_CONSTANT(source, convolutionConstantValues->input_width, 0);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_height, 1);
  REPLACE_CONSTANT(source, convolutionConstantValues->input_channels, 2);
  REPLACE_CONSTANT(source, convolutionConstantValues->weight_length,  3);

  return [NSString stringWithUTF8String:source.c_str()];
}

- (void)configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
                              weightBufferOffset:(NSInteger)weightBufferOffset
                              outputBufferOffset:(NSInteger)outputBufferOffset {
  [commandEncoder setBuffer:_weightBuffer offset:0 atIndex:0];
  [commandEncoder setBuffer:_outputBuffer offset:0 atIndex:1];
}

@end

namespace {
const char *metalCode = R"Metal(
//  Copyright 2004-present Facebook. All Rights Reserved.

#include <metal_stdlib>

using namespace metal;

constant constexpr ushort input_size_x      [[ function_constant(0) ]];
constant constexpr ushort input_size_y      [[ function_constant(1) ]];
constant constexpr ushort input_channels    [[ function_constant(2) ]];
constant constexpr ushort kernels           [[ function_constant(3) ]];

constant constexpr int input_size = input_size_x * input_size_y;

typedef float data_buffer_type;
typedef struct {
  data_buffer_type data[input_channels][input_size];
} output_data;

typedef struct {
  half data[kernels];
} filter_data;

kernel void cnn_prelu_kern(
                           constant filter_data &filter             [[ buffer(0) ]],
                           device output_data &output               [[ buffer(1) ]],
                           uint2 gid                                [[thread_position_in_grid]]) {
  if (gid.x < input_size_x && gid.y < input_size_y) {
    int idx = gid.y * input_size_x + gid.x;
    for (int c = 0; c < input_channels; c++) {
      half             weight = filter.data[kernels > 1 ? c : 0];
      data_buffer_type value  = output.data[c][idx];
      output.data[c][idx]     = value > 0 ? value : value * weight;
    }
  }
}
)Metal";
}
