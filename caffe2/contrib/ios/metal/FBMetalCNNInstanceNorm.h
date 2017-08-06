// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import "arm_neon_support.h"
#import "MetalImageFilter.h"

class FBMetalInstanceNormConstantValues : public FBMetalConstantValues {
  public:
  ushort input_width;
  ushort input_height;
  ushort input_channels;
  ushort prelu_size;

  FBMetalInstanceNormConstantValues(
                                    ushort _input_width,
                                    ushort _input_height,
                                    ushort _input_channels,
                                    ushort _prelu_size)
  : input_width(_input_width),
  input_height(_input_height),
  input_channels(_input_channels),
  prelu_size(_prelu_size) {}

  std::string to_string();
};

@interface FBMetalCNNInstanceNormBase : MetalImageFilter

@property (nonatomic, strong) id<MTLBuffer> dataBuffer;
@property (nonatomic, strong) id<MTLBuffer> outputBuffer;
@property (nonatomic, strong) id<MTLBuffer> scaleBuffer;
@property (nonatomic, strong) id<MTLBuffer> biasBuffer;
@property (nonatomic, strong) id<MTLBuffer> avgBuffer;
@property (nonatomic, strong) id<MTLBuffer> stdevBuffer;
@property (nonatomic, strong) id<MTLBuffer> preluBuffer;
@property (nonatomic, strong) id<MTLBuffer> epsilonBuffer;

@property (nonatomic, copy) NSString *functionName;

+ (instancetype)filterWithContext:(MetalContext*)context
                     functionName:(NSString*)functionName
                   constantValues:(FBMetalInstanceNormConstantValues*)constantValues
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                          channel:(NSUInteger)channel;
@end


@interface FBMetalCNNInstanceNorm: FBMetalCNNInstanceNormBase

+ (instancetype)filterWithContext:(MetalContext*)context
                   constantValues:(FBMetalInstanceNormConstantValues*)constantValues
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                          channel:(NSUInteger)channel
                        withPRelu:(BOOL)withPRelu;

- (void)loadEpsilon:(const float)epsilon;

@end
