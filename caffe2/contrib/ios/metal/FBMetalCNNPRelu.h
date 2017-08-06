// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import "arm_neon_support.h"
#import "MetalImageFilter.h"

class FBMetalPReluConstantValues : public FBMetalConstantValues {
public:
  ushort input_width;
  ushort input_height;
  ushort input_channels;
  ushort weight_length;

  FBMetalPReluConstantValues(
                             ushort _input_width,
                             ushort _input_height,
                             ushort _input_channels,
                             ushort _weight_length)
  : input_width(_input_width),
  input_height(_input_height),
  input_channels(_input_channels),
  weight_length(_weight_length) {}

  std::string to_string();
};

@interface FBMetalCNNPRelu : MetalImageFilter

@property (nonatomic, strong) id<MTLBuffer> dataBuffer;
@property (nonatomic, strong) id<MTLBuffer> outputBuffer;
@property (nonatomic, strong) id<MTLBuffer> weightBuffer;

+ (instancetype)filterWithContext:(MetalContext*)context
                   constantValues:(FBMetalPReluConstantValues*)constantValues
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                          channel:(NSUInteger)channel;

@end
