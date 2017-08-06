// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "arm_neon_support.h"

#import "MetalImageFilter.h"
#import "FBMetalConstantValues.h"

class FBMetalCNNConstantValues : public FBMetalConstantValues {
public:
  ushort input_width;
  ushort input_height;
  ushort input_channels;
  ushort input_stride_x;
  ushort input_stride_y;
  ushort input_pad_t;
  ushort input_pad_l;
  ushort input_pad_b;
  ushort input_pad_r;
  ushort filter_width;
  ushort filter_height;
  ushort output_width;
  ushort output_height;
  ushort output_channels;
  bool transposed;

  FBMetalCNNConstantValues(
      ushort _input_width,
      ushort _input_height,
      ushort _input_channels,
      ushort _input_stride_x,
      ushort _input_stride_y,
      ushort _input_pad_t,
      ushort _input_pad_l,
      ushort _input_pad_b,
      ushort _input_pad_r,
      ushort _filter_width,
      ushort _filter_height,
      ushort _output_width,
      ushort _output_height,
      ushort _output_channels,
      bool _transposed) :
        input_width(_input_width),
        input_height(_input_height),
        input_channels(_input_channels),
        input_stride_x(_input_stride_x),
        input_stride_y(_input_stride_y),
        input_pad_t(_input_pad_t),
        input_pad_l(_input_pad_l),
        input_pad_b(_input_pad_b),
        input_pad_r(_input_pad_r),
        filter_width(_filter_width),
        filter_height(_filter_height),
        output_width(_output_width),
        output_height(_output_height),
        output_channels(_output_channels),
        transposed(_transposed) { }

  std::string to_string();
};

constexpr int MAX_KERNELS_PER_CONVOLUTION = 32;

typedef float32_t data_buffer_type;

typedef float16_t weight_buffer_type;

@interface FBMetalCNNConvolution : MetalImageFilter

@property (nonatomic, strong) id<MTLBuffer> dataBuffer;

@property (nonatomic, strong) id<MTLBuffer> outputBuffer;

@property (nonatomic, strong) id<MTLBuffer> weightBuffer;

@property (nonatomic, strong) id<MTLBuffer> biasBuffer;

+ (id<MTLBuffer>)loadFilterWithImage:(const float*)weight_data
                        weightBuffer:(id<MTLBuffer>)weightBuffer
                             kernels:(NSUInteger)kernels
                       input_kernels:(NSUInteger)input_kernels
                       kernel_offset:(NSUInteger)kernel_offset
                       kernel_stride:(NSUInteger)kernel_stride
                            channels:(NSUInteger)channels
                               width:(NSUInteger)width
                              height:(NSUInteger)height
                          transposed:(bool)transposed
                             context:(MetalContext*)context;

- (void)loadFilterWithImage:(const float*)weight_data
                    kernels:(NSUInteger)kernels
              input_kernels:(NSUInteger)input_kernels
              kernel_offset:(NSUInteger)kernel_offset
              kernel_stride:(NSUInteger)kernel_stride
                   channels:(NSUInteger)channels
                      width:(NSUInteger)image_width
                     height:(NSUInteger)image_height
                 transposed:(bool)transposed;

+ (id<MTLBuffer>)loadDataWithImage:(const float*)imageData
                   imageDataBuffer:(id<MTLBuffer>)imageDataBuffer
                          channels:(NSUInteger)channels
                             width:(NSUInteger)width
                            height:(NSUInteger)height
                           context:(MetalContext*)context;

- (void)loadBiasData:(const float *)bias
              length:(NSUInteger)length;

+ (instancetype)filterWithContext:(MetalContext*)context
                         channels:(NSUInteger)channels
                      kernel_size:(NSUInteger)kernel_size
                   constantValues:(FBMetalCNNConstantValues*)constantValues
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                         stride_x:(NSUInteger)stride_x
                         stride_y:(NSUInteger)stride_y;
@end
