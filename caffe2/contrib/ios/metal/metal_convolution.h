// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import <Metal/Metal.h>

bool calculate_kernels_per_convolution(int& kernels_per_convolution);

bool metal_convolution(
    id<MTLBuffer> inputBuffer,
    int           input_channels,
    int           input_width,
    int           input_height,
    int           input_stride_x,
    int           input_stride_y,
    int           input_pad_t,
    int           input_pad_l,
    int           input_pad_b,
    int           input_pad_r,
    id<MTLBuffer> weightBuffer,
    int           output_channels,
    int           kernel_channels,
    int           kernel_width,
    int           kernel_height,
    id<MTLBuffer> outputBuffer,
    int           output_number,
    int           output_width,
    int           output_height,
    const float*  bias,
    int           bias_length,
    bool          transposed);
