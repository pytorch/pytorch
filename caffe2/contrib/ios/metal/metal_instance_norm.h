// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import <Metal/Metal.h>

bool metal_instance_norm(
                         id<MTLBuffer> inputBuffer,
                         int           input_channels,
                         int           input_width,
                         int           input_height,
                         id<MTLBuffer> scaleDataBuffer,
                         id<MTLBuffer> biasDataBuffer,
                         id<MTLBuffer> outputBuffer,
                         id<MTLBuffer> preluBuffer,
                         int           prelu_length,
                         float         epsilon_);
