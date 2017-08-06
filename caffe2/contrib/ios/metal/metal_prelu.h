// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

bool metal_prelu(
    id<MTLBuffer> inputBuffer,
    int           input_channels,
    int           input_width,
    int           input_height,
    id<MTLBuffer> weightBuffer,
    int           weight_length,
    id<MTLBuffer> outputBuffer);
