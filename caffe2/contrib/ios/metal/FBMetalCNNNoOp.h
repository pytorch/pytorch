// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import "arm_neon_support.h"
#import "MetalImageFilter.h"

@interface FBMetalCNNNoOp: MetalImageFilter

+ (instancetype)filterWithContext:(MetalContext*)context
                            width:(NSUInteger)width
                           height:(NSUInteger)height;

@end
