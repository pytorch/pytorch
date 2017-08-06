// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import <Foundation/Foundation.h>

@protocol MTLDevice, MTLLibrary, MTLCommandQueue;

@interface MetalContext : NSObject

@property (atomic, strong) id<MTLDevice>       device;
@property (atomic, strong) id<MTLLibrary>      library;
@property (atomic, strong) id<MTLCommandQueue> commandQueue;

+ (instancetype)getContext;

@end
