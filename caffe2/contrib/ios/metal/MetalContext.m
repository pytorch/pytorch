// Copyright 2004-present Facebook. All Rights Reserved.

#import "MetalContext.h"

#import <Metal/Metal.h>

static MetalContext* metalContext = NULL;

@implementation MetalContext

+ (instancetype)getContext {
  if (metalContext == NULL) {
    metalContext = [[self alloc] initWithDevice:nil];
  }
  return metalContext;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device {
  if ((self = [super init])) {
    _device       = device ?: MTLCreateSystemDefaultDevice();
    _library      = [_device newDefaultLibrary];
    _commandQueue = [_device newCommandQueue];
  }
  return self;
}

@end
