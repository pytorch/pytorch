// Copyright 2004-present Facebook. All Rights Reserved.

#import "FBMetalCNNNoOp.h"
#import "MetalShaderUtilities.h"

namespace {
  extern const char *metalCode;
}

@implementation FBMetalCNNNoOp

- (MTLSize)threadsPerThreadgroup {
    return MTLSizeMake(1, 1, 1);
}

- (MTLSize)threadgroupsPerGrid {
    return MTLSizeMake(1, 1, 1);
}

+ (instancetype)filterWithContext:(MetalContext*)context
                            width:(NSUInteger)width
                           height:(NSUInteger)height {
  return [[self alloc] initWithContext:context width:width height:height];
}

- (instancetype)initWithContext:(MetalContext*)context
                          width:(NSUInteger)width
                         height:(NSUInteger)height {
  if ((self = [super initWithFunctionName:@"cnn_no_op_kern"
                              libraryName:@"NoOp"
                            librarySource:[NSString stringWithCString:metalCode encoding:NSUTF8StringEncoding]
                                  context:context
                           constantValues:nil])) {
    super.outputTextureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                       width:width
                                                                                      height:height
                                                                                   mipmapped:NO];
  }
  return self;
}
@end

namespace {
const char *metalCode = R"Metal(
//  Copyright 2004-present Facebook. All Rights Reserved.

#include <metal_stdlib>
using namespace metal;

kernel void cnn_no_op_kern() {
  return;
}
)Metal";
}
