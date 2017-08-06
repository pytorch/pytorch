// Copyright 2004-present Facebook. All Rights Reserved.

#import "arm_neon_support.h"
#import "MetalContext.h"
#import "FBMetalCNNPRelu.h"
#import "metal_prelu.h"

#include "caffe2/core/logging.h"

@interface PReluCacheEntry : NSObject
@property (nonatomic, strong) FBMetalCNNPRelu* prelu;
@end
@implementation PReluCacheEntry
@end

static MetalContext* metalContext = NULL;
static NSMutableDictionary<NSString*, PReluCacheEntry*>* preluCache = NULL;

static void init_metal_pipeline() {
  if (metalContext == NULL) {
    metalContext = [MetalContext getContext];
    preluCache   = [NSMutableDictionary<NSString*, PReluCacheEntry*> dictionary];
  }
}

bool metal_prelu(
    id<MTLBuffer> inputBuffer,
    int           input_channels,
    int           input_width,
    int           input_height,
    id<MTLBuffer> weightBuffer,
    int           weight_length,
    id<MTLBuffer> outputBuffer) {
  init_metal_pipeline();

  NSString* key =
      [NSString stringWithFormat:@"X:%d:%d:%d-F:%d", input_channels, input_width, input_height, weight_length];

  PReluCacheEntry* cc = preluCache[key];

  if (cc == NULL) {
    preluCache[key] = cc = [[PReluCacheEntry alloc] init];

    FBMetalPReluConstantValues constantValues =
        FBMetalPReluConstantValues(input_width, input_height, input_channels, weight_length);

    cc.prelu = [FBMetalCNNPRelu filterWithContext:metalContext
                                   constantValues:&constantValues
                                            width:input_width
                                           height:input_height
                                          channel:input_channels];
  }

  FBMetalCNNPRelu* prelu = cc.prelu;

  prelu.dataBuffer   = inputBuffer;
  prelu.outputBuffer = outputBuffer;
  prelu.weightBuffer = weightBuffer;

  [prelu applyFilter:(void (^)(NSError* error)) nullptr];

  return true;
}
