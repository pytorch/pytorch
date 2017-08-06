// Copyright 2004-present Facebook. All Rights Reserved.

#import "arm_neon_support.h"
#import "MetalContext.h"

#import "FBMetalCNNInstanceNorm.h"
#import <Metal/Metal.h>

@interface InstanceNormCacheEntry : NSObject
@property (nonatomic, strong) FBMetalCNNInstanceNorm* instanceNorm;
@end
@implementation InstanceNormCacheEntry
@end

static MetalContext* metalContext = NULL;
static NSMutableDictionary<NSString*, InstanceNormCacheEntry*>* instanceNormCache = NULL;

static void init_metal_pipeline() {
  if (metalContext == NULL) {
    metalContext      = [MetalContext getContext];
    instanceNormCache = [NSMutableDictionary<NSString*, InstanceNormCacheEntry*> dictionary];
  }
}

bool metal_instance_norm(
                         id<MTLBuffer> inputBuffer,
                         int           input_channels,
                         int           input_width,
                         int           input_height,
                         id<MTLBuffer> scaleDataBuffer,
                         id<MTLBuffer> biasDataBuffer,
                         id<MTLBuffer> outputBuffer,
                         id<MTLBuffer> preluBuffer,
                         int           prelu_size,
                         float         epsilon_) {
  init_metal_pipeline();

  NSString* key = [NSString stringWithFormat:@"X:%d:%d:%d-P:%d", input_channels, input_width, input_height, prelu_size];

  InstanceNormCacheEntry* cc = instanceNormCache[key];

  if (cc == NULL) {
    cc                     = [[InstanceNormCacheEntry alloc] init];
    instanceNormCache[key] = cc;

    FBMetalInstanceNormConstantValues constantValues =
    FBMetalInstanceNormConstantValues(input_width, input_height, input_channels, prelu_size);

    id<MTLBuffer> avgBuffer =
    [metalContext.device newBufferWithLength:sizeof(float) * input_channels options:MTLStorageModeShared];
    id<MTLBuffer> stdevBuffer =
    [metalContext.device newBufferWithLength:sizeof(float) * input_channels options:MTLStorageModeShared];

    FBMetalCNNInstanceNorm* instanceNorm = [FBMetalCNNInstanceNorm filterWithContext:metalContext
                                                                      constantValues:&constantValues
                                                                               width:input_width
                                                                              height:input_height
                                                                             channel:input_channels
                                                                           withPRelu:preluBuffer != nil];

    instanceNorm.avgBuffer   = avgBuffer;
    instanceNorm.stdevBuffer = stdevBuffer;
    [instanceNorm loadEpsilon:epsilon_];
    cc.instanceNorm = instanceNorm;
  }

  FBMetalCNNInstanceNorm* instanceNorm = cc.instanceNorm;
  instanceNorm.dataBuffer              = inputBuffer;
  instanceNorm.outputBuffer            = outputBuffer;
  instanceNorm.scaleBuffer             = scaleDataBuffer;
  instanceNorm.biasBuffer              = biasDataBuffer;
  instanceNorm.preluBuffer             = preluBuffer;

  [instanceNorm applyFilter:(void (^)(NSError* error)) nullptr];

  return true;
}
