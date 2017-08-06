// Copyright 2004-present Facebook. All Rights Reserved.

#import "MetalImageFilter.h"
#import "MetalContext.h"

#import <TargetConditionals.h>
#import <Metal/Metal.h>

#include "caffe2/core/logging.h"

@interface MetalImageFilter ()
@property (nonatomic, strong) id<MTLFunction> kernelFunction;
@end

@implementation MetalImageFilter {
  NSString* _functionName;
}
static constexpr size_t kThreadGroupSize_x = 4;
static constexpr size_t kThreadGroupSize_y = 8;
static constexpr bool   kEnableGPUProfiling     = false; // GPU profiling is expensive, have it off by default

#import "MetalShaderUtilities.h"

@synthesize outputTextureDescriptor = _outputTextureDescriptor;

static NSMutableDictionary<NSString*, id<MTLLibrary>>* functionLibraryCache = NULL;

- (id<MTLFunction>)newFunctionWithName:(NSString*)functionName
                           libraryName:(NSString*)libraryName
                         librarySource:(NSString*)librarySource
                               context:(MetalContext*)context
                        constantValues:(FBMetalConstantValues*)constantValues {
  NSString* library_version =
      constantValues ? [libraryName stringByAppendingString:[[NSString alloc]
                                                             initWithUTF8String:constantValues->to_string().c_str()]]
                     : libraryName;

  if (functionLibraryCache == NULL) {
    functionLibraryCache = [NSMutableDictionary<NSString*, id<MTLLibrary>> dictionary];
  }

  id<MTLLibrary> library = functionLibraryCache[library_version];

  if (library == nil) {
    librarySource = [self replaceConstantValues:constantValues librarySource:librarySource];

    MTLCompileOptions* options = [MTLCompileOptions alloc];
    options.fastMathEnabled    = TRUE;
    options.languageVersion    = MTLLanguageVersion1_0;

    NSError* error = NULL;
    library        = [context.device newLibraryWithSource:librarySource options:options error:&error];

    if (error != nil) {
      NSString* description = [[NSString alloc] init];
      LOG(ERROR) << "Problems with library for " << [library_version UTF8String] << " : "
      << [[error localizedDescription] UTF8String];
    }

    functionLibraryCache[library_version] = library;
  }

  return library != nil ? [library newFunctionWithName:functionName] : nil;
}

- (NSString*)replaceConstantValues:(FBMetalConstantValues*)constantValues librarySource:(NSString*)librarySource {
  return librarySource;
}

- (instancetype)initWithFunctionName:(NSString*)functionName
                         libraryName:(NSString*)libraryName
                       librarySource:(NSString*)librarySource
                             context:(MetalContext*)context
                      constantValues:(FBMetalConstantValues*)constantValues {
  if ((self = [super init])) {
    NSError* error = nil;

    _context = context;
    _kernelFunction =
        [self newFunctionWithName:functionName libraryName:libraryName librarySource:librarySource context:context constantValues:constantValues];
    _functionName = functionName;
    _pipeline     = [context.device newComputePipelineStateWithFunction:_kernelFunction error:&error];

    if (!_pipeline) {
      LOG(ERROR) << "Error occurred when building compute pipeline for function " << [functionName UTF8String];
      return nil;
    }
  }

  return self;
}

- (void)configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
                              weightBufferOffset:(NSInteger)weightBufferOffset
                              outputBufferOffset:(NSInteger)outputBufferOffset {
}

- (bool)checkExecution:(id<MTLCommandBuffer>)commandBuffer {
  NSError* error = [commandBuffer error];

  if (error != nil) {
    LOG(ERROR) << "Problems with " << [self->_functionName UTF8String] << " : "
    << [[error localizedDescription] UTF8String];
    return false;
  }

  return true;
}

- (MTLSize)threadsPerThreadgroup {
  return MTLSizeMake(kThreadGroupSize_x, kThreadGroupSize_y, 1);
}

- (MTLSize)threadgroupsPerGrid {
  MTLSize threadsPerThreadgroup = [self threadsPerThreadgroup];

  return MTLSizeMake(
                     (self.outputTextureDescriptor.width + threadsPerThreadgroup.width - 1) /
                     threadsPerThreadgroup.width,
                     (self.outputTextureDescriptor.height + threadsPerThreadgroup.height - 1) /
                     threadsPerThreadgroup.height,
                     1);
}

- (void)applyFilter:(void (^)(NSError*))completionHandler {
  [self applyFilter:completionHandler weightBufferOffset:0 outputBufferOffset:0];
}

- (void)applyFilter:(void (^)(NSError*))completionHandler
 weightBufferOffset:(NSInteger)weightBufferOffset
 outputBufferOffset:(NSInteger)outputBufferOffset {
  MTLTextureDescriptor* textureDescriptor = self.outputTextureDescriptor;

  if (kEnableGPUProfiling) {
    [self.context.commandQueue insertDebugCaptureBoundary];
  }

  id<MTLCommandBuffer> commandBuffer = [self.context.commandQueue commandBuffer];

  /*
   * It is not obvious which grid strategy is best, maximizing the number of threadgroups
   * seems to give better results, but more investigation is needed
   */

  id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
  [commandEncoder setComputePipelineState:self.pipeline];

  [self configureArgumentTableWithCommandEncoder:commandEncoder
                              weightBufferOffset:weightBufferOffset
                              outputBufferOffset:outputBufferOffset];

  [commandEncoder dispatchThreadgroups:[self threadgroupsPerGrid] threadsPerThreadgroup:[self threadsPerThreadgroup]];

  [commandEncoder endEncoding];

  [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> commandBuffer) {
    if (completionHandler != NULL) {
      completionHandler([commandBuffer error]);
    } else {
      [self checkExecution:commandBuffer];
    }
  }];

  [commandBuffer commit];

  if (kEnableGPUProfiling) {
    [self.context.commandQueue insertDebugCaptureBoundary];
  }
}

@end
