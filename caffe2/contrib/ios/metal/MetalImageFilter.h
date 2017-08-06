// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import "MetalContext.h"
#import "FBMetalConstantValues.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState;

@interface MetalImageFilter : NSObject

@property (nonatomic, strong) MetalContext*               context;
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;

@property (atomic, strong) MTLTextureDescriptor* outputTextureDescriptor;

- (instancetype)initWithFunctionName:(NSString*)functionName
                         libraryName:(NSString*)libraryName
                       librarySource:(NSString*)librarySource
                             context:(MetalContext*)context
                      constantValues:(FBMetalConstantValues*)constantValues;

- (NSString*)replaceConstantValues:(FBMetalConstantValues *)constantValues
                     librarySource:(NSString*)librarySource;

- (void)configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
                              weightBufferOffset:(NSInteger)weightBufferOffset
                              outputBufferOffset:(NSInteger)outputBufferOffset;

- (void)applyFilter:(void (^)(NSError*))completionHandler;

- (void) applyFilter:(void(^)(NSError*))completionHandler
  weightBufferOffset:(NSInteger)weightBufferOffset
  outputBufferOffset:(NSInteger)outputBufferOffset;

@end
