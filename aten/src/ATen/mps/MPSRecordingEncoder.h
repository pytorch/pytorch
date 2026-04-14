//  Copyright © 2026 Apple Inc.

#pragma once

#import <Metal/Metal.h>
#include <ATen/mps/MPSStream.h>

// Proxy that wraps id<MTLComputeCommandEncoder> during graph capture.
// Intercepts setComputePipelineState:, setBuffer:offset:atIndex:,
// setBytes:length:atIndex:, dispatchThreads:threadsPerThreadgroup:, and
// dispatchThreadgroups:threadsPerThreadgroup: to record Metal kernel state.
// All other selectors are forwarded to the real encoder via
// forwardingTargetForSelector:.
//
// This centralizes capture recording so that every Metal kernel going through
// MPSStream::commandEncoder() is automatically captured -- no per-site code.
@interface MPSRecordingEncoder : NSObject

- (instancetype)initWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                         stream:(at::mps::MPSStream*)stream;

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)dispatchThreads:(MTLSize)threadsPerGrid
  threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;
- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid
       threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;

@end
