//  Copyright (c) 2026 Apple Inc.

#pragma once

#include <ATen/mps/MPSStream.h>
#import <Metal/Metal.h>

// Proxy that wraps id<MTLComputeCommandEncoder> during graph capture.
// Intercepts setComputePipelineState:, setBuffer:offset:atIndex:,
// setBytes:length:atIndex:, setThreadgroupMemoryLength:atIndex:,
// useResource:usage:, dispatchThreads:threadsPerThreadgroup:, and
// dispatchThreadgroups:threadsPerThreadgroup: to track cumulative encoder
// binding state. On every dispatch we snapshot the full sticky state into a
// CapturedMetalKernel so each captured step is self-contained on replay
// (matches Metal semantics where buffer/threadgroup bindings and declared
// resource usages persist across dispatches and setComputePipelineState
// calls until overwritten). All other selectors are forwarded to the real
// encoder via forwardingTargetForSelector:.
//
// Our own kernels dispatched through MPSStream::commandEncoder() are recorded
// this way with no per-op code. MPS-framework kernels that drive the encoder
// through other selectors are not; ops that use them call assertNotCapturing()
// or take another path while capturing.
@interface MPSRecordingEncoder : NSObject

- (instancetype)initWithEncoder:(id<MTLComputeCommandEncoder>)encoder stream:(at::mps::MPSStream*)stream;

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage;
- (void)dispatchThreads:(MTLSize)threadsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;
- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;

@end
