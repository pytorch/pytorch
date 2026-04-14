//  Copyright © 2026 Apple Inc.

#import <ATen/mps/MPSRecordingEncoder.h>

@implementation MPSRecordingEncoder {
  id<MTLComputeCommandEncoder> _inner;
  at::mps::MPSStream* _stream;
  std::unique_ptr<at::mps::MPSStream::CapturedMetalKernel> _pending;
}

- (instancetype)initWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                         stream:(at::mps::MPSStream*)stream {
  if ((self = [super init])) {
    _inner = [encoder retain];
    _stream = stream;
  }
  return self;
}

- (void)dealloc {
  if (_pending && _pending->pso) {
    [(__bridge id<MTLComputePipelineState>)_pending->pso release];
  }
  [_inner release];
  [super dealloc];
}

// Forward any selector we don't override to the real encoder.
- (id)forwardingTargetForSelector:(SEL)sel {
  return _inner;
}

- (BOOL)respondsToSelector:(SEL)sel {
  return [super respondsToSelector:sel] || [_inner respondsToSelector:sel];
}

#pragma mark - Recording overrides

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
  if (_pending && _pending->pso) {
    [(__bridge id<MTLComputePipelineState>)_pending->pso release];
  }
  _pending = std::make_unique<at::mps::MPSStream::CapturedMetalKernel>();
  _pending->pso = (__bridge void*)state;
  [state retain];

  [_inner setComputePipelineState:state];
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
  if (_pending) {
    _pending->buffers.push_back({
      (__bridge void*)buffer,
      static_cast<size_t>(offset),
      static_cast<unsigned>(index),
      static_cast<size_t>([buffer length]),
    });
  }
  [_inner setBuffer:buffer offset:offset atIndex:index];
}

- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
  if (_pending) {
    at::mps::MPSStream::CapturedMetalKernel::BytesBinding b;
    b.data.assign(static_cast<const uint8_t*>(bytes),
                  static_cast<const uint8_t*>(bytes) + length);
    b.index = static_cast<unsigned>(index);
    _pending->bytes.push_back(std::move(b));
  }
  [_inner setBytes:bytes length:length atIndex:index];
}

- (void)dispatchThreads:(MTLSize)threadsPerGrid
  threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup {
  if (_pending) {
    _pending->gridX = threadsPerGrid.width;
    _pending->gridY = threadsPerGrid.height;
    _pending->gridZ = threadsPerGrid.depth;
    _pending->tgX = threadsPerThreadgroup.width;
    _pending->tgY = threadsPerThreadgroup.height;
    _pending->tgZ = threadsPerThreadgroup.depth;
    _pending->useThreadgroups = false;
    _stream->pushCapturedMetalKernel(std::move(_pending));
  }
  [_inner dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}

- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid
       threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup {
  if (_pending) {
    _pending->gridX = threadgroupsPerGrid.width;
    _pending->gridY = threadgroupsPerGrid.height;
    _pending->gridZ = threadgroupsPerGrid.depth;
    _pending->tgX = threadsPerThreadgroup.width;
    _pending->tgY = threadsPerThreadgroup.height;
    _pending->tgZ = threadsPerThreadgroup.depth;
    _pending->useThreadgroups = true;
    _stream->pushCapturedMetalKernel(std::move(_pending));
  }
  [_inner dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}

@end
