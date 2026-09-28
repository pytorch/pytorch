//  Copyright © 2026 Apple Inc.

#import <ATen/mps/MPSRecordingEncoder.h>

#include <map>

@implementation MPSRecordingEncoder {
  id<MTLComputeCommandEncoder> _inner;
  at::mps::MPSStream* _stream;

  // Sticky binding state, mirrors the current state of `_inner`.
  //
  // Metal encoder binding tables persist across dispatches and across
  // setComputePipelineState: calls until overwritten. Some MPS kernels rely
  // on this (e.g. bind a resource once, switch PSO, dispatch again without
  // rebinding). Recording only the calls since the last setPSO would miss
  // those inherited bindings, so on dispatch we snapshot the FULL current
  // binding state into a CapturedMetalKernel. Each captured step is then
  // self-contained on replay, even when a fresh encoder is created after an
  // interleaved MPSGraph step.
  //
  // Buffer index space is shared between setBuffer:offset:atIndex: and
  // setBytes:length:atIndex:, so setting one at index N clears the other.
  // Threadgroup memory has its own index space.
  //
  // Retain ownership:
  //   _currentPSO holds one retain (released on overwrite or dealloc).
  //   Each buffer in _stickyBuffers holds one retain (released on overwrite,
  //     when displaced by a setBytes at the same index, or on dealloc).
  //   Snapshots into CapturedMetalKernel take a separate retain; those are
  //     released by MPSStream::releaseCapturedStep.
  void* _currentPSO; // id<MTLComputePipelineState>, retained (or nullptr)
  std::map<unsigned, at::mps::MPSStream::CapturedMetalKernel::BufferBinding> _stickyBuffers;
  std::map<unsigned, at::mps::MPSStream::CapturedMetalKernel::BytesBinding> _stickyBytes;
  std::map<unsigned, at::mps::MPSStream::CapturedMetalKernel::ThreadgroupMemoryBinding> _stickyThreadgroupMemory;
  // Keyed by resource pointer rather than index: useResource: has no index of
  // its own, and Metal combines usages when the same resource is declared
  // more than once, so repeat declarations OR into the existing entry instead
  // of each getting their own retain.
  std::map<void*, unsigned long> _stickyResourceUsages;
}

- (instancetype)initWithEncoder:(id<MTLComputeCommandEncoder>)encoder stream:(at::mps::MPSStream*)stream {
  if ((self = [super init])) {
    _inner = [encoder retain];
    _stream = stream;
    _currentPSO = nullptr;
  }
  return self;
}

- (void)releaseStickyState {
  if (_currentPSO) {
    [(__bridge id<MTLComputePipelineState>)_currentPSO release];
    _currentPSO = nullptr;
  }
  for (auto& [idx, buf] : _stickyBuffers) {
    [(__bridge id<MTLBuffer>)buf.buffer release];
  }
  _stickyBuffers.clear();
  _stickyBytes.clear();
  _stickyThreadgroupMemory.clear();
  for (auto& [resource, usage] : _stickyResourceUsages) {
    [(__bridge id<MTLResource>)resource release];
  }
  _stickyResourceUsages.clear();
}

- (void)dealloc {
  [self releaseStickyState];
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
  // Only the PSO changes; sticky buffer/bytes/threadgroup bindings persist
  // across setComputePipelineState per Metal semantics.
  if (_currentPSO) {
    [(__bridge id<MTLComputePipelineState>)_currentPSO release];
  }
  _currentPSO = (__bridge void*)state;
  [state retain];
  [_inner setComputePipelineState:state];
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
  unsigned idx = static_cast<unsigned>(index);
  // Overwriting a prior binding at the same index: release the old buffer.
  auto it = _stickyBuffers.find(idx);
  if (it != _stickyBuffers.end()) {
    [(__bridge id<MTLBuffer>)it->second.buffer release];
    _stickyBuffers.erase(it);
  }
  // Buffer and bytes share the index space on Metal; setBuffer displaces
  // any prior setBytes at the same index.
  _stickyBytes.erase(idx);

  [buffer retain];
  _stickyBuffers[idx] = {
      (__bridge void*)buffer,
      static_cast<size_t>(offset),
      idx,
  };
  [_inner setBuffer:buffer offset:offset atIndex:index];
}

- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
  unsigned idx = static_cast<unsigned>(index);
  // setBytes displaces any prior setBuffer at the same index.
  auto it = _stickyBuffers.find(idx);
  if (it != _stickyBuffers.end()) {
    [(__bridge id<MTLBuffer>)it->second.buffer release];
    _stickyBuffers.erase(it);
  }

  at::mps::MPSStream::CapturedMetalKernel::BytesBinding b;
  b.data.assign(static_cast<const uint8_t*>(bytes), static_cast<const uint8_t*>(bytes) + length);
  b.index = idx;
  _stickyBytes[idx] = std::move(b);
  [_inner setBytes:bytes length:length atIndex:index];
}

- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
  unsigned idx = static_cast<unsigned>(index);
  _stickyThreadgroupMemory[idx] = {
      static_cast<size_t>(length),
      idx,
  };
  [_inner setThreadgroupMemoryLength:length atIndex:index];
}

- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage {
  void* key = (__bridge void*)resource;
  auto it = _stickyResourceUsages.find(key);
  if (it == _stickyResourceUsages.end()) {
    [resource retain];
    _stickyResourceUsages[key] = usage;
  } else {
    // Matches Metal's own semantics: redeclaring a resource combines usages
    // rather than replacing them.
    it->second |= usage;
  }
  [_inner useResource:resource usage:usage];
}

- (std::unique_ptr<at::mps::MPSStream::CapturedMetalKernel>)snapshotDispatch {
  auto kernel = std::make_unique<at::mps::MPSStream::CapturedMetalKernel>();
  kernel->pso = _currentPSO;
  if (_currentPSO) {
    [(__bridge id<MTLComputePipelineState>)_currentPSO retain];
  }
  kernel->buffers.reserve(_stickyBuffers.size());
  for (auto& [idx, buf] : _stickyBuffers) {
    kernel->buffers.push_back(buf);
    [(__bridge id<MTLBuffer>)buf.buffer retain];
  }
  kernel->bytes.reserve(_stickyBytes.size());
  for (auto& [idx, b] : _stickyBytes) {
    kernel->bytes.push_back(b);
  }
  kernel->threadgroupMemory.reserve(_stickyThreadgroupMemory.size());
  for (auto& [idx, tg] : _stickyThreadgroupMemory) {
    kernel->threadgroupMemory.push_back(tg);
  }
  kernel->resourceUsages.reserve(_stickyResourceUsages.size());
  for (auto& [resource, usage] : _stickyResourceUsages) {
    kernel->resourceUsages.push_back({resource, usage});
    [(__bridge id<MTLResource>)resource retain];
  }
  return kernel;
}

- (void)dispatchThreads:(MTLSize)threadsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup {
  if (_currentPSO) {
    auto kernel = [self snapshotDispatch];
    kernel->gridX = threadsPerGrid.width;
    kernel->gridY = threadsPerGrid.height;
    kernel->gridZ = threadsPerGrid.depth;
    kernel->tgX = threadsPerThreadgroup.width;
    kernel->tgY = threadsPerThreadgroup.height;
    kernel->tgZ = threadsPerThreadgroup.depth;
    kernel->useThreadgroups = false;
    _stream->pushCapturedMetalKernel(std::move(kernel));
  }
  [_inner dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}

- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup {
  if (_currentPSO) {
    auto kernel = [self snapshotDispatch];
    kernel->gridX = threadgroupsPerGrid.width;
    kernel->gridY = threadgroupsPerGrid.height;
    kernel->gridZ = threadgroupsPerGrid.depth;
    kernel->tgX = threadsPerThreadgroup.width;
    kernel->tgY = threadsPerThreadgroup.height;
    kernel->tgZ = threadsPerThreadgroup.depth;
    kernel->useThreadgroups = true;
    _stream->pushCapturedMetalKernel(std::move(kernel));
  }
  [_inner dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
}

@end
