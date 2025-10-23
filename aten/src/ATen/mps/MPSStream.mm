//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>

@interface MPSGraphExecutionDescriptor ()
@property(readwrite, atomic) BOOL enableCommitAndContinue;
@end

namespace at::mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

MPSStream::MPSStream(Stream stream) : _stream(stream) {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  TORCH_CHECK(_stream.device_type() == DeviceType::MPS);
  _serialQueue = dispatch_queue_create("metal gpu stream", nullptr);
  _executionDescriptor = [MPSGraphExecutionDescriptor new];
  _compilationDescriptor = [MPSGraphCompilationDescriptor new];

  // disable commitAndContinue if Signpost tracing is enabled
  if (getMPSProfiler().isSignpostTracingEnabled() || getMPSProfiler().isCaptureEnabled()) {
    _enableCommitAndContinue = false;
  }
  _executionDescriptor.enableCommitAndContinue = _enableCommitAndContinue;

  // Choose level which optimizes for GPU
  _compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel0;
  _executionDescriptor.compilationDescriptor = _compilationDescriptor;
}

MPSStream::~MPSStream() {
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];
  [_compilationDescriptor release];
  _executionDescriptor = nil;
  _compilationDescriptor = nil;

  assert(_commandBuffer == nil);
}

MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }

  return _commandBuffer;
}

id<MTLDevice> MPSStream::device() const {
  return [_commandQueue device];
}

id<MTLComputeCommandEncoder> MPSStream::commandEncoder() {
  if (!_commandEncoder) {
    _commandEncoder = [commandBuffer() computeCommandEncoder].retain;
  }

  return _commandEncoder;
}

void MPSStream::synchronize(SyncType syncType) {
  endKernelCoalescing();
  switch (syncType) {
    case SyncType::NONE:
      // typically in GPU to GPU copies we won't commit explicitly
      break;
    case SyncType::COMMIT:
      commit();
      break;
    case SyncType::COMMIT_ADAPTIVE:
      // the adaptive commit only commits if we hit the low watermark memory threshold
      if (getIMPSAllocator()->getLowWatermarkValue() <= 1) {
        commit();
      }
      break;
    case SyncType::COMMIT_AND_WAIT:
      commitAndWait();
      break;
    case SyncType::COMMIT_AND_CONTINUE:
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_enableCommitAndContinue,
                                       "CommitAndContinue is called but it is disabled globally!");
      commitAndContinue();
      break;
  }
}

void MPSStream::commit() {
  // Check if we've exceeded the operation count threshold
  if (_commandBufferOperationCount >= _commandBufferFlushThreshold) {
    // Flush to prevent unbounded accumulation
    flush();
  } else if (_enableCommitAndContinue) {
    // Below threshold: use commitAndContinue for performance
    commitAndContinue();
  } else {
    // CommitAndContinue disabled (e.g., profiling mode)
    flush();
  }
}

bool MPSStream::isOperationCached(uintptr_t opSignature) {
  auto it = _operationCache.find(opSignature);
  if (it != _operationCache.end()) {
    // Move to front of LRU (most recently used)
    _operationLRUList.splice(_operationLRUList.begin(), _operationLRUList, it->second);
    return true;
  }
  return false;
}

void MPSStream::addOperationToCache(uintptr_t opSignature) {
  // Check if already in cache
  if (_operationCache.find(opSignature) != _operationCache.end()) {
    return;
  }

  // If cache is at max size, evict least recently used item (from back of list)
  if (_operationCache.size() >= _maxOperationCacheSize) {
    uintptr_t lruSignature = _operationLRUList.back();
    _operationLRUList.pop_back();
    _operationCache.erase(lruSignature);
  }

  // Add to front of LRU list (most recently used)
  _operationLRUList.push_front(opSignature);
  _operationCache[opSignature] = _operationLRUList.begin();
}

void MPSStream::clearOperationCache() {
  _operationLRUList.clear();
  _operationCache.clear();
}

void MPSStream::commitAndWait() {
  if (_prevCommandBuffer) {
    // the previous command buffer (if exists) has already been committed,
    // so we just wait until it's completed and then dispose it.
    [_prevCommandBuffer waitUntilCompleted];
    [_prevCommandBuffer release];
    _prevCommandBuffer = nil;
  }

  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    [_commandBuffer release];
    _commandBuffer = nil;
  }
}

void MPSStream::commitAndContinue() {
  assert(_commandBuffer);
  [_commandBuffer commitAndContinue];
}

void MPSStream::endKernelCoalescing() {
  if (_commandEncoder) {
    [_commandEncoder endEncoding];
    [_commandEncoder release];
    _commandEncoder = nil;
  }
}

void MPSStream::flush() {
  if (_commandBuffer) {
    [_commandBuffer commit];
    // if commitAndContinue is disabled (e.g., for Profiler), we keep the command
    // buffer so we could wait on it later, if required.
    if (!_enableCommitAndContinue) {
      // Release any existing previous command buffer before overwriting. Again to prevent accumulation
      if (_prevCommandBuffer) {
        [_prevCommandBuffer release];
      }
      _prevCommandBuffer = _commandBuffer;
    } else {
      [_commandBuffer release];
    }
    _commandBuffer = nil;
    // Reset operation count for new command buffer
    // NOTE: LRU cache is NOT cleared - it persists across flushes for performance
    _commandBufferOperationCount = 0;
  }
}

void MPSStream::addCompletedHandler(MTLCommandBufferHandler block) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      [commandBuffer() addCompletedHandler:block];
      // Completion handlers are lightweight - track using a simple signature
      uintptr_t opSignature = reinterpret_cast<uintptr_t>(block);
      if (!isOperationCached(opSignature)) {
        addOperationToCache(opSignature);
      }
      // Increment operation count for flush threshold
      _commandBufferOperationCount++;
    }
  });
}

void MPSStream::fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType) {
  if (length == 0) {
    return;
  }
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      // For some reason fillBufferfor stopped working for lengh > 4Gb on MacOS 26
      // See https://github.com/pytorch/pytorch/issues/163962
      // Workaround by batching copy commands into 4Gb chunks
      constexpr size_t max_copy_size = 0x100000000; // 4GB
      size_t bytes_filled = 0;
      size_t bytes_remains = length;
      while (bytes_remains > 0) {
        NSUInteger bytes_to_copy = std::min(max_copy_size, bytes_remains);
        [blitEncoder fillBuffer:buffer range:NSMakeRange(offset + bytes_filled, bytes_to_copy) value:value];
        bytes_filled += bytes_to_copy;
        bytes_remains -= bytes_to_copy;
      }
      [blitEncoder endEncoding];

      // Track this operation in LRU cache
      // Create signature from buffer pointer (fill operations on same buffer are "same")
      uintptr_t opSignature = reinterpret_cast<uintptr_t>(buffer) ^ 0x1; // XOR with constant to distinguish from copy
      if (!isOperationCached(opSignature)) {
        addOperationToCache(opSignature);
      }

      // Increment operation count for flush threshold
      _commandBufferOperationCount++;

      synchronize(syncType);
    }
  });
}

void MPSStream::copy(id<MTLBuffer> srcBuffer,
                     id<MTLBuffer> dstBuffer,
                     size_t length,
                     size_t srcOffset,
                     size_t dstOffset,
                     uint64_t profileId,
                     SyncType syncType) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      // For some reason copyFromBuffer for 4Gb fails without returning an error
      // See https://github.com/pytorch/pytorch/issues/124335
      // Workaround by batching copy commands into 2Gb chunks
      constexpr size_t max_copy_size = 0x80000000; // 2GB
      size_t bytes_copied = 0;
      size_t bytes_remains = length;
      while (bytes_remains > 0) {
        NSUInteger bytes_to_copy = std::min(max_copy_size, bytes_remains);
        [blitEncoder copyFromBuffer:srcBuffer
                       sourceOffset:(NSUInteger)srcOffset + bytes_copied
                           toBuffer:dstBuffer
                  destinationOffset:(NSUInteger)dstOffset + bytes_copied
                               size:bytes_to_copy];
        bytes_copied += bytes_to_copy;
        bytes_remains -= bytes_to_copy;
      }
      [blitEncoder endEncoding];

      // Track this operation in LRU cache
      // Create signature from src/dst buffer pair (copy between same buffers are "same")
      uintptr_t opSignature = reinterpret_cast<uintptr_t>(srcBuffer) ^
                             (reinterpret_cast<uintptr_t>(dstBuffer) << 1);
      if (!isOperationCached(opSignature)) {
        addOperationToCache(opSignature);
      }

      // Increment operation count for flush threshold
      _commandBufferOperationCount++;

      // profilerId has a value only if copy profiling is enabled
      if (profileId) {
        getMPSProfiler().endProfileCopy(profileId, syncType);
      } else {
        synchronize(syncType);
      }
    }
  });
}

void MPSStream::copy_and_sync(id<MTLBuffer> srcBuffer,
                              id<MTLBuffer> dstBuffer,
                              size_t length,
                              size_t srcOffset,
                              size_t dstOffset,
                              bool non_blocking,
                              uint64_t profileId) {
  copy(srcBuffer,
       dstBuffer,
       length,
       srcOffset,
       dstOffset,
       profileId,
       !non_blocking ? SyncType::COMMIT_AND_WAIT : SyncType::COMMIT);
}

void MPSStream::executeMPSGraph(MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results, SyncType syncType) {
  auto& profiler = getMPSProfiler();
  const bool isGraphProfilingEnabled = profiler.isOperationProfilingEnabled();

  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      if (isGraphProfilingEnabled) {
        // this function call is only relevant for interval-based Signposts
        // which exclude schedule time (only includes GPU run time)
        profiler.beginProfileGPUInterval(mpsGraph);
      }
      // note: CommitAndContinue feature is enabled/disabled via "_executionDescriptor"
      [mpsGraph encodeToCommandBuffer:commandBuffer()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:results
                  executionDescriptor:_executionDescriptor];

      // Track this operation in LRU cache (persists across flushes for performance)
      // Use MPSGraph pointer as operation signature
      uintptr_t opSignature = reinterpret_cast<uintptr_t>(mpsGraph);
      if (!isOperationCached(opSignature)) {
        // New unique operation - add to cache
        addOperationToCache(opSignature);
      }
      // If operation was already cached, it's moved to front of LRU automatically

      // Increment operation count for flush threshold
      _commandBufferOperationCount++;

      SyncType _syncType = syncType;
      // if commitAndContinue is disabled, we need to always commit manually after encoding
      if (!_enableCommitAndContinue && syncType != SyncType::COMMIT_AND_WAIT) {
        _syncType = SyncType::COMMIT;
      }

      // check if graph execution profiling is enabled
      if (isGraphProfilingEnabled) {
        // with profiler enabled, we commit after adding the completedHandler in MPSProfiler
        profiler.endProfileKernel(mpsGraph, _syncType);
      } else {
        synchronize(_syncType);
      }
    }
  });
}

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

MPSStream* MPSStreamImpl::_stream = nullptr;

MPSStream* MPSStreamImpl::getInstance() {
  if (_stream == nullptr) {
    _stream = new MPSStream(Stream(Stream::UNSAFE, c10::Device(DeviceType::MPS, 0), 0));
  }
  return _stream;
}

MPSStreamImpl::MPSStreamImpl() {}

MPSStream* getCurrentMPSStream() {
  return getDefaultMPSStream();
}

MPSStream* getDefaultMPSStream() {
  return MPSStreamImpl::getInstance();
}

} // namespace at::mps
