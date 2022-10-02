//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSStream.h>

namespace at {
namespace mps {

#define USE_MPSCOMMANDBUFFER 1

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

MPSStream::MPSStream(Stream stream) : _stream(stream) {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  TORCH_CHECK(_stream.device_type() == DeviceType::MPS);
  _serialQueue = dispatch_queue_create("metal gpu stream", nullptr);
  _executionDescriptor = [MPSGraphExecutionDescriptor new];
  _executionDescriptor.completionHandler = ^(NSDictionary<MPSGraphTensor *,
                                             MPSGraphTensorData *> * resultsDictionary,
                                             NSError * _Nullable error) { };
}

MPSStream::~MPSStream() {
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];

  assert(_commandBuffer == nil);
}

MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }

  return _commandBuffer;
}

void MPSStream::synchronize(SyncType syncType) {
  if (!_commandBuffer)
    return;
  switch(syncType) {
    case SyncType::NONE:
      // typically in GPU to GPU copies we won't commit explicitly
      break;
    case SyncType::COMMIT:
      flush();
      break;
    case SyncType::COMMIT_AND_WAIT:
      commitAndWait();
      break;
    case SyncType::COMMIT_AND_CONTINUE:
      commitAndContinue();
      break;
  }
}

void MPSStream::commit(bool doFlush) {
#if USE_MPSCOMMANDBUFFER
  [commandBuffer() commitAndContinue];
#else
  if (doFlush) {
    flush();
  }
#endif
}

void MPSStream::commitAndWait() {
  assert(_commandBuffer);
  [_commandBuffer commit];
  [_commandBuffer waitUntilCompleted];
  [_commandBuffer release];
  _commandBuffer = nil;
}

void MPSStream::commitAndContinue() {
  assert(_commandBuffer);
  [_commandBuffer commitAndContinue];
}

void MPSStream::flush() {
  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer release];
    _commandBuffer = nil;
  }
}

void MPSStream::_flush(bool commitAndWait) const {
  assert(_commandBuffer);
  [_commandBuffer commit];
  if (commitAndWait) {
    [_commandBuffer waitUntilCompleted];
  }
  [_commandBuffer release];
}

void MPSStream::fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType)
{
  TORCH_INTERNAL_ASSERT(length >= offset);
  if (length == 0) return;
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      [blitEncoder fillBuffer:buffer
                        range:NSMakeRange(offset, length)
                        value:value];
      [blitEncoder endEncoding];
      synchronize(syncType);
    }
  });
}

void MPSStream::copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
                    size_t length, size_t srcOffset, size_t dstOffset, SyncType syncType) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      [blitEncoder copyFromBuffer:srcBuffer
                     sourceOffset:(NSUInteger)srcOffset
                         toBuffer:dstBuffer
                destinationOffset:(NSUInteger)dstOffset
                             size:(NSUInteger)length];
      [blitEncoder endEncoding];
      synchronize(syncType);
    }
  });
}

void MPSStream::copy_and_sync(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer, size_t length,
                              size_t srcOffset, size_t dstOffset, bool non_blocking) {
  copy(srcBuffer, dstBuffer, length, srcOffset, dstOffset,
       !non_blocking ? SyncType::COMMIT_AND_WAIT : SyncType::COMMIT);
}

void MPSStream::executeMPSGraph(MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results, SyncType syncType) {
  dispatch_sync(_serialQueue, ^() {
#if USE_MPSCOMMANDBUFFER
    [mpsGraph encodeToCommandBuffer:commandBuffer()
                              feeds:feeds
                   targetOperations:nil
                  resultsDictionary:results
                executionDescriptor:_executionDescriptor];
    // mostly the syncType is NONE, but in some cases we may want to sync and wait (e.g., gatherViewTensor)
    synchronize(syncType);
#else
    commit(true);
    [mpsGraph runAsyncWithMTLCommandQueue:_commandQueue
                                    feeds:feeds
                         targetOperations:nil
                        resultsDictionary:results
                      executionDescriptor:_executionDescriptor];
#endif
 });
}

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

MPSStream* MPSStreamImpl::_stream = nullptr;

MPSStream* MPSStreamImpl::getInstance() {
  if (_stream == nullptr) {
    _stream =
        new MPSStream(Stream(Stream::UNSAFE, c10::Device(DeviceType::MPS), 0));
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

//-----------------------------------------------------------------
//  MPSEvent
//-----------------------------------------------------------------

MPSEvent::MPSEvent() {
  _event = [MPSDevice::getInstance()->device() newSharedEvent];
}

MPSEvent::~MPSEvent() {
  [_event release];
  _event = nil;
}

void MPSEvent::recordEvent(MPSStream* stream) {
  @autoreleasepool {
    _isRecorded = true;
    dispatch_sync(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
        [commandBuffer encodeSignalEvent:_event value:_currentValue];
        stream->commit(true);
      }
    });
  }
}

void MPSEvent::waitForEvent(MPSStream* stream) {
  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
      [commandBuffer encodeWaitForEvent:_event value:_currentValue];
      stream->commit(false);
    }
  });
}

bool MPSEvent::queryEvent() {
  return !_isRecorded || (_event.signaledValue >= _currentValue);
}

} // namespace mps
} // namespace at
