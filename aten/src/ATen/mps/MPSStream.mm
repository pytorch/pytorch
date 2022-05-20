//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSStream.h>

namespace at {
namespace mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

MPSStream::MPSStream(Stream stream) : _stream(stream) {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  TORCH_CHECK(_stream.device_type() == DeviceType::MPS);
  _serialQueue = dispatch_queue_create("metal gpu stream", NULL);
}

MPSStream::~MPSStream() {
  [_commandQueue autorelease];
  _commandQueue = nil;

  assert(_commandBuffer == nil);
}

id<MTLCommandBuffer> MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer =
        [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }

  return _commandBuffer;
}

void MPSStream::synchronize() {
  dispatch_sync(queue(), ^() {
    @autoreleasepool {
      commandBuffer();
      commitAndWait();
    }
  });
}

void MPSStream::commit(bool doFlush) {
  if (doFlush) {
    flush();
  }
}

void MPSStream::commitAndWait() {
  assert(_commandBuffer);
  [_commandBuffer commit];
  [_commandBuffer waitUntilCompleted];
  [_commandBuffer release];
  _commandBuffer = nil;
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
