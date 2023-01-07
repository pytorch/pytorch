//  Copyright © 2022 Apple Inc.

#pragma once

#include <cstdint>
#include <utility>

#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>
#include <c10/core/Stream.h>
#include <ATen/mps/MPSDevice.h>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLCommandBuffer> MTLCommandBuffer_t;
typedef id<MTLSharedEvent> MTLSharedEvent_t;
typedef id<MTLDevice> MTLDevice_t;
#else
typedef void* MTLCommandQueue_t;
typedef void* MTLCommandQueue;
typedef void* MTLCommandBuffer_t;
typedef void* MTLCommandBuffer;
typedef void* MTLSharedEvent_t;
typedef void* dispatch_queue_t;
typedef void* MTLDevice_t;
#define nil NULL;
#endif


namespace at {
namespace mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

enum class SyncType {
  NONE,               // no commit to command buffer
  COMMIT,             // commit and flush the command buffer
  COMMIT_AND_WAIT,    // flush and wait for command buffer execution to finish
  COMMIT_AND_CONTINUE,// commit and continue with a new underlying command buffer
  COMMIT_ADAPTIVE,    // commit adaptively based on available memory
};

class TORCH_API MPSStream
{
public:
  enum Unchecked { UNCHECKED };

  /// Construct a MPSStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a MPS stream.
  explicit MPSStream(Stream stream);

  ~MPSStream();
  MTLCommandQueue_t commandQueue() const { return _commandQueue; };
  dispatch_queue_t queue() const { return _serialQueue; }

  MPSCommandBuffer* commandBuffer();
  void commit(bool flush);
  void commitAndWait();
  void commitAndContinue();
  void synchronize(SyncType syncType);
  void fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType = SyncType::NONE);
  void copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
            size_t length, size_t srcOffset, size_t dstOffset, SyncType syncType = SyncType::NONE);
  void copy_and_sync(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
                     size_t length, size_t srcOffset, size_t dstOffset, bool non_blocking);
  void flush();
  void executeMPSGraph(MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results, SyncType syncType = SyncType::NONE);
  void addCompletedHandler(MTLCommandBufferHandler block);

  /// Get the MPS device index that this stream is associated with.
  c10::DeviceIndex device_index() const { return _stream.device_index(); }

  MTLCommandQueue_t stream() const { return _commandQueue; };

  MTLDevice_t device() const { return [_commandQueue device];}

  /// Explicit conversion to Stream.
  Stream unwrap() const { return _stream; }

private:
  Stream _stream;
  MTLCommandQueue_t   _commandQueue = nil;
  MPSCommandBuffer*  _commandBuffer = nil;
  MPSGraphExecutionDescriptor *_executionDescriptor = nil;
  void _flush(bool commitAndWait) const;

  dispatch_queue_t    _serialQueue = nullptr;
};

/**
 * Get the current MPS stream
 */
TORCH_API MPSStream* getCurrentMPSStream();

/**
 * Get the default MPS stream
 */
TORCH_API MPSStream* getDefaultMPSStream();

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

class TORCH_API MPSStreamImpl
{
 public:
  /**
   * Gets single instance of the MPSStream.
   */
  static MPSStream* getInstance();

 private:
  static MPSStream* _stream;
  MPSStreamImpl();
};


//-----------------------------------------------------------------
//  MPSEvent
//-----------------------------------------------------------------

struct TORCH_API MPSEvent
{
  // for a new instance of MPSEvent, sometimes we want an empty shell and don't
  // necessarily want to create events or listeners. So we defer initialization
  // until we actually use the event (e.g., record, notify, etc.)
  MPSEvent(bool deferInitialization = true);
  ~MPSEvent();
  MTLSharedEvent_t event() const {return _event; }

  void recordEvent(bool syncEvent = false);
  void waitForEvent(bool syncEvent = false); // waits on the cpu
  void notifyEvent(MTLSharedEventNotificationBlock block);
  bool queryEvent() const;
  uint64_t getCurrentValue() const { return _signalCounter; }
  void setCurrentValue(uint64_t currValue) { _signalCounter = currValue; }
private:
  bool is_initialized;
  uint64_t _signalCounter;
  MPSStream* _stream;
  MTLSharedEvent_t _event;
  MTLSharedEventListener* _listener;

  void initialize();
};

typedef MPSEvent* mpsEvent_t;


} // namespace mps
} // namespace at
