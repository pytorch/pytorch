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
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef id<MTLSharedEvent> MTLSharedEvent_t;
typedef id<MTLDevice> MTLDevice_t;
#else
typedef void* MTLCommandQueue_t;
typedef void* MTLCommandQueue;
typedef void* MTLCommandBuffer_t;
typedef void* MTLCommandBuffer;
typedef void* MTLComputeCommandEncoder_t;
typedef void* MTLSharedEvent_t;
typedef void* dispatch_queue_t;
typedef void* MTLDevice_t;
#define nil NULL;
#endif


namespace at::mps {

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
  MTLComputeCommandEncoder_t commandEncoder();
  void endKernelCoalescing();
  void synchronize(SyncType syncType);
  void fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType = SyncType::NONE);
  void copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
            size_t length, size_t srcOffset, size_t dstOffset,
            uint64_t profileId, SyncType syncType = SyncType::NONE);
  void copy_and_sync(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
                     size_t length, size_t srcOffset, size_t dstOffset,
                     bool non_blocking, uint64_t profileId);
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
  MTLCommandQueue_t _commandQueue = nil;
  MPSCommandBuffer* _commandBuffer = nil;
  MPSCommandBuffer* _prevCommandBuffer = nil;
  MTLComputeCommandEncoder_t _commandEncoder = nil;
  MPSGraphExecutionDescriptor *_executionDescriptor = nil;
  MPSGraphCompilationDescriptor *_compilationDescriptor = nil;
  dispatch_queue_t _serialQueue = nullptr;
  // CommitAndContinue is enabled by default
  bool _enableCommitAndContinue = true;

  // use synchronize() to access any of these commit functions outside MPSStream
  void commit();
  void commitAndWait();
  void commitAndContinue();
  void flush();
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

} // namespace at::mps
