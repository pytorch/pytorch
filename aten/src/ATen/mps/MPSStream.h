//  Copyright © 2022 Apple Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/mps/MPSDevice.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#ifdef __OBJC__
// Apple framework headers emit deprecation warnings from CarbonCore and
// missing-attribute warnings from MPSGraph on recent macOS SDKs.
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wobjc-property-no-attribute")
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()
typedef MPSCommandBuffer* MPSCommandBuffer_t;
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef id<MTLSharedEvent> MTLSharedEvent_t;
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLBuffer> MTLBuffer_t;
#else
#include <dispatch/dispatch.h>
typedef void* MPSCommandBuffer_t;
typedef void* MPSGraph;
typedef void* MPSGraphExecutionDescriptor;
typedef void* MPSGraphCompilationDescriptor;
typedef void* MTLCommandQueue_t;
typedef void* MTLComputeCommandEncoder_t;
typedef void* MTLSharedEvent_t;
typedef void* MTLDevice_t;
typedef void* MTLBuffer_t;
typedef void* MTLCommandBufferHandler;
typedef void* NSDictionary;
#define nil NULL
#endif

namespace at::mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

enum class SyncType {
  NONE, // no commit to command buffer
  COMMIT, // commit and flush the command buffer
  COMMIT_AND_WAIT, // flush and wait for command buffer execution to finish
  COMMIT_AND_CONTINUE, // commit and continue with a new underlying command buffer
  COMMIT_ADAPTIVE, // commit adaptively based on available memory
};

class TORCH_API MPSStream {
 public:
  enum Unchecked { UNCHECKED };

  /// Construct a MPSStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a MPS stream.
  explicit MPSStream(Stream stream);

  ~MPSStream();

  MTLCommandQueue_t commandQueue() const {
    return _commandQueue;
  }

  dispatch_queue_t queue() const {
    return _serialQueue;
  }

  MPSCommandBuffer_t commandBuffer();
  MTLComputeCommandEncoder_t commandEncoder();
  void endKernelCoalescing();
  void synchronize(SyncType syncType);
  void copy(MTLBuffer_t srcBuffer,
            MTLBuffer_t dstBuffer,
            size_t length,
            size_t srcOffset,
            size_t dstOffset,
            uint64_t profileId,
            SyncType syncType = SyncType::NONE);
  void copy_and_sync(MTLBuffer_t srcBuffer,
                     MTLBuffer_t dstBuffer,
                     size_t length,
                     size_t srcOffset,
                     size_t dstOffset,
                     bool non_blocking,
                     uint64_t profileId);
  void executeMPSGraph(MPSGraph* mpsGraph,
                       NSDictionary* feeds,
                       NSDictionary* results,
                       SyncType syncType = SyncType::NONE);
  void addCompletedHandler(MTLCommandBufferHandler block);

  // Graph capture: record a sequence of MPSGraph ops on the first pass and
  // replay them all in a single dispatch_sync on subsequent passes.
  // Constraints (same as torch.cuda.graph):
  //   - inputs must be updated in-place via .copy_() between replay calls
  //   - tensor shapes and allocations must not change between replays
  //   - profiling must be disabled during capture
  void captureBegin();
  void captureEnd();
  void captureReset();
  void replay();

  // Returns true if a capture is currently in progress.
  // _captureMode is std::atomic<bool> so this is safe to call from any thread.
  bool captureMode() const {
    return _captureMode.load(std::memory_order_acquire);
  }

  size_t capturedStepCount() const {
    return _capturedSteps.size();
  }

  /// Get the MPS device index that this stream is associated with.
  c10::DeviceIndex device_index() const {
    return _stream.device_index();
  }

  MTLCommandQueue_t stream() const {
    return _commandQueue;
  }

  MTLDevice_t device() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const {
    return _stream;
  }

  MTLBuffer_t getErrorBuffer();
  void checkLastError();

 private:
  Stream _stream;
  MTLCommandQueue_t _commandQueue = nil;
  MPSCommandBuffer_t _commandBuffer = nil;
  MPSCommandBuffer_t _prevCommandBuffer = nil;
  MTLComputeCommandEncoder_t _commandEncoder = nil;
  MPSGraphExecutionDescriptor* _executionDescriptor = nil;
  MPSGraphCompilationDescriptor* _compilationDescriptor = nil;
  dispatch_queue_t _serialQueue = nullptr;
  // CommitAndContinue is enabled by default
  bool _enableCommitAndContinue = true;
  // Buffer that contains last raised error
  MTLBuffer_t _errorBuffer = nil;
  // Cache of compiled MPSGraphExecutable objects keyed by MPSGraph pointer.
  // Populated lazily on first executeMPSGraph call per graph; never invalidated
  // because each MPSGraph* in MPSGraphCache is shape-specific.
  std::unordered_map<uintptr_t, void*> _graphExecutableCache;

  // Graph capture state.
  // _capturedSteps stores one entry per executeMPSGraph call recorded during
  // a capture pass. inputsArray/resultsArray are retained ObjC arrays of
  // MPSGraphTensorData* wrapping the MTLBuffers of the captured tensors.
  // On replay those same buffers are re-bound — callers must update input
  // data in-place (via .copy_()) between replay calls to supply new batches.
  struct CapturedStep {
    void* exe = nullptr; // MPSGraphExecutable*, borrowed from _graphExecutableCache
#ifdef __OBJC__
    NSArray<MPSGraphTensorData*>* inputsArray = nil;
    NSArray<MPSGraphTensorData*>* resultsArray = nil;
#else
    void* inputsArray = nullptr;
    void* resultsArray = nullptr;
#endif
  };
  std::atomic<bool> _captureMode{false};
  std::vector<CapturedStep> _capturedSteps;

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

class TORCH_API MPSStreamImpl {
 public:
  /**
   * Gets single instance of the MPSStream.
   */
  static MPSStream* getInstance();

 private:
  static MPSStream* _stream;
  MPSStreamImpl();
};

#ifdef __OBJC__
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());
#endif
} // namespace at::mps
