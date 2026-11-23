//  Copyright © 2022 Apple Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
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
@class MPSRecordingEncoder;
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

  // NOTE: After encoding work onto the command buffer, call `markWorkEncoded`.
  MPSCommandBuffer_t commandBuffer();
  // Must be called from a block running on queue(), and the returned encoder
  // is only valid for the rest of that block.
  MTLComputeCommandEncoder_t commandEncoder();
  void endKernelCoalescing();
  void synchronize(SyncType syncType);
  // Commits once enough kernels are encoded, so that the GPU runs them while the host keeps encoding
  void commitIfNeeded();
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

  bool query();

  // If work was encoded since the last query signal, encodes a new one and
  // returns true, and the caller must commit it. Otherwise, the query signal is
  // not encoded and returns false, and the caller does not need to commit.
  bool encodeQuerySignalIfNeeded();

  // Any caller that encodes work into `commandBuffer` must call this function
  // so that `query` detects the newly encoded work.
  void markWorkEncoded() {
    _hasWorkSinceQuerySignal = true;
  }

  // Graph capture (torch.accelerator.Graph, see MetalGraphImpl.mm): while a
  // capture records, ops run as usual and every Metal kernel dispatch and blit
  // on this stream is also recorded. replay() re-encodes the recorded steps
  // against the same buffers on the stream it is called on.
  struct CapturedMetalKernel {
    void* pso = nullptr; // id<MTLComputePipelineState>, retained
    struct BufferBinding {
      void* buffer; // id<MTLBuffer>
      size_t offset;
      unsigned index;
    };
    struct BytesBinding {
      std::vector<uint8_t> data;
      unsigned index;
    };
    struct ThreadgroupMemoryBinding {
      size_t length;
      unsigned index;
    };
    // useResource:usage: declares a resource accessed only indirectly (e.g. via
    // a raw GPU address embedded in an argument buffer) so Metal can track it
    // for residency/hazards; it is not implied by setBuffer/setBytes, so it
    // must be captured and reissued separately on replay.
    struct ResourceUsage {
      void* resource; // id<MTLResource>, retained
      unsigned long usage; // MTLResourceUsage bitmask
    };
    std::vector<BufferBinding> buffers;
    std::vector<BytesBinding> bytes;
    std::vector<ThreadgroupMemoryBinding> threadgroupMemory;
    std::vector<ResourceUsage> resourceUsages;
    uint64_t gridX = 0, gridY = 0, gridZ = 0;
    uint64_t tgX = 0, tgY = 0, tgZ = 0;
    bool useThreadgroups = false; // true = dispatchThreadgroups, false = dispatchThreads
  };
  // A Metal kernel dispatch, or a buffer-to-buffer blit if metalKernel is null.
  struct CapturedStep {
    std::unique_ptr<CapturedMetalKernel> metalKernel;
    void* blitSrc = nullptr; // id<MTLBuffer>, retained
    void* blitDst = nullptr; // id<MTLBuffer>, retained
    size_t blitLength = 0;
    size_t blitSrcOffset = 0;
    size_t blitDstOffset = 0;
  };
  // The recorded steps, plus every MTLBuffer they re-bind on replay, retained
  // and pinned in the allocator until releaseCapture().
  struct Capture {
    std::vector<CapturedStep> steps;
    std::vector<const void*> boundBuffers; // id<MTLBuffer>
  };

  // Starts recording and returns the capture's id, unique across streams.
  uint64_t captureBegin();
  // Stops recording `captureId`, which must be the capture this stream records.
  Capture captureEnd(uint64_t captureId);
  void replay(const Capture& capture);
  static void releaseCapture(Capture& capture);

  // The id of the capture being recorded, 0 if none.
  uint64_t activeCaptureId() const {
    return _activeCaptureId.load(std::memory_order_acquire);
  }
  bool captureMode() const {
    return activeCaptureId() != 0;
  }

  // Called by MPSRecordingEncoder to push a finalized Metal kernel recording.
  void pushCapturedMetalKernel(std::unique_ptr<CapturedMetalKernel> kernel);

  // Retains an MTLBuffer that the capture being recorded will re-bind on
  // replay and pins it in the allocator, so that once its tensor is freed it is
  // not handed to an unrelated tensor whose data replay would then read. No-op
  // when nothing is recording. Called for every buffer MPSRecordingEncoder binds
  // or declares with useResource, every recorded blit, and get_tensor_gpu_address().
  void captureNoteBuffer(MTLBuffer_t buffer);

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

  // Event and signal counter used for `query`ing the stream
  MTLSharedEvent_t _queryEvent = nullptr;
  uint64_t _querySignalCounter = 0;
  // Whether work may have been encoded since the last query signal or full sync
  bool _hasWorkSinceQuerySignal = false;

  // CommitAndContinue is enabled by default
  bool _enableCommitAndContinue = true;
  uint32_t _kernelsSinceCommit = 0;
  std::atomic<uint32_t> _commandBuffersInFlight{0};
  // Buffer that contains last raised error
  MTLBuffer_t _errorBuffer = nil;
  // First execution error reported by an asynchronously committed command buffer
  // (e.g. out of memory), rethrown by checkLastError() at the next sync point
  struct CommandBufferError {
    bool is_oom;
    int32_t code;
    std::string message;
  };
  std::mutex _commandBufferErrorMutex;
  std::optional<CommandBufferError> _commandBufferError;

  std::atomic<uint64_t> _activeCaptureId{0};
  // The capture being recorded. Queue-confined.
  Capture _recording;
  // Buffers registered by captureNoteBuffer(), which get_tensor_gpu_address() may
  // call from outside the serial queue, so it has its own lock. Moved into the Capture at captureEnd().
  std::mutex _notedBuffersMutex;
  std::unordered_set<const void*> _notedBuffers;
  // Sets _activeCaptureId, keeping the process-global recording count that
  // isAnyStreamCapturing() reads in sync. 0 stops recording. Queue-confined.
  void setActiveCaptureId(uint64_t id);
#ifdef __OBJC__
  MPSRecordingEncoder* _recordingEncoder = nil;
#else
  void* _recordingEncoder = nullptr;
#endif

  // use synchronize() to access any of these commit functions outside MPSStream
  void commit();
  void commitAndWait();
  void commitAndContinue();
  void flush();
  // records execution errors of the current command buffer once it completes
  void addErrorHandler();
  void recordCommandBufferError(CommandBufferError error);
};

/**
 * Get the current MPS stream for this thread. Returns the default stream if no
 * other stream has been set with `setCurrentMPSStream()`.
 */
TORCH_API MPSStream* getCurrentMPSStream();

/**
 * Set the current MPS stream for this thread. Kernels that call
 * `getCurrentMPSStream()` will enqueue their work onto this stream. Passing
 * nullptr sets to the default stream.
 */
TORCH_API void setCurrentMPSStream(MPSStream* stream);

/**
 * Get the default MPS stream
 */
TORCH_API MPSStream* getDefaultMPSStream();

/**
 * True if any MPS stream is currently recording a graph capture. Reading it
 * never creates a stream, so unlike `getCurrentMPSStream()->captureMode()` it can
 * be used as a probe before MPS has been initialized. No stream can be recording
 * unless one already exists, so a false result is authoritative.
 */
TORCH_API bool isAnyStreamCapturing();

/**
 * True if the current stream is recording a graph capture.
 */
TORCH_API bool isCurrentStreamCapturing();

/**
 * Raises if the current stream is recording a graph capture. For work that
 * replay cannot reproduce (CPU fallbacks, MPSGraph and MPS-framework kernels that
 * bypass MPSRecordingEncoder, RNG); `attempt` names it, as in
 * at::cuda::assertNotCapturing.
 */
TORCH_API void assertNotCapturing(std::string_view attempt);

/**
 * Get a stream from the pool. There are 32 streams in the pool which live for
 * the lifetime of a process. The stream returned by this function is chosen
 * in round-robin order. Note: The default stream is not in the pool.
 */
TORCH_API MPSStream* getStreamFromPool();

/**
 * Get a stream by its ID. 0 is the default stream, 1 - 32 are pool streams.
 */
TORCH_API MPSStream* getStreamByID(int64_t stream_id);

/**
 * Synchronize the default stream and any pool streams created so far.
 */
TORCH_API void synchronizeAllMPSStreams(SyncType syncType);

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
