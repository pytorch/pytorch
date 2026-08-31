//  Copyright © 2022 Apple Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ATen/mps/MPSDevice.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Storage.h>
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
  // Multiple independent captures may be alive at once, each identified by
  // the handle captureBegin() returns; recording itself is exclusive (only
  // one capture may be actively recording at a time).
  // Constraints (same as torch.cuda.graph):
  //   - inputs must be updated in-place via .copy_() between replay calls
  //   - tensor shapes and allocations must not change between replays
  //   - profiling must be disabled during capture
  uint64_t captureBegin();
  // Takes the id so one capture cannot end another's recording.
  void captureEnd(uint64_t captureId);
  // Stops recording only if `captureId` is the capture currently recording, and
  // reports whether it was. Lets a destructor tear down a graph dropped
  // mid-recording without using exceptions to ask "am I the recorder?", which
  // cannot distinguish that from a genuine failure.
  bool captureEndIfRecording(uint64_t captureId);
  // Releases the capture identified by `captureId` (retained buffers/executables).
  // Returns false if it was not live, so a destructor can distinguish an
  // already-freed handle from a real failure without a separate lookup.
  bool captureFree(uint64_t captureId);
  void replay(uint64_t captureId);

  // Returns true if a capture is currently being recorded.
  // _activeCaptureId is std::atomic<uint64_t> so this is safe to call from any thread.
  bool captureMode() const {
    return _activeCaptureId.load(std::memory_order_acquire) != 0;
  }

  // Fail loud when an op that cannot be captured runs inside a capture block.
  // Our own Metal shaders are recorded, since MPSRecordingEncoder intercepts the
  // encoder calls that issue them. Ops that instead hand encoding to an
  // MPS-framework kernel (e.g. MPSMatrix*/MPSNDArray* via
  // encodeToCommandEncoder:/encodeToCommandBuffer:) drive the encoder through
  // selectors the proxy does not override, so the work runs but is never recorded;
  // the same applies to ops that fall back to CPU. Replay would silently drop
  // them and produce wrong results, the worst failure mode for users, so such ops
  // call this to raise a clear error instead.
  void assertCapturable(const char* op) const {
    TORCH_CHECK(!captureMode(),
                op,
                " is not supported inside a torch.mps.metal_graph() capture: it uses a path "
                "(opaque MPS-framework kernel encode or CPU fallback) that cannot be recorded for "
                "replay. Run it outside the capture block.");
  }

  // Returns 0 if `captureId` does not refer to a live capture (never captured, or
  // already freed). Queue-confined: every mutation of _captures happens on
  // _serialQueue, so the read must too.
  size_t capturedStepCount(uint64_t captureId) const;

  // Owning handle for one capture, so captured resources are released when the
  // owner goes out of scope rather than only on an explicit free. This is what
  // backs torch.mps.MetalGraph and gives it the same ownership semantics as
  // torch.cuda.CUDAGraph: dropping the object releases the capture.
  //
  // The stream is captured at captureBegin() time and reused for replay and
  // release, so a graph recorded on one stream is never replayed against a
  // different stream's capture table.
  class TORCH_API MetalGraph {
   public:
    MetalGraph() = default;
    ~MetalGraph() {
      reset();
    }
    // Non-copyable: two owners of one capture id would double-free it.
    MetalGraph(const MetalGraph&) = delete;
    MetalGraph& operator=(const MetalGraph&) = delete;
    MetalGraph(MetalGraph&& other) noexcept : _id(other._id), _stream(other._stream) {
      other._id = 0;
      other._stream = nullptr;
    }
    MetalGraph& operator=(MetalGraph&& other) noexcept {
      if (this != &other) {
        reset();
        _id = other._id;
        _stream = other._stream;
        other._id = 0;
        other._stream = nullptr;
      }
      return *this;
    }

    void captureBegin();
    void captureEnd();
    void replay();
    // Releases the capture. Safe to call more than once, and on a graph that
    // never captured anything.
    void reset();
    size_t stepCount() const;
    bool isCaptured() const {
      return _id != 0;
    }

   private:
    uint64_t _id = 0; // 0 = nothing captured
    MPSStream* _stream = nullptr;
  };

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

  // Called by MPSRecordingEncoder to push a finalized Metal kernel recording.
  void pushCapturedMetalKernel(std::unique_ptr<CapturedMetalKernel> kernel);

  // Retains an MTLBuffer that the capture currently recording will re-bind on
  // replay, and holds it for the life of that capture. No-op when nothing is
  // recording.
  //
  // Bindings are recorded by buffer address, so the buffer has to stay both
  // alive and owned by the tensor it was recorded for. Retaining it covers the
  // first part; pinning it in the allocator (pinBufferForCapture) covers the
  // second, by keeping the buffer out of the reuse pool for as long as this
  // capture is alive. Without that pair, a freed tensor's buffer would be handed
  // to an unrelated tensor and replay would read its data.
  //
  // executeMPSGraph() cannot recover the buffers behind its feeds/results
  // dictionaries, so ops register them as they wrap them (see OperationUtils.mm).
  void captureNoteBuffer(MTLBuffer_t buffer);

  // Hands the capture currently recording a reference to host memory that one of
  // its recorded blits reads or writes. No-op when nothing is recording.
  //
  // A CPU tensor's pages are only borrowed by the MTLBuffer wrapping them
  // (newBufferWithBytesNoCopy), so retaining that buffer is not enough to keep
  // the pages alive for a replay. The blit's own deallocator cannot hold the
  // storage either: it runs on Metal's completion thread, where dropping the
  // last reference to a CPU tensor would need the GIL. The capture holds it
  // instead and drops it from captureFree(), on the calling thread.
  void captureRetainStorage(const c10::Storage& storage);

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

  // Graph capture state.
  // Each entry in _captures stores one vector of steps (one per
  // executeMPSGraph call OR raw Metal kernel dispatch) recorded during a
  // capture pass, keyed by the handle captureBegin() returned for it.
  // On replay the same buffers are re-bound: callers must update input
  // data in-place (via .copy_()) between replay calls to supply new batches.

  struct CapturedStep {
    enum class Kind { MPSGraph, MetalKernel, BlitCopy };
    Kind kind = Kind::MPSGraph;
    // MPSGraph fields
    void* exe = nullptr; // MPSGraphExecutable*, owned by this step (released in releaseCapturedStep)
#ifdef __OBJC__
    NSArray<MPSGraphTensorData*>* inputsArray = nil;
    NSArray<MPSGraphTensorData*>* resultsArray = nil;
#else
    void* inputsArray = nullptr;
    void* resultsArray = nullptr;
#endif
    // Metal kernel fields
    std::unique_ptr<CapturedMetalKernel> metalKernel;
    // Blit copy fields (buffer-to-buffer copy recorded from MPSStream::copy).
    void* blitSrc = nullptr; // id<MTLBuffer>, retained
    void* blitDst = nullptr; // id<MTLBuffer>, retained
    size_t blitLength = 0;
    size_t blitSrcOffset = 0;
    size_t blitDstOffset = 0;
  };
  std::atomic<uint64_t> _activeCaptureId{0}; // 0 = no capture currently recording

  // One recorded capture: its steps, plus every MTLBuffer those steps re-bind on
  // replay, retained for the life of the capture.
  struct Capture {
    std::vector<CapturedStep> steps;
    std::vector<const void*> boundBuffers; // id<MTLBuffer>, retained
    std::vector<c10::Storage> hostStorages; // host pages a recorded blit touches
  };
  std::unordered_map<uint64_t, Capture> _captures;
  // Buffers registered by captureNoteBuffer() for the capture being recorded.
  // Ops register from outside the serial queue, so this has its own lock; it is
  // moved into the Capture at captureEnd(). Recording is exclusive, so one
  // pending set per stream is enough.
  std::mutex _notedBuffersMutex;
  std::unordered_set<const void*> _notedBuffers;
  std::vector<c10::Storage> _notedStorages;
  uint64_t _nextCaptureId = 1; // allocated only inside _serialQueue dispatches

  // Retains and clears the noted-buffer set into `capture`.
  void takeNotedBuffers(Capture& capture);
  // Drops the noted-buffer set without handing it to a capture.
  void discardNotedBuffers();
  // Releases the buffers a capture retained.
  static void releaseCaptureBuffers(Capture& capture);

  // Release retained Objective-C refs held by a captured step (PSO + bound
  // MTLBuffers for MetalKernel steps, inputs/results arrays for MPSGraph steps).
  static void releaseCapturedStep(CapturedStep& step);
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
 * True if any MPS stream is currently recording a MetalGraph capture. Reading it
 * never creates a stream, so unlike `getCurrentMPSStream()->captureMode()` it can
 * be used as a probe before MPS has been initialized. No stream can be recording
 * unless one already exists, so a false result is authoritative.
 */
TORCH_API bool isAnyStreamCapturing();

/**
 * Get a stream from the pool. There are 32 streams in the pool which live for
 * the lifetime of a process. The stream returned by this function is chosen
 * in round-robin order. Note: The default stream is not in the pool.
 */
TORCH_API MPSStream* getStreamFromPool();

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
