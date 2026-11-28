//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#import <ATen/mps/MPSRecordingEncoder.h>
#include <ATen/mps/MPSStream.h>
#include <c10/metal/error.h>
#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>

#include <array>
#include <atomic>

@interface MPSGraphExecutionDescriptor ()
@property(readwrite, atomic) BOOL enableCommitAndContinue;
@end

@interface MPSStreamCommandBuffer : MPSCommandBuffer
@property(nonatomic) std::function<void()> willCommitAndContinue;
@end
@implementation MPSStreamCommandBuffer
- (void)commitAndContinue {
  _willCommitAndContinue();
  [super commitAndContinue];
}
@end

namespace at::mps {
namespace {
// Returns true if the command buffer failed to execute (e.g. was aborted by the driver)
bool commandBufferFailed(id<MTLCommandBuffer> cb, bool& is_oom, int32_t& code, std::string& message) {
  if (cb.status != MTLCommandBufferStatusError) {
    return false;
  }
  NSError* error = cb.error;
  is_oom = [error.domain isEqualToString:MTLCommandBufferErrorDomain] && error.code == MTLCommandBufferErrorOutOfMemory;
  code = error ? static_cast<int32_t>(error.code) : 0;
  message = error ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
  return true;
}
} // namespace

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

namespace {
// Number of streams currently recording a capture. Lets isAnyStreamCapturing()
// answer without touching (and thereby creating) a stream.
std::atomic<int> recording_stream_count{0};
// Process-wide, so a capture id names its capture across streams.
std::atomic<uint64_t> next_capture_id{1};
} // namespace

bool isAnyStreamCapturing() {
  return recording_stream_count.load(std::memory_order_acquire) != 0;
}

bool isCurrentStreamCapturing() {
  return isAnyStreamCapturing() && getCurrentMPSStream()->captureMode();
}

void assertNotCapturing(std::string_view attempt) {
  TORCH_CHECK(!isCurrentStreamCapturing(),
              attempt,
              " during MPS graph capture: replay could not reproduce it. Run it outside the capture.");
}

MPSStream::MPSStream(Stream stream) : _stream(stream) {
  _queryEvent = [MPSDevice::getInstance()->device() newSharedEvent];
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

  _errorBuffer = [MPSDevice::getInstance()->device() newBufferWithLength:sizeof(c10::metal::ErrorMessages)
                                                                 options:MTLResourceStorageModeShared];
  std::memset([_errorBuffer contents], 0, 1024);
}

MPSStream::~MPSStream() {
  [_queryEvent release];
  _queryEvent = nil;
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];
  [_compilationDescriptor release];
  _executionDescriptor = nil;
  [_errorBuffer release];
  _errorBuffer = nil;
  _compilationDescriptor = nil;

  assert(_commandBuffer == nil);
}

MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    auto cb = [[MPSStreamCommandBuffer alloc] initWithCommandBuffer:[_commandQueue commandBuffer]];
    cb.willCommitAndContinue = [this] { addErrorHandler(); };
    _commandBuffer = cb;
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
  ++_kernelsSinceCommit;
  markWorkEncoded();

  if (_activeCaptureId.load(std::memory_order_acquire) != 0) {
    if (!_recordingEncoder) {
      _recordingEncoder = [[MPSRecordingEncoder alloc] initWithEncoder:_commandEncoder stream:this];
    }
    return (id<MTLComputeCommandEncoder>)_recordingEncoder;
  }
  return _commandEncoder;
}

void MPSStream::commitIfNeeded() {
  // Encoded kernels only start running once their command buffer is committed, so commit every kKernelsPerCommit
  // of them. Metal blocks the creation of a command buffer while its queue has 64 uncompleted ones, which would
  // stall the caller, so skip the commit while kMaxCommandBuffersInFlight of the stream's own commits are in
  // flight (MPSGraph's are not counted).
  constexpr uint32_t kKernelsPerCommit = 16; // See https://github.com/pytorch/pytorch/pull/200181 for the sweep
  constexpr uint32_t kMaxCommandBuffersInFlight = 32;
  if (_enableCommitAndContinue && _kernelsSinceCommit >= kKernelsPerCommit &&
      _commandBuffersInFlight < kMaxCommandBuffersInFlight) {
    synchronize(SyncType::COMMIT);
  }
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
  _kernelsSinceCommit = 0;
  if (_enableCommitAndContinue) {
    [commandBuffer() commitAndContinue];
  } else {
    flush();
  }
}

void MPSStream::commitAndWait() {
  _kernelsSinceCommit = 0;
  if (_prevCommandBuffer) {
    // the previous command buffer (if exists) has already been committed,
    // so we just wait until it's completed and then dispose it.
    [_prevCommandBuffer waitUntilCompleted];
    CommandBufferError error;
    if (commandBufferFailed(_prevCommandBuffer, error.is_oom, error.code, error.message)) {
      recordCommandBufferError(std::move(error));
    }
    [_prevCommandBuffer release];
    _prevCommandBuffer = nil;
    checkLastError();
  }

  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    // check the status directly, as the completed handlers may not have run yet
    CommandBufferError error;
    if (commandBufferFailed(_commandBuffer, error.is_oom, error.code, error.message)) {
      recordCommandBufferError(std::move(error));
    }
    [_commandBuffer release];
    _commandBuffer = nil;
    checkLastError();
  }
  _hasWorkSinceQuerySignal = false;
}

void MPSStream::commitAndContinue() {
  assert(_commandBuffer);
  _kernelsSinceCommit = 0;
  [_commandBuffer commitAndContinue];
}

void MPSStream::endKernelCoalescing() {
  if (_commandEncoder) {
    [_commandEncoder endEncoding];
    [_commandEncoder release];
    _commandEncoder = nil;
    // Recording encoder wraps the now-stale inner encoder; release it.
    [_recordingEncoder release];
    _recordingEncoder = nil;
  }
}

void MPSStream::flush() {
  if (_commandBuffer) {
    addErrorHandler();
    [_commandBuffer commit];
    // if commitAndContinue is disabled (e.g., for Profiler), we keep the command
    // buffer so we could wait on it later, if required.
    if (!_enableCommitAndContinue) {
      _prevCommandBuffer = _commandBuffer;
    } else {
      [_commandBuffer release];
    }
    _commandBuffer = nil;
  }
}

void MPSStream::addErrorHandler() {
  // Metal reports execution errors (e.g. when the resources referenced by a command buffer
  // exceed the working set limit) only once the command buffer completes, and it skips
  // the whole command buffer, leaving its outputs unwritten. Record the first such error
  // so that checkLastError() raises it at the next synchronization point, rather than
  // silently returning garbage, similar to how CUDA reports asynchronous errors.
  ++_commandBuffersInFlight;
  [commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    --_commandBuffersInFlight;
    CommandBufferError error;
    if (commandBufferFailed(cb, error.is_oom, error.code, error.message)) {
      recordCommandBufferError(std::move(error));
    }
  }];
}

void MPSStream::recordCommandBufferError(CommandBufferError error) {
  std::lock_guard<std::mutex> lock(_commandBufferErrorMutex);
  // keep the first error, as the subsequent ones are likely caused by it
  if (!_commandBufferError) {
    _commandBufferError = std::move(error);
  }
}

void MPSStream::addCompletedHandler(MTLCommandBufferHandler block) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      [commandBuffer() addCompletedHandler:block];
    }
  });
}

bool MPSStream::query() {
  // Metal has no built-in way to check if there is any unfinished enqueued
  // work. The strategy here is to enqueue an event, called a "query event". If
  // the query event ever finishes before any other work has been enqueued, then
  // we know that all enqueued work has finished.

  __block bool done;
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    @autoreleasepool {
      if (encodeQuerySignalIfNeeded()) {
        // Commit now because `query` must be able to be called in a loop that
        // eventually finishes.
        synchronize(SyncType::COMMIT);
      }
      // If all the enqueued query events have signaled then all the work
      // enqueued before the event has finished.
      done = _queryEvent.signaledValue >= _querySignalCounter;
    }
  });
  return done;
}

bool MPSStream::encodeQuerySignalIfNeeded() {
  // `_hasWorkSinceQuerySignal` is set wherever work is encoded (see
  // `markWorkEncoded`). So if it is false, no new work has been enqueued.
  if (!_hasWorkSinceQuerySignal) {
    return false;
  }
  endKernelCoalescing();
  [commandBuffer() encodeSignalEvent:_queryEvent value:++_querySignalCounter];
  // Reset this flag since the newly encoded event marks the end of all
  // previously encoded work.
  _hasWorkSinceQuerySignal = false;
  return true;
}

void MPSStream::copy(id<MTLBuffer> srcBuffer,
                     id<MTLBuffer> dstBuffer,
                     size_t length,
                     size_t srcOffset,
                     size_t dstOffset,
                     uint64_t profileId,
                     SyncType syncType) {
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];
      markWorkEncoded();

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

      // Blits bypass the recording encoder, so record them here. Buffers are
      // retained here and released in releaseCapture().
      if (captureMode()) {
        CapturedStep step;
        step.blitSrc = (__bridge void*)srcBuffer;
        step.blitDst = (__bridge void*)dstBuffer;
        [srcBuffer retain];
        [dstBuffer retain];
        step.blitLength = length;
        step.blitSrcOffset = srcOffset;
        step.blitDstOffset = dstOffset;
        _recording.steps.push_back(std::move(step));
        captureNoteBuffer(srcBuffer);
        captureNoteBuffer(dstBuffer);
      }

      // profilerId has a value only if copy profiling is enabled
      if (profileId) {
        getMPSProfiler().endProfileCopy(profileId, syncType, this);
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
  // A capture records Metal kernels only.
  TORCH_CHECK(!captureMode(),
              "An op implemented with MPSGraph during MPS graph capture: replay could not reproduce it. ",
              "Run it outside the capture.");
  auto& profiler = getMPSProfiler();
  const bool isGraphProfilingEnabled = profiler.isOperationProfilingEnabled();

  dispatch_sync_with_rethrow(_serialQueue, ^() {
    endKernelCoalescing();
    if (isGraphProfilingEnabled) {
      // this function call is only relevant for interval-based Signposts
      // which exclude schedule time (only includes GPU run time)
      profiler.beginProfileGPUInterval(mpsGraph, this);
    }
    // note: CommitAndContinue feature is enabled/disabled via "_executionDescriptor"
    [mpsGraph encodeToCommandBuffer:commandBuffer()
                              feeds:feeds
                   targetOperations:nil
                  resultsDictionary:results
                executionDescriptor:_executionDescriptor];
    markWorkEncoded();

    SyncType _syncType = syncType;
    // if commitAndContinue is disabled, we need to always commit manually after encoding
    if (!_enableCommitAndContinue && syncType != SyncType::COMMIT_AND_WAIT) {
      _syncType = SyncType::COMMIT;
    }

    // check if graph execution profiling is enabled
    if (isGraphProfilingEnabled) {
      // with profiler enabled, we commit after adding the completedHandler in MPSProfiler
      profiler.endProfileKernel(mpsGraph, this, _syncType);
    } else {
      synchronize(_syncType);
    }
  });
}

id<MTLBuffer> MPSStream::getErrorBuffer() {
  return _errorBuffer;
}

void MPSStream::setActiveCaptureId(uint64_t id) {
  const bool wasRecording = _activeCaptureId.load(std::memory_order_relaxed) != 0;
  const bool isRecording = id != 0;
  // Publish the transition in the order that keeps the count a safe over-
  // approximation: a reader must never see count == 0 while a stream is still
  // recording, or it would skip the per-stream check and report "not capturing".
  if (isRecording && !wasRecording) {
    recording_stream_count.fetch_add(1, std::memory_order_release);
  }
  _activeCaptureId.store(id, std::memory_order_release);
  if (!isRecording && wasRecording) {
    recording_stream_count.fetch_sub(1, std::memory_order_release);
  }
}

void MPSStream::captureNoteBuffer(id<MTLBuffer> buffer) {
  if (!buffer || !captureMode()) {
    return;
  }
  const void* ptr = (__bridge const void*)buffer;
  std::lock_guard<std::mutex> lock(_notedBuffersMutex);
  // Re-check under the lock: captureEnd() drains this set, and an op may still
  // be wrapping buffers as the capture ends.
  if (!captureMode()) {
    return;
  }
  if (_notedBuffers.insert(ptr).second) {
    [buffer retain];
    getIMPSAllocator()->pinBufferForCapture(ptr);
  }
}

uint64_t MPSStream::captureBegin() {
  __block uint64_t captureId = 0;
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    // Checked inside the queue, so two simultaneous callers cannot both pass.
    TORCH_CHECK(!getMPSProfiler().isOperationProfilingEnabled(),
                "MPS graph capture requires MPSProfiler operation profiling to be disabled.");
    TORCH_CHECK(!captureMode(), "MPS graph capture already in progress on this stream");
    captureId = next_capture_id.fetch_add(1, std::memory_order_relaxed);
    // Start on a fresh encoder: encoder bindings are sticky, so recording into
    // one that pre-capture ops already bound would let a captured kernel inherit
    // state that was never recorded and so is absent on replay.
    endKernelCoalescing();
    setActiveCaptureId(captureId);
  });
  return captureId;
}

MPSStream::Capture MPSStream::captureEnd(uint64_t captureId) {
  Capture capture;
  Capture* ended = &capture;
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    TORCH_CHECK(activeCaptureId() == captureId, "capture ", captureId, " is not the one recording on this stream");
    setActiveCaptureId(0);
    // Close the encoder the capture recorded into, so later ops do not encode
    // against inherited capture state.
    endKernelCoalescing();
    *ended = std::move(_recording);
    _recording = Capture();
    // Drained after clearing the active id, so a late registration cannot land
    // in the set after it has been moved.
    std::lock_guard<std::mutex> lock(_notedBuffersMutex);
    ended->boundBuffers.assign(_notedBuffers.begin(), _notedBuffers.end());
    _notedBuffers.clear();
  });
  return capture;
}

void MPSStream::releaseCapture(Capture& capture) {
  for (auto& step : capture.steps) {
    if (!step.metalKernel) {
      [(__bridge id<MTLBuffer>)step.blitSrc release];
      [(__bridge id<MTLBuffer>)step.blitDst release];
      continue;
    }
    for (auto& b : step.metalKernel->buffers) {
      [(__bridge id<MTLBuffer>)b.buffer release];
    }
    for (auto& r : step.metalKernel->resourceUsages) {
      [(__bridge id<MTLResource>)r.resource release];
    }
    [(__bridge id<MTLComputePipelineState>)step.metalKernel->pso release];
  }
  for (const void* ptr : capture.boundBuffers) {
    getIMPSAllocator()->unpinBufferForCapture(ptr);
    [(__bridge id<MTLBuffer>)ptr release];
  }
  capture = Capture();
}

void MPSStream::pushCapturedMetalKernel(std::unique_ptr<CapturedMetalKernel> kernel) {
  // The recording encoder is only driven from blocks on _serialQueue, which is
  // what makes the unsynchronized _recording access below safe.
  dispatch_assert_queue(_serialQueue);
  TORCH_INTERNAL_ASSERT(captureMode(), "pushCapturedMetalKernel called outside capture mode");
  for (const auto& b : kernel->buffers) {
    captureNoteBuffer((__bridge id<MTLBuffer>)b.buffer);
  }
  // useResource: resources reach the kernel by raw GPU address (argument buffers
  // in the fused optimizers, index tensors in advanced indexing) rather than
  // through setBuffer, so they need the same reservation. Pinning something the
  // allocator does not own is harmless: the pin key simply never matches.
  for (const auto& r : kernel->resourceUsages) {
    captureNoteBuffer((__bridge id<MTLBuffer>)r.resource);
  }
  CapturedStep step;
  step.metalKernel = std::move(kernel);
  _recording.steps.push_back(std::move(step));
}

void MPSStream::replay(const Capture& capture) {
  const Capture* replayed = &capture;
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    if (replayed->steps.empty()) {
      return;
    }
    endKernelCoalescing();
    // A replay inside a capture on this stream must land in that capture. Kernel
    // steps are recorded by the recording encoder as they dispatch; blits encode
    // directly, so they are duplicated below. The outer capture re-executes this
    // one's work, so it reserves the same buffers: this graph may be released first.
    const bool recording = captureMode();
    if (recording) {
      for (const void* ptr : replayed->boundBuffers) {
        captureNoteBuffer((__bridge id<MTLBuffer>)ptr);
      }
    }
    for (const auto& step : replayed->steps) {
      if (!step.metalKernel) {
        endKernelCoalescing();
        id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)step.blitSrc;
        id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)step.blitDst;
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];
        markWorkEncoded();
        // Match the 2GB-chunked copy in MPSStream::copy (see #124335).
        constexpr size_t max_copy_size = 0x80000000; // 2GB
        size_t bytes_copied = 0;
        size_t bytes_remains = step.blitLength;
        while (bytes_remains > 0) {
          NSUInteger bytes_to_copy = std::min(max_copy_size, bytes_remains);
          [blitEncoder copyFromBuffer:srcBuffer
                         sourceOffset:(NSUInteger)step.blitSrcOffset + bytes_copied
                             toBuffer:dstBuffer
                    destinationOffset:(NSUInteger)step.blitDstOffset + bytes_copied
                                 size:bytes_to_copy];
          bytes_copied += bytes_to_copy;
          bytes_remains -= bytes_to_copy;
        }
        [blitEncoder endEncoding];
        if (recording) {
          CapturedStep dup;
          dup.blitSrc = (__bridge void*)[srcBuffer retain];
          dup.blitDst = (__bridge void*)[dstBuffer retain];
          dup.blitLength = step.blitLength;
          dup.blitSrcOffset = step.blitSrcOffset;
          dup.blitDstOffset = step.blitDstOffset;
          _recording.steps.push_back(std::move(dup));
        }
      } else {
        const auto& mk = *step.metalKernel;
        auto enc = commandEncoder();
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)mk.pso];
        for (const auto& b : mk.buffers) {
          [enc setBuffer:(__bridge id<MTLBuffer>)b.buffer offset:b.offset atIndex:b.index];
        }
        for (const auto& b : mk.bytes) {
          [enc setBytes:b.data.data() length:b.data.size() atIndex:b.index];
        }
        for (const auto& tm : mk.threadgroupMemory) {
          [enc setThreadgroupMemoryLength:tm.length atIndex:tm.index];
        }
        for (const auto& r : mk.resourceUsages) {
          [enc useResource:(__bridge id<MTLResource>)r.resource usage:(MTLResourceUsage)r.usage];
        }
        auto gridSize = MTLSizeMake(mk.gridX, mk.gridY, mk.gridZ);
        auto tgSize = MTLSizeMake(mk.tgX, mk.tgY, mk.tgZ);
        if (mk.useThreadgroups) {
          [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
        } else {
          [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        }
      }
      // As eager does after every op, so the GPU starts on the first steps while the rest are still being encoded.
      commitIfNeeded();
    }
    synchronize(SyncType::COMMIT_ADAPTIVE);
  });
}

void MPSStream::checkLastError() {
  std::optional<CommandBufferError> cb_error;
  {
    std::lock_guard<std::mutex> lock(_commandBufferErrorMutex);
    std::swap(cb_error, _commandBufferError);
  }
  if (cb_error) {
    const auto msg = "MPS command buffer execution failed: " + cb_error->message +
        ". This error may have been asynchronously reported, so the stack trace below might be incorrect.";
    TORCH_CHECK_WITH(OutOfMemoryError, !cb_error->is_oom, msg);
    throw c10::AcceleratorError({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, cb_error->code, msg);
  }
  auto msgs = reinterpret_cast<c10::metal::ErrorMessages*>([_errorBuffer contents]);
  if (!msgs) {
    return;
  }
  const auto& msg = msgs->msg[0];
  unsigned int count = 0;
  std::swap(count, msgs->count);
  if (!count) {
    return;
  }
  throw c10::AcceleratorError({msg.func, msg.file, msg.line}, 1, msg.message);
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

namespace {
thread_local MPSStream* current_stream = nullptr;
} // namespace

MPSStream* getCurrentMPSStream() {
  return current_stream ? current_stream : getDefaultMPSStream();
}

void setCurrentMPSStream(MPSStream* stream) {
  current_stream = stream;
}

MPSStream* getDefaultMPSStream() {
  return MPSStreamImpl::getInstance();
}

//-----------------------------------------------------------------
//  MPS stream pool
//-----------------------------------------------------------------

namespace {
constexpr int kMPSStreamsPerPool = 32;

std::array<MPSStream*, kMPSStreamsPerPool> stream_pool{};
c10::once_flag stream_pool_flag;
std::atomic<uint32_t> stream_pool_counter{0};
std::atomic<bool> stream_pool_initialized{false};

void initStreamPool() {
  // Pool ids start at 1; id 0 is reserved for the default stream.
  for (const auto i : c10::irange(kMPSStreamsPerPool)) {
    stream_pool[i] = new MPSStream(Stream(Stream::UNSAFE, c10::Device(DeviceType::MPS, 0), i + 1));
  }
  stream_pool_initialized.store(true, std::memory_order_release);
}
} // namespace

MPSStream* getStreamFromPool() {
  c10::call_once(stream_pool_flag, initStreamPool);
  return stream_pool[stream_pool_counter++ % kMPSStreamsPerPool];
}

MPSStream* getStreamByID(int64_t stream_id) {
  if (stream_id == 0) {
    return at::mps::getDefaultMPSStream();
  }
  TORCH_CHECK(stream_id >= 1 && stream_id <= kMPSStreamsPerPool, "stream_id=", stream_id, " not found");
  c10::call_once(stream_pool_flag, initStreamPool);
  return stream_pool[stream_id - 1];
}

void synchronizeAllMPSStreams(SyncType syncType) {
  auto sync = [syncType](MPSStream* stream) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      stream->synchronize(syncType);
    });
  };
  sync(getDefaultMPSStream());
  // don't eagerly create the pool just to synchronize it
  if (stream_pool_initialized.load(std::memory_order_acquire)) {
    for (auto* stream : stream_pool) {
      sync(stream);
    }
  }
}

// Helper methods
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)()) {
  __block std::optional<std::exception_ptr> block_exception;
  dispatch_sync(queue, ^() {
    try {
      block();
      if (auto stream = getCurrentMPSStream(); stream->queue() == queue) {
        stream->commitIfNeeded();
      }
    } catch (...) {
      block_exception = std::current_exception();
    }
  });
  if (block_exception) {
    std::rethrow_exception(*block_exception);
  }
}

} // namespace at::mps
