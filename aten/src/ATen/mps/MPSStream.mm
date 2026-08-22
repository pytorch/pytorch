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

  _errorBuffer = [MPSDevice::getInstance()->device() newBufferWithLength:sizeof(c10::metal::ErrorMessages)
                                                                 options:MTLResourceStorageModeShared];
  std::memset([_errorBuffer contents], 0, 1024);
}

MPSStream::~MPSStream() {
  for (auto& [id, steps] : _captures) {
    for (auto& step : steps) {
      MPSStream::releaseCapturedStep(step);
    }
  }
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];
  [_compilationDescriptor release];
  _executionDescriptor = nil;
  [_errorBuffer release];
  _errorBuffer = nil;
  _compilationDescriptor = nil;

  TORCH_INTERNAL_ASSERT(_commandBuffer == nil);
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
  if (_activeCaptureId.load(std::memory_order_acquire) != 0) {
    if (!_recordingEncoder) {
      _recordingEncoder = [[MPSRecordingEncoder alloc] initWithEncoder:_commandEncoder stream:this];
    }
    return (id<MTLComputeCommandEncoder>)_recordingEncoder;
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
  if (_enableCommitAndContinue) {
    [commandBuffer() commitAndContinue];
  } else {
    flush();
  }
}

void MPSStream::commitAndWait() {
  if (_prevCommandBuffer) {
    // the previous command buffer (if exists) has already been committed,
    // so we just wait until it's completed and then dispose it.
    [_prevCommandBuffer waitUntilCompleted];
    [_prevCommandBuffer release];
    _prevCommandBuffer = nil;
    checkLastError();
  }

  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    [_commandBuffer release];
    _commandBuffer = nil;
    checkLastError();
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
    // Recording encoder wraps the now-stale inner encoder; release it.
    [_recordingEncoder release];
    _recordingEncoder = nil;
  }
}

void MPSStream::flush() {
  if (_commandBuffer) {
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

void MPSStream::addCompletedHandler(MTLCommandBufferHandler block) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      [commandBuffer() addCompletedHandler:block];
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
  dispatch_sync_with_rethrow(_serialQueue, ^() {
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

      // Record the blit so replay() re-issues it. Ops like cat (contiguous) and
      // copy_ encode through here rather than executeMPSGraph or the recording
      // compute encoder, so without this they would be silently dropped on
      // replay. Buffers are retained here and released in releaseCapturedStep().
      if (uint64_t id = _activeCaptureId.load(std::memory_order_acquire); id != 0) {
        CapturedStep step;
        step.kind = CapturedStep::Kind::BlitCopy;
        step.blitSrc = (__bridge void*)srcBuffer;
        step.blitDst = (__bridge void*)dstBuffer;
        [srcBuffer retain];
        [dstBuffer retain];
        step.blitLength = length;
        step.blitSrcOffset = srcOffset;
        step.blitDstOffset = dstOffset;
        _captures[id].push_back(std::move(step));
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
  auto& profiler = getMPSProfiler();
  const bool isGraphProfilingEnabled = profiler.isOperationProfilingEnabled();

  dispatch_sync_with_rethrow(_serialQueue, ^() {
    endKernelCoalescing();
    if (isGraphProfilingEnabled) {
      // this function call is only relevant for interval-based Signposts
      // which exclude schedule time (only includes GPU run time)
      profiler.beginProfileGPUInterval(mpsGraph, this);
    }

    if (uint64_t id = _activeCaptureId.load(std::memory_order_acquire); id != 0) {
      // Capture: compile the graph into an MPSGraphExecutable, encode it, and
      // record the step so replay() can re-encode it in a single dispatch. The
      // executable is owned by the CapturedStep and released on
      // captureFree()/captureReset()/~MPSStream() -- it is scoped to the capture,
      // not cached persistently. (Capture rejects profiling in captureBegin(), so
      // the profiler branch never overlaps this path.)
      NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feedShapes =
          [[NSMutableDictionary alloc] initWithCapacity:[feeds count]];
      for (MPSGraphTensor* t in feeds) {
        MPSGraphTensorData* tdata = (MPSGraphTensorData*)feeds[t];
        feedShapes[t] = [[[MPSGraphShapedType alloc] initWithShape:tdata.shape dataType:tdata.dataType] autorelease];
      }
      MPSGraphExecutable* exe = [[mpsGraph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device()]
                                                       feeds:feedShapes
                                               targetTensors:[results allKeys]
                                            targetOperations:nil
                                       compilationDescriptor:_compilationDescriptor] retain];
      [feedShapes release];

      // Build ordered input/output arrays using the stable ordering from the executable.
      NSArray<MPSGraphTensor*>* feedTensors = exe.feedTensors;
      NSArray<MPSGraphTensor*>* targetTensors = exe.targetTensors;
      NSMutableArray<MPSGraphTensorData*>* inputsArray = [[NSMutableArray alloc] initWithCapacity:feedTensors.count];
      for (MPSGraphTensor* t in feedTensors) {
        [inputsArray addObject:(MPSGraphTensorData*)feeds[t]];
      }
      NSMutableArray<MPSGraphTensorData*>* resultsArray = [[NSMutableArray alloc] initWithCapacity:targetTensors.count];
      for (MPSGraphTensor* t in targetTensors) {
        [resultsArray addObject:(MPSGraphTensorData*)results[t]];
      }

      [exe encodeToCommandBuffer:commandBuffer()
                     inputsArray:inputsArray
                    resultsArray:resultsArray
             executionDescriptor:nil];

      CapturedStep step;
      step.kind = CapturedStep::Kind::MPSGraph;
      step.exe = (__bridge void*)exe; // owned by the step
      step.inputsArray = [inputsArray retain];
      step.resultsArray = [resultsArray retain];
      _captures[id].push_back(std::move(step));
      [inputsArray release];
      [resultsArray release];
    } else {
      // Normal path: encode the graph directly. No persistent executable cache;
      // the compiled-executable path is used only during capture (above).
      // note: CommitAndContinue feature is enabled/disabled via "_executionDescriptor"
      [mpsGraph encodeToCommandBuffer:commandBuffer()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:results
                  executionDescriptor:_executionDescriptor];
    }

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

uint64_t MPSStream::captureBegin() {
  __block uint64_t captureId = 0;
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    // Both checks run inside the queue: testing them before entering it would let
    // two simultaneous callers pass the exclusivity check, and would leave a
    // window for profiling to be enabled after the check but before recording.
    TORCH_CHECK(!getMPSProfiler().isOperationProfilingEnabled(),
                "MPS graph capture requires MPSProfiler operation profiling to be disabled. "
                "Disable operation profiling before capturing into a torch.mps.MetalGraph.");
    TORCH_CHECK(_activeCaptureId.load(std::memory_order_relaxed) == 0, "MPS graph capture already in progress");
    captureId = _nextCaptureId++;
    _captures.try_emplace(captureId);
    // Start on a fresh encoder: encoder bindings are sticky, so recording into
    // one that pre-capture ops already bound would let a captured kernel inherit
    // state that was never recorded and so is absent on replay.
    endKernelCoalescing();
    _activeCaptureId.store(captureId, std::memory_order_release);
  });
  return captureId;
}

size_t MPSStream::capturedStepCount(uint64_t captureId) const {
  __block size_t count = 0;
  dispatch_sync(_serialQueue, ^() {
    auto it = _captures.find(captureId);
    count = it != _captures.end() ? it->second.size() : 0;
  });
  return count;
}

void MPSStream::captureEnd(uint64_t captureId) {
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    // Checked inside the queue against the id that is actually recording, so a
    // graph cannot stop a different graph's capture, and the check cannot race
    // with a concurrent begin/end.
    const uint64_t active = _activeCaptureId.load(std::memory_order_relaxed);
    TORCH_CHECK(active != 0, "captureEnd() called without a matching captureBegin()");
    TORCH_CHECK(active == captureId,
                "captureEnd() called for capture ",
                captureId,
                " but capture ",
                active,
                " is the one currently recording");
    _activeCaptureId.store(0, std::memory_order_release);
    // Symmetric with captureBegin(): close the encoder the capture recorded into
    // so post-capture ops do not encode against inherited capture state.
    endKernelCoalescing();
  });
}

bool MPSStream::captureFree(uint64_t captureId) {
  __block bool freed = false;
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    auto it = _captures.find(captureId);
    if (it == _captures.end()) {
      return;
    }
    TORCH_CHECK(captureId != _activeCaptureId.load(std::memory_order_acquire),
                "Cannot free a capture that is still being recorded. Call captureEnd() "
                "(i.e. exit the torch.mps.metal_graph() block) first.");
    for (auto& step : it->second) {
      releaseCapturedStep(step);
    }
    _captures.erase(it);
    freed = true;
  });
  return freed;
}

void MPSStream::captureReset() {
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    _activeCaptureId.store(0, std::memory_order_release);
    for (auto& [captureId, steps] : _captures) {
      for (auto& step : steps) {
        releaseCapturedStep(step);
      }
    }
    _captures.clear();
    [_recordingEncoder release];
    _recordingEncoder = nil;
  });
}

void MPSStream::releaseCapturedStep(CapturedStep& step) {
  if (step.kind == CapturedStep::Kind::MPSGraph) {
    [(__bridge MPSGraphExecutable*)step.exe release];
    [(__bridge NSArray*)step.inputsArray release];
    [(__bridge NSArray*)step.resultsArray release];
  } else if (step.kind == CapturedStep::Kind::BlitCopy) {
    [(__bridge id<MTLBuffer>)step.blitSrc release];
    [(__bridge id<MTLBuffer>)step.blitDst release];
  } else if (step.metalKernel) {
    for (auto& b : step.metalKernel->buffers) {
      [(__bridge id<MTLBuffer>)b.buffer release];
    }
    for (auto& r : step.metalKernel->resourceUsages) {
      [(__bridge id<MTLResource>)r.resource release];
    }
    [(__bridge id<MTLComputePipelineState>)step.metalKernel->pso release];
  }
}

void MPSStream::pushCapturedMetalKernel(std::unique_ptr<CapturedMetalKernel> kernel) {
  uint64_t captureId = _activeCaptureId.load(std::memory_order_acquire);
  TORCH_INTERNAL_ASSERT(captureId != 0, "pushCapturedMetalKernel called outside capture mode");
  CapturedStep step;
  step.kind = CapturedStep::Kind::MetalKernel;
  step.metalKernel = std::move(kernel);
  _captures[captureId].push_back(std::move(step));
}

void MPSStream::replay(uint64_t captureId) {
  dispatch_sync_with_rethrow(_serialQueue, ^() {
    auto it = _captures.find(captureId);
    TORCH_CHECK(
        it != _captures.end(), "No such capture handle: ", captureId, ". It may have been freed, or never captured.");
    TORCH_CHECK(captureId != _activeCaptureId.load(std::memory_order_acquire),
                "Cannot replay a capture that is still being recorded. Call captureEnd() "
                "(i.e. exit the torch.mps.metal_graph() block) first.");
    auto& steps = it->second;
    if (steps.empty()) {
      TORCH_WARN(
          "MetalGraph.replay() called with no captured steps. "
          "Did the capture block contain any MPS ops?");
      return;
    }
    endKernelCoalescing();
    // If a DIFFERENT capture is actively recording while this replay runs
    // (e.g. one MetalGraph's replay() is called inside another MetalGraph's
    // capture block), every step re-executed here must also land in that
    // active capture, or it ends up silently missing whatever this replay
    // reissues. MetalKernel-kind steps already get this for free: they
    // dispatch through commandEncoder(), which returns the active capture's
    // MPSRecordingEncoder and records them automatically. MPSGraph and
    // BlitCopy steps encode directly and bypass that proxy entirely, so they
    // need to be duplicated into the active capture explicitly below.
    const uint64_t recordInto = _activeCaptureId.load(std::memory_order_relaxed);
    for (auto& step : steps) {
      if (step.kind == CapturedStep::Kind::MPSGraph) {
        endKernelCoalescing(); // End compute encoder before MPSGraph encoding
        MPSGraphExecutable* exe = (__bridge MPSGraphExecutable*)step.exe;
        NSArray<MPSGraphTensorData*>* ins = (__bridge NSArray*)step.inputsArray;
        NSArray<MPSGraphTensorData*>* outs = (__bridge NSArray*)step.resultsArray;
        [exe encodeToCommandBuffer:commandBuffer() inputsArray:ins resultsArray:outs executionDescriptor:nil];
        if (recordInto != 0 && recordInto != captureId) {
          CapturedStep dup;
          dup.kind = CapturedStep::Kind::MPSGraph;
          dup.exe = (__bridge void*)[exe retain];
          dup.inputsArray = [ins retain];
          dup.resultsArray = [outs retain];
          _captures[recordInto].push_back(std::move(dup));
        }
      } else if (step.kind == CapturedStep::Kind::BlitCopy) {
        endKernelCoalescing(); // End compute encoder before blit encoding
        id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)step.blitSrc;
        id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)step.blitDst;
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];
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
        if (recordInto != 0 && recordInto != captureId) {
          CapturedStep dup;
          dup.kind = CapturedStep::Kind::BlitCopy;
          dup.blitSrc = (__bridge void*)[srcBuffer retain];
          dup.blitDst = (__bridge void*)[dstBuffer retain];
          dup.blitLength = step.blitLength;
          dup.blitSrcOffset = step.blitSrcOffset;
          dup.blitDstOffset = step.blitDstOffset;
          _captures[recordInto].push_back(std::move(dup));
        }
      } else {
        auto& mk = *step.metalKernel;
        auto enc = commandEncoder();
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)mk.pso;
        [enc setComputePipelineState:pso];
        for (auto& b : mk.buffers) {
          auto mtlBuf = (__bridge id<MTLBuffer>)b.buffer;
          TORCH_CHECK(mtlBuf.length == b.bufferLength,
                      "Graph replay: buffer at index ",
                      b.index,
                      " changed size from ",
                      b.bufferLength,
                      " to ",
                      mtlBuf.length,
                      ". Tensor storage was reallocated between capture and replay. "
                      "Use .copy_() to update tensor data in-place.");
          [enc setBuffer:mtlBuf offset:b.offset atIndex:b.index];
        }
        for (auto& b : mk.bytes) {
          [enc setBytes:b.data.data() length:b.data.size() atIndex:b.index];
        }
        for (auto& tm : mk.threadgroupMemory) {
          [enc setThreadgroupMemoryLength:tm.length atIndex:tm.index];
        }
        for (auto& r : mk.resourceUsages) {
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
    }
    synchronize(SyncType::COMMIT_ADAPTIVE);
  });
}

void MPSStream::checkLastError() {
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

//-----------------------------------------------------------------
//  MPSStream::MetalGraph
//-----------------------------------------------------------------

// Binds to whichever stream is current when capture starts, matching the stream
// ops actually enqueue on (getCurrentMPSStream), and keeps using it for replay
// and release so the capture id is always looked up in the table that owns it.
void MPSStream::MetalGraph::captureBegin() {
  TORCH_CHECK(_id == 0,
              "This MetalGraph has already captured. Call reset() before capturing into it again, "
              "or use a new MetalGraph.");
  _stream = getCurrentMPSStream();
  _id = _stream->captureBegin();
}

void MPSStream::MetalGraph::captureEnd() {
  TORCH_CHECK(_id != 0, "captureEnd() called on a MetalGraph that never began a capture");
  _stream->captureEnd(_id);
}

void MPSStream::MetalGraph::replay() {
  TORCH_CHECK(_id != 0, "replay() called on a MetalGraph with nothing captured. Capture into it first.");
  _stream->replay(_id);
}

void MPSStream::MetalGraph::reset() {
  if (_id == 0) {
    return;
  }
  MPSStream* stream = _stream;
  const uint64_t id = _id;
  // Clear our own state first, so an exception below cannot leave this object
  // pointing at a capture it no longer owns.
  _id = 0;
  _stream = nullptr;
  // A graph dropped mid-recording must stop recording first: otherwise
  // captureFree refuses, and the stream would be left with an active capture id
  // that nothing can clear, wedging every later capture on this stream.
  if (stream->captureMode()) {
    try {
      stream->captureEnd(id);
    } catch (const std::exception&) {
      // Not the recording capture; nothing to stop.
    }
  }
  // Runs from the destructor, so it must not throw. A handle that is already
  // gone (released by captureReset()) returns false and needs no action;
  // anything else is a real bug and is surfaced rather than silently dropped.
  try {
    stream->captureFree(id);
  } catch (const std::exception& e) {
    TORCH_WARN("Failed to release MetalGraph capture ", id, ": ", e.what());
  }
}

size_t MPSStream::MetalGraph::stepCount() const {
  return _id == 0 ? 0 : _stream->capturedStepCount(_id);
}

// Helper methods
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)()) {
  __block std::optional<std::exception_ptr> block_exception;
  dispatch_sync(queue, ^() {
    try {
      block();
    } catch (...) {
      block_exception = std::current_exception();
    }
  });
  if (block_exception) {
    std::rethrow_exception(*block_exception);
  }
}

} // namespace at::mps
