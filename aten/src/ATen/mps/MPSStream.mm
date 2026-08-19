//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>
#include <c10/metal/error.h>
#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>

#include <array>
#include <atomic>

#include <cstdio>
#include <cstdlib>

@interface MPSGraphExecutionDescriptor ()
@property(readwrite, atomic) BOOL enableCommitAndContinue;
@end

namespace at::mps {
//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

// DIAGNOSTIC (env MPS_FAULT_NAME_DIAG) fault-naming hooks; also forward-declared in Indexing.mm.
void mpsSetCurrentOp(const char* name);
bool mpsFaultNameDiag();

// env FULL_SYNC: per-op COMMIT_AND_WAIT (hazard test). Local to this TU; declared here because
// commandEncoder()/trackRootCommandBuffer() below are defined ahead of them.
static bool mpsFullSync();
static void mpsSetEncoderLive(bool live);
static void mpsClearCurrentOp();

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
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];
  [_compilationDescriptor release];
  _executionDescriptor = nil;
  [_errorBuffer release];
  _errorBuffer = nil;
  _compilationDescriptor = nil;
  [_trackedRoot release];
  _trackedRoot = nil;
  [_pendingFaultRoot release];
  _pendingFaultRoot = nil;

  assert(_commandBuffer == nil);
}

MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }
  // The wrapper outlives its root: commitAndContinue (ours or MPSGraph's) swaps a fresh root in
  // behind it, so re-check on every access rather than only on creation.
  trackRootCommandBuffer(_commandBuffer);

  return _commandBuffer;
}

id<MTLDevice> MPSStream::device() const {
  return [_commandQueue device];
}

id<MTLComputeCommandEncoder> MPSStream::commandEncoder() {
  if (!_commandEncoder) {
    // DIAGNOSTIC (env FULL_SYNC): per-op hazard test. Draining here fully completes all previously
    // encoded work before this op encodes, serializing every op; if a command-buffer abort vanishes
    // under FULL_SYNC, the fault is an execution-order hazard. This is the only safe point for the
    // drain: COMMIT_AND_WAIT ends the open encoder via endKernelCoalescing, and most call sites hold
    // their encoder in a local across kernel setup, so draining once they have one would leave them
    // encoding into an ended encoder. Here no encoder is outstanding yet, by construction.
    if (mpsFullSync()) {
      commitAndWait();
    }
    _commandEncoder = [commandBuffer() computeCommandEncoder].retain;
    if (mpsFaultNameDiag()) {
      mpsSetEncoderLive(true);
    }
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

// DIAGNOSTIC (env MPS_FAULT_NAME_DIAG): name the op whose command buffer faults. When the env var is
// set, commitAndWait LOGS the faulting op instead of throwing, so a forced per-op commit+wait can
// attribute the FIRST fault without the throw-through-ObjC SIGABRT. Off by default -> deliverable
// behavior unchanged.
//
// Guarded because the two ends run on different threads: ops encode on whatever thread called into
// ATen, while a drain can come from another thread entirely (torch.mps.synchronize, the allocator
// reclaiming buffers, the profiler at shutdown). An unguarded std::string here would be a data race,
// not merely a stale read -- assigning one while another thread copies it can free the buffer mid-read.
//
// Two slots, because an op names itself when it resolves its pipeline state, which is not yet the point
// where its work joins the command buffer. Only once it holds an encoder is its work really in the root
// being built, so a fault can fairly be blamed on it. Ops that name themselves before taking an encoder
// (getLibraryPipelineState runs first at many call sites) park the name in `pending`; it becomes
// `current` when the encoder is handed out. This also keeps the FULL_SYNC drain in commandEncoder()
// honest: that drain reports work encoded BEFORE this op, so it must not see this op's name.
static std::mutex g_op_name_mutex;
static std::string g_pending_op; // named itself, not encoding yet
static std::string g_current_op; // holds an encoder, so its work is in the root being built
static bool g_encoder_live = false;

void mpsSetCurrentOp(const char* name) {
  std::lock_guard<std::mutex> lock(g_op_name_mutex);
  // Kernel coalescing keeps one encoder open across consecutive ops, so an op that names itself while
  // an encoder is already live is encoding into the current root right now: name it directly.
  (g_encoder_live ? g_current_op : g_pending_op) = name ? name : "";
}

bool mpsFaultNameDiag() {
  static const bool on = (std::getenv("MPS_FAULT_NAME_DIAG") != nullptr);
  return on;
}

static std::string mpsCurrentOp() {
  std::lock_guard<std::mutex> lock(g_op_name_mutex);
  return g_current_op;
}

// Called with the encoder live/dead state that just changed. Handing out an encoder promotes whatever
// op was parked in `pending`; ending one leaves the name in place, since work already encoded stays in
// the root and remains a fair thing to blame.
static void mpsSetEncoderLive(bool live) {
  std::lock_guard<std::mutex> lock(g_op_name_mutex);
  g_encoder_live = live;
  if (live) {
    g_current_op = g_pending_op;
  }
}

// Drop the name when the root is replaced, so the new root does not inherit blame for the old root's
// work. A live encoder is left alone: its work is going into the root being built, so the name still
// applies (in practice callers end coalescing before swapping roots, so this is belt-and-braces).
static void mpsClearCurrentOp() {
  std::lock_guard<std::mutex> lock(g_op_name_mutex);
  if (!g_encoder_live) {
    g_current_op.clear();
  }
}

static bool mpsFullSync() {
  static const bool on = (std::getenv("FULL_SYNC") != nullptr);
  return on;
}

// Surface GPU command-buffer faults that commitAndWait would otherwise swallow.
// waitUntilCompleted returns even when the buffer aborted (e.g. a GPU interactivity/timeout
// abort or a faulting encoder), and checkLastError() only inspects the in-kernel shared error
// buffer (c10::metal::ErrorMessages), never MTLCommandBuffer.status/.error. Without this, a
// faulted command buffer passes silently and downstream reads its uninitialized / incomplete
// output as a garbage integer.
static std::string describeCommandBufferFault(id<MTLCommandBuffer> root, const std::string& attribution) {
  NSError* error = [root error];
  std::string msg = "MPS command buffer execution failed (MTLCommandBufferStatusError, code ";
  msg += std::to_string(error ? static_cast<long>([error code]) : -1L);
  msg += "): ";
  msg += error ? [[error localizedDescription] UTF8String] : "unknown error";
  msg += attribution;
  return msg;
}

// Status must be read from -rootCommandBuffer, never the MPSCommandBuffer wrapper: per
// MPSCommandBuffer.h, commitAndContinue commits the root and swaps in a fresh one, so the wrapper's
// status reflects only the newest root. Returns the fault message (empty if none); the message is
// read while the buffer is still alive, and the CALLER must release/nil the buffer before throwing so
// the stream isn't left holding an already-committed buffer (a re-commit aborts the process).
std::string MPSStream::rootFaultMessage(MPSCommandBuffer_t buffer) {
  id<MTLCommandBuffer> root = [buffer rootCommandBuffer];
  if ([root status] != MTLCommandBufferStatusError) {
    return {};
  }
  std::string attribution;
  if (mpsFaultNameDiag()) {
    const std::string op = mpsCurrentOp();
    if (!op.empty()) {
      attribution = " [current op: " + op + "]";
    }
  }
  return describeCommandBufferFault(root, attribution);
}

// Install a completion handler on the root MTLCommandBuffer so a fault is still recorded when the
// root is retired before anyone can query it. commitAndContinue commits the current root and swaps in
// a new one, and MPSGraph does this internally whenever _executionDescriptor.enableCommitAndContinue
// is set, so most roots never pass through our own commit path. Per MPSCommandBuffer.h, handlers stay
// attached to the root they were added to and fire as that root completes. Called from commandBuffer(),
// which every op goes through before encoding, so each new root is tracked before work lands on it.
void MPSStream::trackRootCommandBuffer(MPSCommandBuffer_t buffer) {
  id<MTLCommandBuffer> root = [buffer rootCommandBuffer];
  if (root == _trackedRoot) {
    return;
  }
  // Retained so the identity check above stays sound: a deallocated root could otherwise be
  // replaced by a new one at the same address and silently skip tracking.
  [_trackedRoot release];
  _trackedRoot = [root retain];
  // A fresh root holds none of the previous root's work, so the name must not carry over: otherwise a
  // root containing only MPSGraph work would be blamed on the last custom Metal kernel to run. The
  // caller in commandEncoder() re-promotes the pending name right after this, so an op that is about
  // to encode still gets named.
  if (mpsFaultNameDiag()) {
    mpsClearCurrentOp();
  }

  // Capturing `this` is safe: the stream is a leaked singleton (MPSStreamImpl::getInstance).
  [root addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    if ([cb status] != MTLCommandBufferStatusError) {
      return;
    }
    // The faulting op cannot be named here: this fires asynchronously, long after the op that encoded
    // into this root has moved on, so the current-op name no longer refers to it.
    std::string msg = describeCommandBufferFault(cb, " [retired command buffer; rerun with FULL_SYNC=1 to attribute]");
    std::lock_guard<std::mutex> lock(_faultMutex);
    // Keep the FIRST fault: later ones are usually cascade damage from the same root cause.
    if (_pendingFault.empty()) {
      _pendingFault = std::move(msg);
      _pendingFaultRoot = [cb retain];
    }
  }];
}

// Hand back a fault recorded by a completion handler. A fault belonging to `currentRoot` is dropped
// rather than returned: the caller can still read that root's status directly and build a better
// attributed message, so returning it here would only shadow the more informative one.
std::string MPSStream::takePendingFault(MTLCommandBuffer_t currentRoot) {
  std::lock_guard<std::mutex> lock(_faultMutex);
  std::string msg;
  if (_pendingFaultRoot != currentRoot) {
    msg = std::move(_pendingFault);
  }
  _pendingFault.clear();
  [_pendingFaultRoot release];
  _pendingFaultRoot = nil;
  return msg;
}

// Surface both fault channels of a completed command buffer. The in-kernel error buffer is drained
// FIRST and unconditionally: a throw that skips checkLastError() leaves the reported error queued,
// where it resurfaces on a later unrelated sync attributed to the wrong op. checkLastError() also
// raises the more specific c10::AcceleratorError, so a kernel-reported error outranks the buffer
// status. Under MPS_FAULT_NAME_DIAG the fault is logged rather than thrown (so a forced per-op
// commit+wait can attribute the FIRST fault), and is logged before the drain can throw.
static void handleCbFault(MPSStream* stream, const std::string& cbError) {
  if (!cbError.empty() && mpsFaultNameDiag()) {
    fprintf(stderr, "MPS_FAULT_OP: %s\n", cbError.c_str());
    fflush(stderr);
    stream->checkLastError();
    return;
  }
  stream->checkLastError();
  TORCH_CHECK(cbError.empty(), cbError);
}

// commitAndWait reports the FIRST fault it can find: a fault recorded by a completion handler
// belongs to an earlier root than the buffer we just waited on, so it wins over that buffer's own
// status. Buffers are detached from the stream before reporting, since reporting may throw and a
// stream left holding an already-committed buffer aborts the process on the next commit.
void MPSStream::commitAndWait() {
  if (_prevCommandBuffer) {
    // the previous command buffer (if exists) has already been committed,
    // so we just wait until it's completed and then dispose it.
    [_prevCommandBuffer waitUntilCompleted];
    MPSCommandBuffer_t buffer = _prevCommandBuffer;
    _prevCommandBuffer = nil;
    std::string cbError = rootFaultMessage(buffer);
    std::string pending = takePendingFault([buffer rootCommandBuffer]);
    [buffer release];
    handleCbFault(this, pending.empty() ? cbError : pending);
  }

  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    MPSCommandBuffer_t buffer = _commandBuffer;
    _commandBuffer = nil;
    std::string cbError = rootFaultMessage(buffer);
    std::string pending = takePendingFault([buffer rootCommandBuffer]);
    [buffer release];
    handleCbFault(this, pending.empty() ? cbError : pending);
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
    if (mpsFaultNameDiag()) {
      mpsSetEncoderLive(false);
    }
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
    // note: CommitAndContinue feature is enabled/disabled via "_executionDescriptor"
    [mpsGraph encodeToCommandBuffer:commandBuffer()
                              feeds:feeds
                   targetOperations:nil
                  resultsDictionary:results
                executionDescriptor:_executionDescriptor];

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
