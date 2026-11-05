//  Copyright © 2023 Apple Inc.

#include <ATen/mps/MPSEvent.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/ScopeExit.h>

#include <chrono>
#include <thread>
#include <unordered_map>
#include <vector>

namespace at::mps {

namespace {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = at::native::mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Timestamp_metallib.h>
#endif

id<MTLCounterSet> timestampCounterSet(id<MTLDevice> device) {
  if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
    return nil;
  }
  for (id<MTLCounterSet> set in device.counterSets) {
    if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
      return set;
    }
  }
  return nil;
}

// GPU timestamps for timing events, sampled at the start of a tiny compute
// encoder fenced on both sides by an MTLEvent signal/wait. The fences make the
// sample land after all earlier work on the queue has completed and before any
// later work starts, so a record's timestamp marks a point in the stream (like
// a CUDA event) rather than the end of a whole command buffer. Requires
// MTLCounterSamplingPointAtStageBoundary; callers fall back to the command
// buffer's GPUEndTime otherwise.
//
// Each interval carries a floor of about 2 us (two records back to back, on
// an M4 Pro). A record drains the GPU pipeline at its barrier. Deferred
// records start earlier work only once the stream is committed.
class StageTimestamps {
 public:
  static StageTimestamps* get(id<MTLDevice> device) {
    // leaked on purpose: completed handlers may run during process teardown
    static StageTimestamps* instance = new StageTimestamps(device);
    return instance->m_ready ? instance : nullptr;
  }

  int64_t slotsInUse() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return static_cast<int64_t>(m_sample_buffers.size()) * kSlots - static_cast<int64_t>(m_free_slots.size());
  }

  // Returns a free sample slot, adding a sample buffer when all are in flight,
  // or -1 once Metal refuses another sample buffer (it allows 32 per process).
  int64_t acquireSlot(MPSStream* stream) {
    // Prepare the barrier before reserving a slot or changing event state.
    barrierFor(stream);
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_free_slots.empty()) {
      id<MTLCounterSampleBuffer> buffer = newSampleBuffer(m_device, m_counter_set);
      if (!buffer) {
        return -1;
      }
      auto release_buffer = c10::make_scope_exit([&] { [buffer release]; });
      addSampleBuffer(buffer);
      release_buffer.release();
    }
    const auto slot = m_free_slots.back();
    m_free_slots.pop_back();
    return slot;
  }

  void releaseSlot(int64_t slot) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_free_slots.push_back(slot);
  }

  // Encodes: barrier, an encoder whose start samples `slot`, barrier.
  void encode(MPSStream* stream, id<MTLCommandBuffer> commandBuffer, int64_t slot) {
    Barrier& b = barrierFor(stream);
    id<MTLEvent> barrier = b.event;
    uint64_t& value = b.value;
    [commandBuffer encodeSignalEvent:barrier value:++value];
    [commandBuffer encodeWaitForEvent:barrier value:value];
    MTLComputePassDescriptor* desc = [MTLComputePassDescriptor computePassDescriptor];
    desc.sampleBufferAttachments[0].sampleBuffer = sampleBuffer(slot);
    desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = static_cast<NSUInteger>(slot % kSlots);
    desc.sampleBufferAttachments[0].endOfEncoderSampleIndex = MTLCounterDontSample;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoderWithDescriptor:desc];
    // an encoder without a dispatch is elided and never writes its sample
    [encoder setComputePipelineState:m_noop_pipeline];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commandBuffer encodeSignalEvent:barrier value:++value];
    [commandBuffer encodeWaitForEvent:barrier value:value];
  }

  // Reads `slot` once its command buffer has completed and frees it. Returns
  // the timestamp in seconds, or 0 when the sample was not written.
  double resolveAndRelease(int64_t slot, bool completed) {
    double seconds = 0.0;
    if (completed) {
      @autoreleasepool {
        NSData* data = [sampleBuffer(slot) resolveCounterRange:NSMakeRange(static_cast<NSUInteger>(slot % kSlots), 1)];
        if (data && data.length >= sizeof(MTLCounterResultTimestamp)) {
          const uint64_t ticks = static_cast<const MTLCounterResultTimestamp*>(data.bytes)[0].timestamp;
          if (ticks != 0 && ticks != MTLCounterErrorValue) {
            seconds = static_cast<double>(ticks) / m_ticks_per_second;
          }
        }
      }
    }
    releaseSlot(slot);
    return seconds;
  }

 private:
  // 32 KB of timestamps, the size of one sample buffer
  static constexpr int64_t kSlots = 4096;

  explicit StageTimestamps(id<MTLDevice> device) {
    id<MTLCounterSet> timestamps = timestampCounterSet(device);
    if (!timestamps) {
      return;
    }
    m_noop_pipeline = lib.getPipelineStateForFunc("timestamp_noop");
    m_ticks_per_second = calibrateTicksPerSecond(device);
    NSError* error = nil;
    id<MTLCounterSampleBuffer> buffer = newSampleBuffer(device, timestamps, &error);
    TORCH_CHECK(buffer,
                "MPS event timing could not allocate a GPU timestamp sample buffer: ",
                error ? error.localizedDescription.UTF8String : "allocation failed");
    auto release_buffer = c10::make_scope_exit([&] { [buffer release]; });
    addSampleBuffer(buffer);
    m_device = device;
    // device.counterSets is autoreleased and rebuilt per call; keep this one
    m_counter_set = [timestamps retain];
    m_ready = true;
    release_buffer.release();
  }

  static id<MTLCounterSampleBuffer> newSampleBuffer(id<MTLDevice> device,
                                                    id<MTLCounterSet> counterSet,
                                                    NSError** error = nullptr) {
    MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor new] autorelease];
    desc.counterSet = counterSet;
    desc.storageMode = MTLStorageModeShared;
    desc.sampleCount = kSlots;
    return [device newCounterSampleBufferWithDescriptor:desc error:error];
  }

  // m_mutex held, or during construction
  void addSampleBuffer(id<MTLCounterSampleBuffer> buffer) {
    const auto base = static_cast<int64_t>(m_sample_buffers.size() * kSlots);
    m_sample_buffers.reserve(m_sample_buffers.size() + 1);
    m_free_slots.reserve(base + kSlots);
    m_sample_buffers.push_back(buffer);
    for (int64_t i = kSlots; i > 0; --i) {
      m_free_slots.push_back(base + i - 1);
    }
  }

  id<MTLCounterSampleBuffer> sampleBuffer(int64_t slot) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_sample_buffers[slot / kSlots];
  }

  // GPU timestamp ticks per second: two GPU timestamps 20 ms apart, each
  // placed at the midpoint of the host steady-clock reads around it (the unit
  // of the CPU half of sampleTimestamps is not specified, so it is not used).
  static double calibrateTicksPerSecond(id<MTLDevice> device) {
    using clock = std::chrono::steady_clock;
    auto sample = [&](MTLTimestamp& gpu) {
      MTLTimestamp cpu = 0;
      const auto before = clock::now();
      [device sampleTimestamps:&cpu gpuTimestamp:&gpu];
      const auto after = clock::now();
      return before + (after - before) / 2;
    };
    MTLTimestamp gpu0 = 0, gpu1 = 0;
    const auto t0 = sample(gpu0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto t1 = sample(gpu1);
    TORCH_CHECK(gpu0 > 0 && gpu1 > gpu0, "MPS event timing: GPU timestamp clock did not advance");
    return static_cast<double>(gpu1 - gpu0) / std::chrono::duration<double>(t1 - t0).count();
  }

  // One per stream: a barrier's signal and wait values must rise in the order
  // they are encoded, which only a single stream's serial queue guarantees.
  struct Barrier {
    id<MTLEvent> event;
    // only touched on the owning stream's serial queue
    uint64_t value = 0;
  };

  // Must be called on the stream's serial queue. References into the map stay
  // valid across insertions by other streams.
  Barrier& barrierFor(MPSStream* stream) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_barriers.find(stream);
    if (it == m_barriers.end()) {
      id<MTLEvent> event = [m_device newEvent];
      TORCH_CHECK(event, "MPS event timing could not allocate its stream barrier");
      auto release_event = c10::make_scope_exit([&] { [event release]; });
      it = m_barriers.emplace(stream, Barrier{event}).first;
      release_event.release();
    }
    return it->second;
  }

  // set once the device supports stage-boundary sampling and setup succeeded
  bool m_ready = false;
  id<MTLDevice> m_device = nil;
  id<MTLCounterSet> m_counter_set = nil;
  std::vector<id<MTLCounterSampleBuffer>> m_sample_buffers;
  id<MTLComputePipelineState> m_noop_pipeline = nil;
  double m_ticks_per_second = 1e9;
  std::mutex m_mutex;
  std::vector<int64_t> m_free_slots;
  std::unordered_map<MPSStream*, Barrier> m_barriers;
};

} // namespace

MPSEvent::MPSEvent(id_t ID, MPSStream* stream, bool enable_timing)
    : m_id(ID), m_enable_timing(enable_timing), m_stream(stream), m_event([stream->device() newSharedEvent]) {}

MPSEvent::~MPSEvent() {
  if (m_event) {
    [m_event release];
    m_event = nil;
  }
}

void MPSEvent::recordLocked(bool syncEvent, MPSStream* stream) {
  // Take the timestamp slot before touching any event state, so a failure
  // leaves the event as it was.
  MPSStream* record_stream = stream ? stream : m_stream;
  StageTimestamps* stage = m_enable_timing ? StageTimestamps::get(record_stream->device()) : nullptr;
  const int64_t slot = stage ? stage->acquireSlot(record_stream) : -1;
  TORCH_CHECK(!stage || slot >= 0,
              "MPS event timing: all ",
              stage ? stage->slotsInUse() : 0,
              " GPU timestamp slots are in flight; synchronize before recording more timing events");
  auto release_slot = c10::make_scope_exit([&] {
    if (slot >= 0) {
      stage->releaseSlot(slot);
    }
  });
  m_stream = record_stream;
  // If the stream has been changed since the last `record` and there are still
  // pending signals on the previous stream, create a new Metal event so that
  // pending waits stay bound to the old Metal event.
  if (m_signal_stream && m_signal_stream != m_stream && m_event.signaledValue < m_signalCounter.load()) {
    [m_event release];
    m_event = [m_stream->device() newSharedEvent];
  }
  m_signal_stream = m_stream;
  // Keep stage-timed records buffered until a progress operation submits them.
  // This lets the timestamp bracket GPU work without host encoding gaps.
  m_pending_commit.store(slot >= 0 && !syncEvent);
  // active encoders must end before encoding or waiting
  m_stream->endKernelCoalescing();
  const uint64_t signalCounter = ++m_signalCounter;
  id<MTLCommandBuffer> commandBuffer = m_stream->commandBuffer();
  if (m_enable_timing) {
    uint64_t timingGeneration;
    {
      std::lock_guard<std::mutex> lock(m_cpu_sync_mutex);
      timingGeneration = ++m_timing_generation;
    }
    if (slot >= 0) {
      stage->encode(m_stream, commandBuffer, slot);
      [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        notifyCpuSync(timingGeneration, stage->resolveAndRelease(slot, cb.status == MTLCommandBufferStatusCompleted));
      }];
      release_slot.release();
    } else {
      // Without stage-boundary sampling, the event commits right after the
      // signal and the command buffer's GPU end time is its timestamp.
      syncEvent = true;
      [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        notifyCpuSync(timingGeneration, cb.GPUEndTime);
      }];
    }
  }
  [commandBuffer encodeSignalEvent:m_event value:signalCounter];
  m_recorded.store(true);
  if (syncEvent) {
    m_stream->synchronize(SyncType::COMMIT);
  }
}

void MPSEvent::commitPendingRecord(bool waitForCommit) const {
  if (!m_pending_commit.exchange(false)) {
    return;
  }
  MPSStream* stream = m_signal_stream;
  dispatch_block_t commit = ^() {
    @autoreleasepool {
      stream->synchronize(SyncType::COMMIT);
    }
  };
  if (waitForCommit) {
    dispatch_sync(stream->queue(), commit);
  } else {
    dispatch_async(stream->queue(), commit);
  }
}

bool MPSEvent::waitLocked(bool syncEvent, MPSStream* stream) {
  const uint64_t signalCounter = m_signalCounter.load();
  // skip the wait if the event is unrecorded or already signaled
  if (!m_recorded.load() || m_event.signaledValue >= signalCounter) {
    return false;
  }
  MPSStream* wait_stream = stream ? stream : m_stream;
  // A wait on another stream would block on a signal that never gets
  // committed. The GPU wait only needs the commit to happen eventually, and
  // this runs on wait_stream's queue, so do not block on the signal stream's.
  if (wait_stream != m_signal_stream) {
    commitPendingRecord(/*waitForCommit=*/false);
  }
  // active encoders must end before encoding or waiting
  wait_stream->endKernelCoalescing();
  id<MTLCommandBuffer> commandBuffer = wait_stream->commandBuffer();
  [commandBuffer encodeWaitForEvent:m_event value:signalCounter];
  if (syncEvent) {
    wait_stream->synchronize(SyncType::COMMIT);
  }
  return true;
}

void MPSEvent::record(bool needsLock, bool syncEvent, MPSStream* stream) {
  if (!needsLock) {
    recordLocked(syncEvent, stream);
    return;
  }
  // Rethrows a failed record (e.g. timestamp slots exhausted) to the caller.
  // Not dispatch_sync_with_rethrow: that may commit the stream afterwards,
  // which after a start record would let the GPU idle inside the timed window.
  __block std::exception_ptr error;
  MPSStream* dispatch_stream = stream ? stream : m_stream;
  dispatch_sync(dispatch_stream->queue(), ^() {
    @autoreleasepool {
      try {
        recordLocked(syncEvent, stream);
      } catch (...) {
        error = std::current_exception();
      }
    }
  });
  if (error) {
    std::rethrow_exception(error);
  }
}

bool MPSEvent::wait(bool needsLock, bool syncEvent, MPSStream* stream) {
  __block bool waited = false;
  if (!needsLock) {
    return waitLocked(syncEvent, stream);
  }
  MPSStream* dispatch_stream = stream ? stream : m_stream;
  dispatch_sync(dispatch_stream->queue(), ^() {
    @autoreleasepool {
      waited = waitLocked(syncEvent, stream);
    }
  });
  return waited;
}

void MPSEvent::notifyCpuSync(uint64_t timingGeneration, double completionTime) {
  std::lock_guard<std::mutex> lock(m_cpu_sync_mutex);
  if (timingGeneration > m_completed_timing_generation) {
    m_completion_time = completionTime;
    m_completed_timing_generation = timingGeneration;
  }
  m_cpu_sync_cv.notify_one();
}

double MPSEvent::waitForTiming() {
  std::unique_lock<std::mutex> lock(m_cpu_sync_mutex);
  m_cpu_sync_cv.wait(lock, [&] { return m_completed_timing_generation >= m_timing_generation; });
  return m_completion_time;
}

bool MPSEvent::synchronize() {
  const uint64_t signalCounter = m_signalCounter.load();
  if (!m_recorded.load() || m_event.signaledValue >= signalCounter) {
    return false;
  }
  commitPendingRecord();
  // Metal has no documented infinite-timeout value. Retry finite waits to
  // preserve synchronize()'s unbounded semantics without risking timeout
  // conversion overflow; a never-signaled event remains blocked as before.
  constexpr uint64_t wait_interval_ms = 60'000;
  while (![m_event waitUntilSignaledValue:signalCounter timeoutMS:wait_interval_ms]) {
  }
  return true;
}

bool MPSEvent::query() const {
  const uint64_t signalCounter = m_signalCounter.load();
  if (!m_recorded.load()) {
    return false;
  }
  if (m_event.signaledValue >= signalCounter) {
    return true;
  }
  // an uncommitted signal would keep the event pending forever; commit it
  // without blocking, as query() must not
  commitPendingRecord(/*waitForCommit=*/false);
  return false;
}

void MPSEvent::reset(MPSStream* stream, bool enable_timing) {
  // Deliberately do not reset `m_signalCounter` or `m_event.signaledValue`.
  // Their absolute values don't matter, only the difference between them does.
  // Resetting them could cause a subsequent `waitLocked` call to incorrectly
  // determine that an event was satisfied without waiting for it.
  m_stream = stream;
  {
    std::lock_guard<std::mutex> lock(m_cpu_sync_mutex);
    m_completion_time = 0.0;
  }
  m_enable_timing = enable_timing;
  m_pending_commit.store(false);
  m_recorded.store(false);
};

//-----------------------------------------------------------------
//  MPSEventPtrTarget
//-----------------------------------------------------------------

MPSEventPtrTarget::~MPSEventPtrTarget() {
  m_pool->returnEventToPool(m_event);
}

//-----------------------------------------------------------------
//  MPSEventPool
//-----------------------------------------------------------------

MPSEventPool::MPSEventPool(MPSStream* default_stream) : m_default_stream(default_stream) {}

void MPSEventPool::returnEventToPool(MPSEvent* event) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  m_pool.push(std::unique_ptr<MPSEvent>(event));
}

MPSEventPool::~MPSEventPool() {
  emptyCache();
}

MPSEventPtr MPSEventPool::acquireEvent(bool enable_timing, MPSStream* stream) {
  if (!stream) {
    stream = m_default_stream;
  }
  {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_pool.empty()) {
      auto event = m_pool.top().release();
      m_pool.pop();
      event->reset(stream, enable_timing);
      return c10::make_intrusive<MPSEventPtrTarget>(event, this);
    }
  }
  auto new_event = std::make_unique<MPSEvent>(++m_event_counter, stream, enable_timing);
  return c10::make_intrusive<MPSEventPtrTarget>(new_event.release(), this);
}

void MPSEventPool::emptyCache() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  while (!m_pool.empty()) {
    m_pool.pop();
  }
}

id_t MPSEventPool::acquireEvent(bool enable_timing) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  MPSEventPtr event = acquireEvent(enable_timing, nullptr);
  TORCH_INTERNAL_ASSERT(event);
  id_t event_id = event->getID();
  m_in_use_events.emplace(event_id, std::move(event));
  return event_id;
}

void MPSEventPool::releaseEvent(id_t event_id) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  TORCH_CHECK(m_in_use_events.count(event_id) > 0, "Invalid Event ID: ", event_id);
  // returns the event back to the MPSEventPool
  m_in_use_events.erase(event_id);
}

void MPSEventPool::recordEvent(id_t event_id, MPSStream* stream) {
  MPSEvent* event = getInUseEvent(event_id);
  event->record(/*needsLock=*/true, /*syncEvent=*/!event->isTimingEnabled(), stream);
}

void MPSEventPool::waitForEvent(id_t event_id, bool syncEvent, MPSStream* stream) {
  MPSEvent* event = getInUseEvent(event_id);
  event->wait(/*needsLock*/ true, syncEvent, stream);
}

void MPSEventPool::synchronizeEvent(id_t event_id) {
  MPSEvent* event = getInUseEvent(event_id);
  event->synchronize();
}

bool MPSEventPool::queryEvent(id_t event_id) {
  MPSEvent* event = getInUseEvent(event_id);
  return event->query();
}

double MPSEventPool::elapsedTime(id_t start_event_id, id_t end_event_id) {
  // Events recorded on other streams may still be waiting for a commit.
  getInUseEvent(start_event_id)->commitPendingRecord();
  getInUseEvent(end_event_id)->commitPendingRecord();
  // First make sure the command buffers containing both events have completed.
  dispatch_sync(m_default_stream->queue(), ^() {
    m_default_stream->synchronize(SyncType::COMMIT_AND_WAIT);
  });
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  MPSEvent* start_event = getInUseEvent(start_event_id, false);
  MPSEvent* end_event = getInUseEvent(end_event_id, false);
  TORCH_CHECK(start_event->isTimingEnabled() && end_event->isTimingEnabled(),
              "Events were not created with argument 'enable_timing=True'");
  TORCH_CHECK(start_event->isRecorded() && end_event->isRecorded(),
              "Both events must be recorded before calculating elapsed time");
  const double start_time = start_event->waitForTiming();
  const double end_time = end_event->waitForTiming();

  TORCH_CHECK(start_time > 0.0 && end_time > 0.0, "MPS event timing failed because a GPU timestamp was unavailable");
  TORCH_CHECK(
      end_time >= start_time, "End event ", end_event_id, " was not recorded after start event ", start_event_id);
  return (end_time - start_time) * 1e3;
}

MPSEvent* MPSEventPool::getInUseEvent(id_t event_id, bool locked) {
  if (locked) {
    m_mutex.lock();
  }
  TORCH_CHECK(m_in_use_events.count(event_id) > 0, "Invalid Event ID: ", event_id);
  MPSEvent* event = m_in_use_events[event_id]->get();
  if (locked) {
    m_mutex.unlock();
  }
  return event;
}

std::shared_ptr<MPSEventPool> getMPSEventPool() {
  static std::shared_ptr<MPSEventPool> event_pool = std::make_shared<MPSEventPool>(getDefaultMPSStream());
  return event_pool;
}

bool hasStageBoundaryTiming() {
  @autoreleasepool {
    return timestampCounterSet(MPSDevice::getInstance()->device()) != nil;
  }
}

} // namespace at::mps
