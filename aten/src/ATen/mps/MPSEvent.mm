//  Copyright © 2023 Apple Inc.

#include <ATen/mps/MPSEvent.h>

namespace at::mps {

MPSEvent::MPSEvent(id_t ID, MPSStream* stream, bool enable_timing)
    : m_id(ID), m_enable_timing(enable_timing), m_stream(stream), m_event([stream->device() newSharedEvent]) {}

MPSEvent::~MPSEvent() {
  if (m_event) {
    [m_event release];
    m_event = nil;
  }
  if (m_listener) {
    [m_listener release];
    m_listener = nil;
  }
}

void MPSEvent::recordLocked(bool syncEvent) {
  // active encoders must end before encoding or waiting
  m_stream->endKernelCoalescing();
  ++m_signalCounter;
  if (m_enable_timing) {
    notifyLocked(^(id<MTLSharedEvent>, uint64_t) {
      m_completion_time = getTime();
      notifyCpuSync();
    });
  }
  id<MTLCommandBuffer> commandBuffer = m_stream->commandBuffer();
  [commandBuffer encodeSignalEvent:m_event value:m_signalCounter];
  if (syncEvent) {
    m_stream->synchronize(SyncType::COMMIT);
  }
}

bool MPSEvent::waitLocked(bool syncEvent) {
  // check if event is not recorded yet
  if (m_event.signaledValue >= m_signalCounter) {
    return false;
  }
  // active encoders must end before encoding or waiting
  m_stream->endKernelCoalescing();
  id<MTLCommandBuffer> commandBuffer = m_stream->commandBuffer();
  [commandBuffer encodeWaitForEvent:m_event value:m_signalCounter];
  if (syncEvent) {
    m_stream->synchronize(SyncType::COMMIT);
  }
  return true;
}

bool MPSEvent::notifyLocked(MTLSharedEventNotificationBlock block) {
  // check if event is not recorded yet
  if (m_event.signaledValue >= m_signalCounter) {
    return false;
  }
  if (!m_listener) {
    m_listener = [[MTLSharedEventListener alloc] init];
  }
  [m_event notifyListener:m_listener atValue:m_signalCounter block:block];
  return true;
}

void MPSEvent::record(bool needsLock, bool syncEvent) {
  if (!needsLock) {
    recordLocked(syncEvent);
    return;
  }
  dispatch_sync(m_stream->queue(), ^() {
    @autoreleasepool {
      recordLocked(syncEvent);
    }
  });
}

bool MPSEvent::wait(bool needsLock, bool syncEvent) {
  __block bool waited = false;
  if (!needsLock) {
    return waitLocked(syncEvent);
  }
  dispatch_sync(m_stream->queue(), ^() {
    @autoreleasepool {
      waited = waitLocked(syncEvent);
    }
  });
  return waited;
}

bool MPSEvent::notify(bool needsLock, MTLSharedEventNotificationBlock block) {
  if (!needsLock) {
    return notifyLocked(block);
  }
  __block bool scheduledNotify = false;
  dispatch_sync(m_stream->queue(), ^() {
    @autoreleasepool {
      scheduledNotify = notifyLocked(block);
    }
  });
  return scheduledNotify;
}

void MPSEvent::notifyCpuSync() {
  std::lock_guard<std::mutex> lock(m_cpu_sync_mutex);
  m_cpu_sync_completed = true;
  m_cpu_sync_cv.notify_one();
}

void MPSEvent::waitForCpuSync() {
  std::unique_lock<std::mutex> lock(m_cpu_sync_mutex);
  m_cpu_sync_cv.wait(lock, [&] { return m_cpu_sync_completed; });
  m_cpu_sync_completed = false;
}

bool MPSEvent::synchronize() {
  bool scheduledNotify = notifyLocked(^(id<MTLSharedEvent>, uint64_t) {
    m_completion_time = getTime();
    notifyCpuSync();
  });

  if (scheduledNotify) {
    waitForCpuSync();
    return true;
  }
  return false;
}

bool MPSEvent::query() const {
  // return false if not recorded or signaled yet
  return m_signalCounter && (m_event.signaledValue >= m_signalCounter);
}

void MPSEvent::reset(MPSStream* stream, bool enable_timing) {
  if (stream != m_stream) {
    m_signalCounter = 0;
    m_event.signaledValue = 0;
    m_stream = stream;
  }
  // reset record time
  m_completion_time = 0;
  m_enable_timing = enable_timing;
  m_cpu_sync_completed = false;
};

//-----------------------------------------------------------------
//  MPSEventPool
//-----------------------------------------------------------------

MPSEventPool::MPSEventPool(MPSStream* default_stream) : m_default_stream(default_stream) {
  // default deleter to return the event back to pool after it gets released
  m_default_deleter = [&](MPSEvent* event) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_pool.push(std::unique_ptr<MPSEvent>(event));
  };
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
      return MPSEventPtr(event, m_default_deleter);
    }
  }
  auto new_event = std::make_unique<MPSEvent>(++m_event_counter, stream, enable_timing);
  return MPSEventPtr(new_event.release(), m_default_deleter);
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

void MPSEventPool::recordEvent(id_t event_id, bool syncEvent) {
  MPSEvent* event = getInUseEvent(event_id);
  event->record(/*needsLock*/ true, syncEvent);
}

void MPSEventPool::waitForEvent(id_t event_id, bool syncEvent) {
  MPSEvent* event = getInUseEvent(event_id);
  event->wait(/*needsLock*/ true, syncEvent);
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
  // first make sure notifyListeners are called to capture events' completion times
  dispatch_sync(m_default_stream->queue(), ^() {
    m_default_stream->synchronize(SyncType::COMMIT_AND_WAIT);
  });
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  MPSEvent* start_event = getInUseEvent(start_event_id, false);
  MPSEvent* end_event = getInUseEvent(end_event_id, false);
  // the notify is called on a separate thread, so this waits for that
  end_event->waitForCpuSync();
  const uint64_t start_time = start_event->getCompletionTime();
  const uint64_t end_time = end_event->getCompletionTime();

  TORCH_CHECK(start_time > 0 && end_time > 0, "Events were not created with argument 'enable_timing=True'");
  TORCH_CHECK(
      end_time > start_time, "End event ", end_event_id, " was not recorded after start event ", start_event_id);
  return double(end_time - start_time) * 1e-6;
}

MPSEvent* MPSEventPool::getInUseEvent(id_t event_id, bool locked) {
  if (locked) {
    m_mutex.lock();
  }
  TORCH_CHECK(m_in_use_events.count(event_id) > 0, "Invalid Event ID: ", event_id);
  MPSEvent* event = m_in_use_events[event_id].get();
  if (locked) {
    m_mutex.unlock();
  }
  return event;
}

std::shared_ptr<MPSEventPool> getMPSEventPool() {
  static std::shared_ptr<MPSEventPool> event_pool = std::make_shared<MPSEventPool>(getDefaultMPSStream());
  return event_pool;
}

} // namespace at::mps
