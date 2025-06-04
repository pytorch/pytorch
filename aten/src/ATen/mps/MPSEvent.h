//  Copyright © 2023 Apple Inc.

#pragma once

#include <ATen/mps/MPSStream.h>
#include <ctime>
#include <stack>

namespace at::mps {

// NOTE: don't create instances of this class directly.
// Use MPSEventPool to acquire instances of MPSEvent.
class MPSEvent {
 public:
  explicit MPSEvent(id_t ID, MPSStream* stream, bool enable_timing);
  ~MPSEvent();

  // records an event on the stream
  void record(bool needsLock, bool syncEvent = false);
  // makes all future work submitted to the stream wait for this event.
  bool wait(bool needsLock, bool syncEvent = false);
  // schedules a notifyListener callback for the event.
  bool notify(bool needsLock, MTLSharedEventNotificationBlock block);
  // checks if events are already signaled.
  bool query() const;
  // blocks the CPU thread until all the GPU work that were scheduled
  // prior to recording this event are completed.
  bool synchronize();
  // resets this event with new parameters in case it gets reused from the event
  // pool
  void reset(MPSStream* stream, bool enable_timing);
  // returns the unique ID of the event instance
  id_t getID() const {
    return m_id;
  }
  // returns the completion timestamp of the event
  uint64_t getCompletionTime() const {
    return m_completion_time;
  }
  // if already recorded, waits for cpu_sync_cv to be signaled
  void waitForCpuSync();

 private:
  id_t m_id;
  // enables measuring the completion time of the notifyListener of this event
  bool m_enable_timing;
  uint64_t m_signalCounter = 0;
  MPSStream* m_stream = nullptr;
  MTLSharedEvent_t m_event = nullptr;
  MTLSharedEventListener* m_listener = nullptr;
  // used to sync the events created on this Stream with CPU
  std::mutex m_cpu_sync_mutex{};
  std::condition_variable m_cpu_sync_cv{};
  // CondVar predicate to sync the events created on this Stream with CPU
  bool m_cpu_sync_completed = false;
  // used to compute elapsed time
  uint64_t m_completion_time = 0;

  void recordLocked(bool syncEvent);
  bool waitLocked(bool syncEvent);
  bool notifyLocked(MTLSharedEventNotificationBlock block);
  void notifyCpuSync();
  static uint64_t getTime() {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
  }
};

typedef std::unique_ptr<MPSEvent, std::function<void(MPSEvent*)>> MPSEventPtr;

class MPSEventPool {
 public:
  explicit MPSEventPool(MPSStream* default_stream);
  ~MPSEventPool();

  MPSEventPtr acquireEvent(bool enable_timing, MPSStream* stream);
  void emptyCache();

  // these are mainly used for MPSHooks and torch.mps.Event() bindings
  id_t acquireEvent(bool enable_timing);
  void releaseEvent(id_t event_id);
  void recordEvent(id_t event_id, bool syncEvent);
  void waitForEvent(id_t event_id, bool syncEvent);
  void synchronizeEvent(id_t event_id);
  bool queryEvent(id_t event_id);
  // returns elapsed time between two recorded events in milliseconds
  double elapsedTime(id_t start_event_id, id_t end_event_id);

 private:
  MPSStream* m_default_stream = nullptr;
  std::recursive_mutex m_mutex;
  std::stack<std::unique_ptr<MPSEvent>> m_pool{};
  // dictionary to associate event IDs with event objects
  // used to retain in-use events out of the pool
  // for torch.mps.Event() bindings.
  std::unordered_map<id_t, MPSEventPtr> m_in_use_events{};
  uint64_t m_event_counter = 0;
  std::function<void(MPSEvent*)> m_default_deleter;

  MPSEvent* getInUseEvent(id_t event_id, bool locked = true);
};

// shared_ptr is used to get MPSEventPool destroyed after dependent instances
std::shared_ptr<MPSEventPool> getMPSEventPool();

} // namespace at::mps
