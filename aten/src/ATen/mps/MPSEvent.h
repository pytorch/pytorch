//  Copyright © 2023 Apple Inc.

#pragma once

#include <ATen/mps/MPSStream.h>
#include <c10/util/intrusive_ptr.h>
#include <atomic>
#include <stack>

namespace at::mps {

// NOTE: don't create instances of this class directly.
// Use MPSEventPool to acquire instances of MPSEventPtr.
class MPSEvent {
 public:
  explicit MPSEvent(id_t ID, MPSStream* stream, bool enable_timing);
  ~MPSEvent();

  // records an event on the stream
  void record(bool needsLock, bool syncEvent = false);
  // makes all future work submitted to the stream wait for this event.
  bool wait(bool needsLock, bool syncEvent = false);
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
  // returns whether timing is enabled for this event
  bool isTimingEnabled() const {
    return m_enable_timing;
  }
  // returns whether this event has been recorded since it was last acquired
  bool isRecorded() const {
    return m_recorded.load();
  }

 private:
  id_t m_id;
  // Enables measuring the GPU completion time of this event.
  bool m_enable_timing;
  // Tracks whether this event has been recorded since it was last acquired.
  std::atomic<bool> m_recorded{false};
  // Tracks the latest value encoded for the Metal shared event.
  std::atomic<uint64_t> m_signalCounter{0};
  // Stream on which this event is recorded.
  MPSStream* m_stream = nullptr;
  // Metal event used to signal and wait for GPU progress.
  MTLSharedEvent_t m_event = nullptr;
  // Guards timing state shared with command-buffer completion handlers.
  std::mutex m_cpu_sync_mutex{};
  std::condition_variable m_cpu_sync_cv{};
  // Each timing record receives a monotonically increasing generation. The
  // completed generation identifies the newest handler whose timestamp is in
  // m_completion_time. Generations survive pool reuse so a delayed handler
  // cannot satisfy a wait for, or overwrite the timestamp of, a newer record.
  uint64_t m_timing_generation = 0;
  uint64_t m_completed_timing_generation = 0;
  // used to compute elapsed time
  double m_completion_time = 0.0;

  void recordLocked(bool syncEvent);
  bool waitLocked(bool syncEvent);
  void notifyCpuSync(uint64_t timingGeneration, double completionTime);
  // assumes timing is enabled and waits for the latest recording's timestamp
  double waitForTiming();

  friend class MPSEventPool;
};

class MPSEventPool;

// Refcounted handle to an MPSEvent. When all `intrusive_ptr`s to the same event
// are destroyed, the event is returned to the event pool instead of being
// destroyed, to avoid some overhead of creating new events.
class MPSEventPtrTarget : public c10::intrusive_ptr_target {
 public:
  MPSEventPtrTarget(MPSEvent* event, MPSEventPool* pool)
      : m_event(event), m_pool(pool) {}
  ~MPSEventPtrTarget() override;

  MPSEvent* get() const {
    return m_event;
  }

  void record(bool needsLock, bool syncEvent = false) {
    m_event->record(needsLock, syncEvent);
  }
  bool synchronize() {
    return m_event->synchronize();
  }
  id_t getID() const {
    return m_event->getID();
  }

 private:
  MPSEvent* m_event;
  MPSEventPool* m_pool;
};

using MPSEventPtr = c10::intrusive_ptr<MPSEventPtrTarget>;

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

  friend class MPSEventPtrTarget;
  void returnEventToPool(MPSEvent* event);

  MPSEvent* getInUseEvent(id_t event_id, bool locked = true);
};

// shared_ptr is used to get MPSEventPool destroyed after dependent instances
std::shared_ptr<MPSEventPool> getMPSEventPool();

} // namespace at::mps
