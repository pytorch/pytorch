#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

#include <c10/util/variant.h>

namespace torch {
namespace monitor {

// metadata_value_t is the type for Event metadata values.
using metadata_value_t = c10::variant<std::string, double, int64_t, bool>;

// Event represents a single event that can be logged out to an external
// tracker. This does acquire a lock on logging so should be used relatively
// infrequently to avoid performance issues.
struct Event {
  // type is the type of the event. This is a static string that's used to
  // differentiate between event types for programmatic access. The type should
  // be in the format of a fully qualified Python-style class name.
  // Ex: torch.monitor.MonitorEvent
  std::string type;

  // message is a human readable name. This is optional for machine intended
  // stats.
  std::string message;

  // timestamp is a timestamp relative to the Unix epoch time.
  std::chrono::system_clock::time_point timestamp;

  // metadata contains rich information about the event. The contents are event
  // specific so you should check the type to ensure it's what you expect before
  // accessing the metadata.
  //
  // NOTE: these events are not versioned and it's up to the consumer of the
  // events to check the fields to ensure backwards compatibility.
  std::unordered_map<std::string, metadata_value_t> metadata;
};

inline bool operator==(const Event& lhs, const Event& rhs) {
  return lhs.type == rhs.type && lhs.message == rhs.message &&
      lhs.timestamp == rhs.timestamp && lhs.metadata == rhs.metadata;
}

// EventHandler represents an abstract event handler that can be registered to
// capture events. Every time an event is logged every handler will be called
// with the events contents.
//
// NOTE: The handlers should avoid any IO, blocking calls or heavy computation
// as this may block the main thread and cause performance issues.
class EventHandler {
 public:
  virtual ~EventHandler() = default;

  // handle needs to be implemented to handle the events. This may be called
  // from multiple threads so needs to be thread safe.
  virtual void handle(const Event& e) = 0;
};

// logEvent calls each registered event handler with the event. This method can
// be called from concurrently from multiple threads.
void logEvent(const Event& e);

// registerEventHandler registers an EventHandler so it receives any logged
// events. Typically an EventHandler will be registered during program
// setup and unregistered at the end.
void registerEventHandler(std::shared_ptr<EventHandler> p);

// unregisterEventHandler unregisters the event handler pointed to by the
// shared_ptr.
void unregisterEventHandler(const std::shared_ptr<EventHandler>& p);

} // namespace monitor
} // namespace torch
