#pragma once

#include <c10/util/Optional.h>
#include <string>

namespace c10d {

enum class TORCH_API EventKind { CollectionStart, CollectionEnd };

struct TORCH_API EventInfo {
  EventKind event_kind;
  std::string pg_name;
  std::string backend;
  int64_t sequence_number;
  std::string operation;
  int64_t timestamp;
  c10::optional<float> duration_ms;
  c10::optional<std::string> error_message;
};

typedef std::function<void(const EventInfo&)> CollectiveEventCallback;

/**
 * Register a callback that is invoked whenever a collective event happens.
 *
 * Locking:
 *  Registration takes a subsystem specific lock.
 *  callback invocation happens when the same lock held.
 *
 * Callbacks must not block or run for a long period of time.
 * They are invoked from threads are part of PyTorch's critical distributed
 * infrastrucute. The recomended pattern is for callbacks to enqueue the events
 * on some queue and have a separate thread process those events. If the
 * callback deadlocks, it will hang the whole process and stop PyTorch from
 * detecting failures. Do not call into CUDA.
 *
 * n.b. Currently the user needs to call ProcessGroup::enableCollectivesTiming
 *   to enable start event collection.
 *
 * @param callback  callback to invoke on collective every event.
 */

TORCH_API void register_collective_callback(CollectiveEventCallback&& callback);

namespace details {

// TODO do we want to expose something else here?
// TORCH_API bool dequeue_c10d_event(EventInfo& evt);
TORCH_API void enqueue_c10d_event(EventInfo&& evt);

} // namespace details
} // namespace c10d
