/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <event_tracer.h>

/**
 * @file
 *
 * This file contains the hooks that are inserted across various parts of the
 * core runtime code to call into the EventTracer class for logging of profiling
 * and debugging events. Any calls made to the EventTracer from the runtime must
 * be made via these hooks.
 * Users shouldn't directly add these hooks in their code and it's meant only
 * for usage in ExecuTorch internal code.
 *
 * The benefit of defining these hooks is that we can easily control whether or
 * not we want to compile in the EventTracer code based on the status of the
 * ET_EVENT_TRACER_ENABLED flag.
 */

namespace torch {
namespace executor {
namespace internal {

/**
 * This class enables scope based profiling where needed using RAII.
 * Profiling will be started when the object is created and will end
 * when the object goes out of scope.
 */
class EventTracerProfileScope final {
 public:
  EventTracerProfileScope(EventTracer* event_tracer, const char* name) {
    event_tracer_ = event_tracer;
    if (event_tracer_ == nullptr) {
      return;
    }
    event_entry_ = event_tracer->start_profiling(name);
  }

  ~EventTracerProfileScope() {
    if (event_tracer_ == nullptr) {
      return;
    }
    event_tracer_->end_profiling(event_entry_);
  }

 private:
  EventTracer* event_tracer_;
  EventTracerEntry event_entry_;
};

/**
 * This class helps us set and then clear out the chain id and debug handle
 * values stored in the event tracer class using RAII. This is typically called
 * in the executor loop before entering the codegen layer to configure the chain
 * id and debug handle of the current instruction being executed.
 * After we return from the kernel execution we can then reset the chain id and
 * debug handle to defaults when this object goes out of scope.
 */
class EventTracerProfileInstructionScope final {
 public:
  EventTracerProfileInstructionScope(
      EventTracer* event_tracer,
      ChainID chain_idx,
      DebugHandle debug_handle) {
    event_tracer_ = event_tracer;
    if (event_tracer_ == nullptr) {
      return;
    }
    event_tracer_->set_chain_debug_handle(chain_idx, debug_handle);
  }

  ~EventTracerProfileInstructionScope() {
    if (event_tracer_ == nullptr) {
      return;
    }
    event_tracer_->set_chain_debug_handle(kUnsetChainId, kUnsetDebugHandle);
  }

 private:
  EventTracer* event_tracer_;
};

/**
 * Create a new event block with the specified name. Any events logged
 * after this will be associated with this new event block.
 */
inline void event_tracer_create_event_block(
    EventTracer* event_tracer,
    char const* name) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    event_tracer->create_event_block(name);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)name;
#endif
}

/**
 * Explicitly mark the beginning of a new profiling event. This returns
 * an instance of an EventTracerEntry object that the user needs to keep
 * around and pass into the corresponding event_tracer_end_profiling_event
 * call.
 */
inline EventTracerEntry event_tracer_begin_profiling_event(
    EventTracer* event_tracer,
    char const* name) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    return event_tracer->start_profiling(name);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)name;
#endif
  // There is no active tracer; this value will be ignored.
  return EventTracerEntry();
}

/**
 * Mark the end of a profiling event passing in the entry token
 * returned by a previous call to ET_EVENT_TRACER_BEGIN_PROFILING_EVENT.
 */
inline void event_tracer_end_profiling_event(
    EventTracer* event_tracer,
    EventTracerEntry event) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    event_tracer->end_profiling(event);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)event;
#endif
}

/**
 * Start the tracking of the allocator represented by this name and returns
 * an AllocatorID that will be used to track all subsequent allocations done by
 * this allocator.
 */
inline AllocatorID event_tracer_track_allocator(
    EventTracer* event_tracer,
    const char* name) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    return event_tracer->track_allocator(name);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)name;
#endif
  // There is no active tracer; this value will be ignored.
  return 0;
}

/// Log the allocation event done via the allocator represented by id.
inline void event_tracer_track_allocation(
    EventTracer* event_tracer,
    AllocatorID id,
    size_t size) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    event_tracer->track_allocation(id, size);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)id;
  (void)size;
#endif
}

} // namespace internal
} // namespace executor
} // namespace torch
