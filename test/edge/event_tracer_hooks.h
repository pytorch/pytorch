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
  EventTracerProfileScope(EventTracer* event_tracer, const char* name) {};

  ~EventTracerProfileScope() {};

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
      DebugHandle debug_handle) {};

  ~EventTracerProfileInstructionScope() {};

 private:
  EventTracer* event_tracer_;
};

void event_tracer_log_evalue(EventTracer* event_tracer, EValue& evalue) {
  (void)evalue;
}

} // namespace internal
} // namespace executor
} // namespace torch
