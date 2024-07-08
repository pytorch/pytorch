/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <cstdint>

#pragma once

namespace torch {
namespace executor {

typedef uint32_t AllocatorID;
typedef int32_t ChainID;
typedef uint32_t DebugHandle;

/**
 * EventTracer is a class that users can inherit and implement to
 * log/serialize/stream etc. the profiling and debugging events that are
 * generated at runtime for a model. An example of this is the ETDump
 * implementation in the SDK codebase that serializes these events to a
 * flatbuffer.
 */
class EventTracer {};

struct EventTracerEntry {};

} // namespace executor
} // namespace torch
