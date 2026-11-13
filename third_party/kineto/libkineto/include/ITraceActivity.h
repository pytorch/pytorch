/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ActivityType.h"
#include "TypedMetadata.h"

namespace libkineto {

class ActivityLogger;
struct TraceSpan;

// Generic activity interface is borrowed from tensorboard protobuf format.
struct ITraceActivity {
  virtual ~ITraceActivity() = default;
  // Device is a physical or logical entity, e.g. CPU, GPU or process
  [[nodiscard]] virtual int64_t deviceId() const = 0;
  // A resource is something on the device, h/w thread,
  // functional units etc.
  [[nodiscard]] virtual int64_t resourceId() const = 0;
  // s/w thread
  [[nodiscard]] virtual int32_t getThreadId() const = 0;
  // Start timestamp in nanoseconds
  [[nodiscard]] virtual int64_t timestamp() const = 0;
  // Duration in nanoseconds
  [[nodiscard]] virtual int64_t duration() const = 0;
  // Used to link up async activities
  [[nodiscard]] virtual int64_t correlationId() const = 0;
  // Part of a flow, identified by flow id and type
  [[nodiscard]] virtual int flowType() const = 0;
  [[nodiscard]] virtual int64_t flowId() const = 0;
  [[nodiscard]] virtual bool flowStart() const = 0;
  [[nodiscard]] virtual ActivityType type() const = 0;
  [[nodiscard]] virtual const std::string name() const = 0;
  // Optional linked activity
  [[nodiscard]] virtual const ITraceActivity* linkedActivity() const = 0;
  // Optional containing trace object
  [[nodiscard]] virtual const TraceSpan* traceSpan() const = 0;
  // Log activity
  virtual void log(ActivityLogger& logger) const = 0;
  // Return json formatted metadata
  // FIXME: Return iterator to dynamic type map here instead
  [[nodiscard]] virtual const std::string metadataJson() const = 0;
  // Write structured metadata to a consumer.
  virtual void visitTypedMetadata(
      [[maybe_unused]] ITypedMetadataVisitor& visitor) const {}
  // Return the metadata value in string format with key
  // @lint-ignore CLANGTIDY: clang-diagnostic-unused-parameter
  [[nodiscard]] virtual const std::string getMetadataValue(
      [[maybe_unused]] const std::string& key) const {
    return "";
  }
  // Return typed counter values (name, value) for activities with
  // floating-point metadata that should not be round-tripped through strings.
  [[nodiscard]] virtual const std::vector<std::pair<std::string, double>>&
  counterValues() const {
    static const std::vector<std::pair<std::string, double>> kEmpty;
    return kEmpty;
  }

  static int64_t nsToUs(int64_t ns) {
    // It's important that this conversion is the same everywhere.
    // No rounding!
    return ns / 1000;
  }
};

} // namespace libkineto
