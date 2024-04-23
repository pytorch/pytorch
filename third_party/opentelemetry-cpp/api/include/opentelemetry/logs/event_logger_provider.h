// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace logs
{

class EventLogger;
class Logger;

/**
 * Creates new EventLogger instances.
 */
class EventLoggerProvider
{
public:
  virtual ~EventLoggerProvider() = default;

  /**
   * Creates a named EventLogger instance.
   *
   */

  virtual nostd::shared_ptr<EventLogger> CreateEventLogger(
      nostd::shared_ptr<Logger> delegate_logger,
      nostd::string_view event_domain) noexcept = 0;
};
}  // namespace logs
OPENTELEMETRY_END_NAMESPACE
