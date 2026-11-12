/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <memory>
#include <string>
#include <system_error>

#include "ActivityLoggerFactory.h"
#include "ActivityTraceInterface.h"
#include "ThrowUtil.h"
#include "output_json.h"
#include "output_membuf.h"

namespace libkineto {

class ActivityTrace : public ActivityTraceInterface {
 public:
  ActivityTrace(
      std::unique_ptr<MemoryTraceLogger> tmpLogger,
      const ActivityLoggerFactory& factory)
      : memLogger_(std::move(tmpLogger)), loggerFactory_(factory) {}

  const std::vector<const ITraceActivity*>* activities() override {
    return memLogger_->traceActivities();
  }

  void save(const std::string& url) override {
    std::string prefix;
    // if no protocol is specified, default to file
    if (url.find("://") == url.npos) {
      prefix = "file://";
    }
    memLogger_->setChromeLogger(loggerFactory_.makeLogger(prefix + url));
    const auto logger = memLogger_->getChromeLogger();
    memLogger_->log(*logger);
    // Logging only reads the buffered trace, so the caller can retry.
    if (const auto error = logger->writeError()) {
      KINETO_THROW(
          std::system_error,
          error,
          fmt::format("Failed to save trace to '{}'", url));
    }
  }

 private:
  // Activities are logged into a buffer
  std::unique_ptr<MemoryTraceLogger> memLogger_;

  // Alternative logger used by save() if protocol prefix is specified
  const ActivityLoggerFactory& loggerFactory_;
};

} // namespace libkineto
