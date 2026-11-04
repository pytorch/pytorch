/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>

#include "ActivityLoggerFactory.h"
#include "ActivityTraceInterface.h"
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
    memLogger_->log(*memLogger_->getChromeLogger());
  }

 private:
  // Activities are logged into a buffer
  std::unique_ptr<MemoryTraceLogger> memLogger_;

  // Alternative logger used by save() if protocol prefix is specified
  const ActivityLoggerFactory& loggerFactory_;
};

} // namespace libkineto
