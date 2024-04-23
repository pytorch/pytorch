// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace logs
{

/**
 * EventId class which acts the Id of the event with an optional name.
 */
class EventId
{
public:
  EventId(int64_t id, nostd::string_view name) noexcept : id_{id}
  {
    name_ = nostd::unique_ptr<char[]>{new char[name.length() + 1]};
    std::copy(name.begin(), name.end(), name_.get());
    name_.get()[name.length()] = 0;
  }

  EventId(int64_t id) noexcept : id_{id}, name_{nullptr} {}

public:
  int64_t id_;
  nostd::unique_ptr<char[]> name_;
};

}  // namespace logs
OPENTELEMETRY_END_NAMESPACE
