// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace plugin
{
/**
 * Manage the ownership of a dynamically loaded library.
 */
class DynamicLibraryHandle
{
public:
  virtual ~DynamicLibraryHandle() = default;
};
}  // namespace plugin
OPENTELEMETRY_END_NAMESPACE
