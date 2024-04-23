// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace plugin
{
/**
 * LoaderInfo describes the versioning of the loader.
 *
 * Plugins can check against this information and properly error out if they were built against an
 * incompatible OpenTelemetry API.
 */
struct LoaderInfo
{
  nostd::string_view opentelemetry_version     = OPENTELEMETRY_VERSION;
  nostd::string_view opentelemetry_abi_version = OPENTELEMETRY_ABI_VERSION;
};
}  // namespace plugin
OPENTELEMETRY_END_NAMESPACE
