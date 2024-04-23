// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#ifdef _WIN32
#  include "opentelemetry/plugin/detail/dynamic_load_windows.h"
#else
#  include "opentelemetry/plugin/detail/dynamic_load_unix.h"
#endif
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace plugin
{

class Factory;

/**
 * Load an OpenTelemetry implementation as a plugin.
 * @param plugin the path to the plugin to load
 * @param error_message on failure this is set to an error message
 * @return a Factory that can be used to create OpenTelemetry objects or nullptr on failure.
 */
std::unique_ptr<Factory> LoadFactory(const char *plugin, std::string &error_message) noexcept;
}  // namespace plugin
OPENTELEMETRY_END_NAMESPACE
