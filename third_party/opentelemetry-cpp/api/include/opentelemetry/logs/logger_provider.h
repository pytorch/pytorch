// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/key_value_iterable.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/span.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/type_traits.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace logs
{

class Logger;

/**
 * Creates new Logger instances.
 */
class OPENTELEMETRY_EXPORT LoggerProvider
{
public:
  virtual ~LoggerProvider() = default;

  /**
   * Gets or creates a named Logger instance.
   *
   * Optionally a version can be passed to create a named and versioned Logger
   * instance.
   *
   * Optionally a configuration file name can be passed to create a configuration for
   * the Logger instance.
   *
   */

  virtual nostd::shared_ptr<Logger> GetLogger(
      nostd::string_view logger_name,
      nostd::string_view library_name            = "",
      nostd::string_view library_version         = "",
      nostd::string_view schema_url              = "",
      const common::KeyValueIterable &attributes = common::NoopKeyValueIterable()) = 0;

  nostd::shared_ptr<Logger> GetLogger(
      nostd::string_view logger_name,
      nostd::string_view library_name,
      nostd::string_view library_version,
      nostd::string_view schema_url,
      std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes)
  {
    return GetLogger(logger_name, library_name, library_version, schema_url,
                     nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                         attributes.begin(), attributes.end()});
  }

  template <class T,
            nostd::enable_if_t<common::detail::is_key_value_iterable<T>::value> * = nullptr>
  nostd::shared_ptr<Logger> GetLogger(nostd::string_view logger_name,
                                      nostd::string_view library_name,
                                      nostd::string_view library_version,
                                      nostd::string_view schema_url,
                                      const T &attributes)
  {
    return GetLogger(logger_name, library_name, library_version, schema_url,
                     common::KeyValueIterableView<T>(attributes));
  }
};
}  // namespace logs
OPENTELEMETRY_END_NAMESPACE
