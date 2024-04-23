// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/key_value_iterable.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{

class Tracer;

/**
 * Creates new Tracer instances.
 */
class OPENTELEMETRY_EXPORT TracerProvider
{
public:
  virtual ~TracerProvider() = default;

#if OPENTELEMETRY_ABI_VERSION_NO >= 2

  /**
   * Gets or creates a named Tracer instance (ABI).
   *
   * @since ABI_VERSION 2
   *
   * @param[in] name Tracer instrumentation scope
   * @param[in] version Instrumentation scope version
   * @param[in] schema_url Instrumentation scope schema URL
   * @param[in] attributes Instrumentation scope attributes (optional, may be nullptr)
   */
  virtual nostd::shared_ptr<Tracer> GetTracer(
      nostd::string_view name,
      nostd::string_view version,
      nostd::string_view schema_url,
      const common::KeyValueIterable *attributes) noexcept = 0;

  /**
   * Gets or creates a named Tracer instance (API helper).
   *
   * @since ABI_VERSION 2
   *
   * @param[in] name Tracer instrumentation scope
   * @param[in] version Instrumentation scope version, optional
   * @param[in] schema_url Instrumentation scope schema URL, optional
   */
  nostd::shared_ptr<Tracer> GetTracer(nostd::string_view name,
                                      nostd::string_view version    = "",
                                      nostd::string_view schema_url = "")
  {
    return GetTracer(name, version, schema_url, nullptr);
  }

  /**
   * Gets or creates a named Tracer instance (API helper).
   *
   * @since ABI_VERSION 2
   *
   * @param[in] name Tracer instrumentation scope
   * @param[in] version Instrumentation scope version
   * @param[in] schema_url Instrumentation scope schema URL
   * @param[in] attributes Instrumentation scope attributes
   */
  nostd::shared_ptr<Tracer> GetTracer(
      nostd::string_view name,
      nostd::string_view version,
      nostd::string_view schema_url,
      std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes)
  {
    /* Build a container from std::initializer_list. */
    nostd::span<const std::pair<nostd::string_view, common::AttributeValue>> attributes_span{
        attributes.begin(), attributes.end()};

    /* Build a view on the container. */
    common::KeyValueIterableView<
        nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>>
        iterable_attributes{attributes_span};

    /* Add attributes using the view. */
    return GetTracer(name, version, schema_url, &iterable_attributes);
  }

  /**
   * Gets or creates a named Tracer instance (API helper).
   *
   * @since ABI_VERSION 2
   *
   * @param[in] name Tracer instrumentation scope
   * @param[in] version Instrumentation scope version
   * @param[in] schema_url Instrumentation scope schema URL
   * @param[in] attributes Instrumentation scope attributes container
   */
  template <class T,
            nostd::enable_if_t<common::detail::is_key_value_iterable<T>::value> * = nullptr>
  nostd::shared_ptr<Tracer> GetTracer(nostd::string_view name,
                                      nostd::string_view version,
                                      nostd::string_view schema_url,
                                      const T &attributes)
  {
    /* Build a view on the container. */
    common::KeyValueIterableView<T> iterable_attributes(attributes);

    /* Add attributes using the view. */
    return GetTracer(name, version, schema_url, &iterable_attributes);
  }

#else

  /**
   * Gets or creates a named tracer instance.
   *
   * Optionally a version can be passed to create a named and versioned tracer
   * instance.
   */
  virtual nostd::shared_ptr<Tracer> GetTracer(nostd::string_view name,
                                              nostd::string_view version    = "",
                                              nostd::string_view schema_url = "") noexcept = 0;
#endif
};
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
