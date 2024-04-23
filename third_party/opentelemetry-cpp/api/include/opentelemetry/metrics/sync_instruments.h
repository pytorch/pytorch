// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/nostd/span.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/type_traits.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace metrics
{

class SynchronousInstrument
{
public:
  SynchronousInstrument()          = default;
  virtual ~SynchronousInstrument() = default;
};

/* A Counter instrument that adds values. */
template <class T>
class Counter : public SynchronousInstrument
{

public:
  /**
   * Record a value
   *
   * @param value The increment amount. MUST be non-negative.
   */
  virtual void Add(T value) noexcept = 0;

  /**
   * Record a value
   *
   * @param value The increment amount. MUST be non-negative.
   * @param context The explicit context to associate with this measurement.
   */
  virtual void Add(T value, const context::Context &context) noexcept = 0;

  /**
   * Record a value with a set of attributes.
   *
   * @param value The increment amount. MUST be non-negative.
   * @param attributes A set of attributes to associate with the value.
   */

  virtual void Add(T value, const common::KeyValueIterable &attributes) noexcept = 0;

  /**
   * Record a value with a set of attributes.
   *
   * @param value The increment amount. MUST be non-negative.
   * @param attributes A set of attributes to associate with the value.
   * @param context The explicit context to associate with this measurement.
   */
  virtual void Add(T value,
                   const common::KeyValueIterable &attributes,
                   const context::Context &context) noexcept = 0;

  template <class U,
            nostd::enable_if_t<common::detail::is_key_value_iterable<U>::value> * = nullptr>
  void Add(T value, const U &attributes) noexcept
  {
    this->Add(value, common::KeyValueIterableView<U>{attributes});
  }

  template <class U,
            nostd::enable_if_t<common::detail::is_key_value_iterable<U>::value> * = nullptr>
  void Add(T value, const U &attributes, const context::Context &context) noexcept
  {
    this->Add(value, common::KeyValueIterableView<U>{attributes}, context);
  }

  void Add(T value,
           std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>>
               attributes) noexcept
  {
    this->Add(value, nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                         attributes.begin(), attributes.end()});
  }

  void Add(T value,
           std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes,
           const context::Context &context) noexcept
  {
    this->Add(value,
              nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                  attributes.begin(), attributes.end()},
              context);
  }
};

/** A histogram instrument that records values. */

template <class T>
class Histogram : public SynchronousInstrument
{
public:
#if OPENTELEMETRY_ABI_VERSION_NO >= 2
  /**
   * @since ABI_VERSION 2
   * Records a value.
   *
   * @param value The measurement value. MUST be non-negative.
   */
  virtual void Record(T value) noexcept = 0;

  /**
   * @since ABI_VERSION 2
   * Records a value with a set of attributes.
   *
   * @param value The measurement value. MUST be non-negative.
   * @param attribute A set of attributes to associate with the value.
   */
  virtual void Record(T value, const common::KeyValueIterable &attribute) noexcept = 0;

  template <class U,
            nostd::enable_if_t<common::detail::is_key_value_iterable<U>::value> * = nullptr>
  void Record(T value, const U &attributes) noexcept
  {
    this->Record(value, common::KeyValueIterableView<U>{attributes});
  }

  void Record(T value,
              std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>>
                  attributes) noexcept
  {
    this->Record(value, nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                            attributes.begin(), attributes.end()});
  }
#endif

  /**
   * Records a value.
   *
   * @param value The measurement value. MUST be non-negative.
   * @param context The explicit context to associate with this measurement.
   */
  virtual void Record(T value, const context::Context &context) noexcept = 0;

  /**
   * Records a value with a set of attributes.
   *
   * @param value The measurement value. MUST be non-negative.
   * @param attributes A set of attributes to associate with the value..
   * @param context The explicit context to associate with this measurement.
   */
  virtual void Record(T value,
                      const common::KeyValueIterable &attributes,
                      const context::Context &context) noexcept = 0;

  template <class U,
            nostd::enable_if_t<common::detail::is_key_value_iterable<U>::value> * = nullptr>
  void Record(T value, const U &attributes, const context::Context &context) noexcept
  {
    this->Record(value, common::KeyValueIterableView<U>{attributes}, context);
  }

  void Record(
      T value,
      std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes,
      const context::Context &context) noexcept
  {
    this->Record(value,
                 nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                     attributes.begin(), attributes.end()},
                 context);
  }
};

/** An up-down-counter instrument that adds or reduce values. */

template <class T>
class UpDownCounter : public SynchronousInstrument
{
public:
  /**
   * Record a value.
   *
   * @param value The increment amount. May be positive, negative or zero.
   */
  virtual void Add(T value) noexcept = 0;

  /**
   * Record a value.
   *
   * @param value The increment amount. May be positive, negative or zero.
   * @param context The explicit context to associate with this measurement.
   */
  virtual void Add(T value, const context::Context &context) noexcept = 0;

  /**
   * Record a value with a set of attributes.
   *
   * @param value The increment amount. May be positive, negative or zero.
   * @param attributes A set of attributes to associate with the count.
   */
  virtual void Add(T value, const common::KeyValueIterable &attributes) noexcept = 0;

  /**
   * Record a value with a set of attributes.
   *
   * @param value The increment amount. May be positive, negative or zero.
   * @param attributes A set of attributes to associate with the count.
   * @param context The explicit context to associate with this measurement.
   */
  virtual void Add(T value,
                   const common::KeyValueIterable &attributes,
                   const context::Context &context) noexcept = 0;

  template <class U,
            nostd::enable_if_t<common::detail::is_key_value_iterable<U>::value> * = nullptr>
  void Add(T value, const U &attributes) noexcept
  {
    this->Add(value, common::KeyValueIterableView<U>{attributes});
  }

  template <class U,
            nostd::enable_if_t<common::detail::is_key_value_iterable<U>::value> * = nullptr>
  void Add(T value, const U &attributes, const context::Context &context) noexcept
  {
    this->Add(value, common::KeyValueIterableView<U>{attributes}, context);
  }

  void Add(T value,
           std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>>
               attributes) noexcept
  {
    this->Add(value, nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                         attributes.begin(), attributes.end()});
  }

  void Add(T value,
           std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes,
           const context::Context &context) noexcept
  {
    this->Add(value,
              nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                  attributes.begin(), attributes.end()},
              context);
  }
};

}  // namespace metrics
OPENTELEMETRY_END_NAMESPACE
