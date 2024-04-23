// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/nostd/function_ref.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace common
{
/**
 * Supports internal iteration over a collection of key-value pairs.
 */
class KeyValueIterable
{
public:
  virtual ~KeyValueIterable() = default;

  /**
   * Iterate over key-value pairs
   * @param callback a callback to invoke for each key-value. If the callback returns false,
   * the iteration is aborted.
   * @return true if every key-value pair was iterated over
   */
  virtual bool ForEachKeyValue(nostd::function_ref<bool(nostd::string_view, common::AttributeValue)>
                                   callback) const noexcept = 0;

  /**
   * @return the number of key-value pairs
   */
  virtual size_t size() const noexcept = 0;
};

/**
 * Supports internal iteration over a collection of key-value pairs.
 */
class NoopKeyValueIterable : public KeyValueIterable
{
public:
  ~NoopKeyValueIterable() override = default;

  /**
   * Iterate over key-value pairs
   * @param callback a callback to invoke for each key-value. If the callback returns false,
   * the iteration is aborted.
   * @return true if every key-value pair was iterated over
   */
  bool ForEachKeyValue(
      nostd::function_ref<bool(nostd::string_view, common::AttributeValue)>) const noexcept override
  {
    return true;
  }

  /**
   * @return the number of key-value pairs
   */
  size_t size() const noexcept override { return 0; }
};

}  // namespace common
OPENTELEMETRY_END_NAMESPACE
