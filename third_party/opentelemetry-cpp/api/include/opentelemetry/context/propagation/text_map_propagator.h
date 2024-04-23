// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/context/context.h"
#include "opentelemetry/nostd/function_ref.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace context
{
namespace propagation
{

// TextMapCarrier is the storage medium used by TextMapPropagator.
class TextMapCarrier
{
public:
  // returns the value associated with the passed key.
  virtual nostd::string_view Get(nostd::string_view key) const noexcept = 0;

  // stores the key-value pair.
  virtual void Set(nostd::string_view key, nostd::string_view value) noexcept = 0;

  /* list of all the keys in the carrier.
   By default, it returns true without invoking callback */
  virtual bool Keys(nostd::function_ref<bool(nostd::string_view)> /* callback */) const noexcept
  {
    return true;
  }
  virtual ~TextMapCarrier() = default;
};

// The TextMapPropagator class provides an interface that enables extracting and injecting
// context into carriers that travel in-band across process boundaries. HTTP frameworks and clients
// can integrate with TextMapPropagator by providing the object containing the
// headers, and a getter and setter function for the extraction and
// injection of values, respectively.

class TextMapPropagator
{
public:
  // Returns the context that is stored in the carrier with the TextMapCarrier as extractor.
  virtual context::Context Extract(const TextMapCarrier &carrier,
                                   context::Context &context) noexcept = 0;

  // Sets the context for carrier with self defined rules.
  virtual void Inject(TextMapCarrier &carrier, const context::Context &context) noexcept = 0;

  // Gets the fields set in the carrier by the `inject` method
  virtual bool Fields(nostd::function_ref<bool(nostd::string_view)> callback) const noexcept = 0;

  virtual ~TextMapPropagator() = default;
};
}  // namespace propagation
}  // namespace context
OPENTELEMETRY_END_NAMESPACE
