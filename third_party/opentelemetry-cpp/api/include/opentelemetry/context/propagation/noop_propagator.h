// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace context
{
namespace propagation
{

/**
 * No-op implementation TextMapPropagator
 */
class NoOpPropagator : public TextMapPropagator
{
public:
  /** Noop extract function does nothing and returns the input context */
  context::Context Extract(const TextMapCarrier & /*carrier*/,
                           context::Context &context) noexcept override
  {
    return context;
  }

  /** Noop inject function does nothing */
  void Inject(TextMapCarrier & /*carrier*/,
              const context::Context & /* context */) noexcept override
  {}

  bool Fields(nostd::function_ref<bool(nostd::string_view)> /* callback */) const noexcept override
  {
    return true;
  }
};
}  // namespace propagation
}  // namespace context
OPENTELEMETRY_END_NAMESPACE
