// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/trace/span_metadata.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{

class Span;

/**
 * Controls how long a span is active.
 *
 * On creation of the Scope object, the given span is set to the currently
 * active span. On destruction, the given span is ended and the previously
 * active span will be the currently active span again.
 */
class Scope final
{
public:
  /**
   * Initialize a new scope.
   * @param span the given span will be set as the currently active span.
   */
  Scope(const nostd::shared_ptr<Span> &span) noexcept
      : token_(context::RuntimeContext::Attach(
            context::RuntimeContext::GetCurrent().SetValue(kSpanKey, span)))
  {}

private:
  nostd::unique_ptr<context::Token> token_;
};

}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
