// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/variant.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace baggage
{
class Baggage;
}  // namespace baggage

namespace trace
{
class Span;
class SpanContext;
}  // namespace trace

namespace context
{
using ContextValue = nostd::variant<nostd::monostate,
                                    bool,
                                    int64_t,
                                    uint64_t,
                                    double,
                                    nostd::shared_ptr<trace::Span>,
                                    nostd::shared_ptr<trace::SpanContext>,
                                    nostd::shared_ptr<baggage::Baggage>>;
}  // namespace context
OPENTELEMETRY_END_NAMESPACE
