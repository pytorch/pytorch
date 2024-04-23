// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/baggage/baggage.h"
#include "opentelemetry/baggage/baggage_context.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace baggage
{
namespace propagation
{

class BaggagePropagator : public context::propagation::TextMapPropagator
{
public:
  void Inject(context::propagation::TextMapCarrier &carrier,
              const context::Context &context) noexcept override
  {
    auto baggage = baggage::GetBaggage(context);
    auto header  = baggage->ToHeader();
    if (header.size())
    {
      carrier.Set(kBaggageHeader, header);
    }
  }

  context::Context Extract(const context::propagation::TextMapCarrier &carrier,
                           context::Context &context) noexcept override
  {
    nostd::string_view baggage_str = carrier.Get(baggage::kBaggageHeader);
    auto baggage                   = baggage::Baggage::FromHeader(baggage_str);

    if (baggage->ToHeader().size())
    {
      return baggage::SetBaggage(context, baggage);
    }
    else
    {
      return context;
    }
  }

  bool Fields(nostd::function_ref<bool(nostd::string_view)> callback) const noexcept override
  {
    return callback(kBaggageHeader);
  }
};
}  // namespace propagation
}  // namespace baggage
OPENTELEMETRY_END_NAMESPACE
