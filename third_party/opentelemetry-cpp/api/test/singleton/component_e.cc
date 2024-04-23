// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/version.h"

#define BUILD_COMPONENT_E

#include "component_e.h"

namespace trace = opentelemetry::trace;
namespace nostd = opentelemetry::nostd;

static nostd::shared_ptr<trace::Tracer> get_tracer()
{
  auto provider = trace::Provider::GetTracerProvider();
  return provider->GetTracer("E", "50.5");
}

static void f1()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("E::f1"));
}

static void f2()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("E::f2"));

  f1();
  f1();
}

void do_something_in_e()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("E::library"));

  f2();
}
