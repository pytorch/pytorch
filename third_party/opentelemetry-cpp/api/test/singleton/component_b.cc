// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/version.h"

#include "component_b.h"

namespace trace = opentelemetry::trace;
namespace nostd = opentelemetry::nostd;

static nostd::shared_ptr<trace::Tracer> get_tracer()
{
  auto provider = trace::Provider::GetTracerProvider();
  return provider->GetTracer("B", "20.2");
}

static void f1()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("B::f1"));
}

static void f2()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("B::f2"));

  f1();
  f1();
}

void do_something_in_b()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("B::library"));

  f2();
}
