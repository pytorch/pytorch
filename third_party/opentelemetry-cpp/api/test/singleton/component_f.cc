// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/version.h"

#define BUILD_COMPONENT_F

#include "component_f.h"

namespace trace = opentelemetry::trace;
namespace nostd = opentelemetry::nostd;

static nostd::shared_ptr<trace::Tracer> get_tracer()
{
  auto provider = trace::Provider::GetTracerProvider();
  return provider->GetTracer("F", "60.6");
}

static void f1()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("F::f1"));
}

static void f2()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("F::f2"));

  f1();
  f1();
}

void do_something_in_f()
{
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("F::library"));

  f2();
}
