// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/noop.h"
#include "opentelemetry/trace/span_id.h"
#include "opentelemetry/trace/trace_id.h"

#include <cstdint>

#include <benchmark/benchmark.h>

using opentelemetry::trace::SpanContext;
namespace trace_api = opentelemetry::trace;
namespace nostd     = opentelemetry::nostd;
namespace context   = opentelemetry::context;

namespace
{

std::shared_ptr<trace_api::Tracer> initTracer()
{
  return std::shared_ptr<trace_api::Tracer>(new trace_api::NoopTracer());
}

// Test to measure performance for span creation
void BM_SpanCreation(benchmark::State &state)
{
  auto tracer = initTracer();
  while (state.KeepRunning())
  {
    auto span = tracer->StartSpan("span");
    span->End();
  }
}
BENCHMARK(BM_SpanCreation);

// Test to measure performance for single span creation with scope
void BM_SpanCreationWithScope(benchmark::State &state)
{
  auto tracer = initTracer();
  while (state.KeepRunning())
  {
    auto span  = tracer->StartSpan("span");
    auto scope = tracer->WithActiveSpan(span);
    span->End();
  }
}
BENCHMARK(BM_SpanCreationWithScope);

// Test to measure performance for nested span creation with scope
void BM_NestedSpanCreationWithScope(benchmark::State &state)
{
  auto tracer = initTracer();
  while (state.KeepRunning())
  {
    auto o_span  = tracer->StartSpan("outer");
    auto o_scope = tracer->WithActiveSpan(o_span);
    {
      auto i_span  = tracer->StartSpan("inner");
      auto i_scope = tracer->WithActiveSpan(i_span);
      {
        auto im_span  = tracer->StartSpan("innermost");
        auto im_scope = tracer->WithActiveSpan(im_span);
        im_span->End();
      }
      i_span->End();
    }
    o_span->End();
  }
}

BENCHMARK(BM_NestedSpanCreationWithScope);

// Test to measure performance for nested span creation with manual span context management
void BM_SpanCreationWithManualSpanContextPropagation(benchmark::State &state)
{
  auto tracer              = initTracer();
  constexpr uint8_t buf1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  trace_api::SpanId span_id(buf1);
  constexpr uint8_t buf2[] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
  trace_api::TraceId trace_id(buf2);

  while (state.KeepRunning())
  {
    auto outer_span = nostd::shared_ptr<trace_api::Span>(
        new trace_api::DefaultSpan(SpanContext(trace_id, span_id, trace_api::TraceFlags(), false)));
    trace_api::StartSpanOptions options;
    options.parent          = outer_span->GetContext();
    auto inner_span         = tracer->StartSpan("inner", options);
    auto inner_span_context = inner_span->GetContext();
    options.parent          = inner_span_context;
    auto innermost_span     = tracer->StartSpan("innermost", options);
    innermost_span->End();
    inner_span->End();
  }
}
BENCHMARK(BM_SpanCreationWithManualSpanContextPropagation);

// Test to measure performance for nested span creation with context propagation
void BM_SpanCreationWitContextPropagation(benchmark::State &state)
{
  auto tracer              = initTracer();
  constexpr uint8_t buf1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  trace_api::SpanId span_id(buf1);
  constexpr uint8_t buf2[] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
  trace_api::TraceId trace_id(buf2);

  while (state.KeepRunning())
  {
    auto current_ctx        = context::RuntimeContext::GetCurrent();
    auto outer_span_context = SpanContext(trace_id, span_id, trace_api::TraceFlags(), false);
    auto outer_span =
        nostd::shared_ptr<trace_api::Span>(new trace_api::DefaultSpan(outer_span_context));
    trace_api::SetSpan(current_ctx, outer_span);
    auto inner_child = tracer->StartSpan("inner");
    auto inner_scope = tracer->WithActiveSpan(inner_child);
    {
      auto innermost_child = tracer->StartSpan("innermost");
      auto innermost_scope = tracer->WithActiveSpan(innermost_child);
      innermost_child->End();
    }
    inner_child->End();
  }
}
BENCHMARK(BM_SpanCreationWitContextPropagation);
}  // namespace
BENCHMARK_MAIN();
