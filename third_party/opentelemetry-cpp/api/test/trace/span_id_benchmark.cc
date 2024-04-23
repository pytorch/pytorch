// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/span_id.h"

#include <benchmark/benchmark.h>
#include <cstdint>

namespace
{
using opentelemetry::trace::SpanId;
constexpr uint8_t bytes[] = {1, 2, 3, 4, 5, 6, 7, 8};

void BM_SpanIdDefaultConstructor(benchmark::State &state)
{
  while (state.KeepRunning())
  {
    benchmark::DoNotOptimize(SpanId());
  }
}
BENCHMARK(BM_SpanIdDefaultConstructor);

void BM_SpanIdConstructor(benchmark::State &state)
{
  while (state.KeepRunning())
  {
    benchmark::DoNotOptimize(SpanId(bytes));
  }
}
BENCHMARK(BM_SpanIdConstructor);

void BM_SpanIdToLowerBase16(benchmark::State &state)
{
  SpanId id(bytes);
  char buf[SpanId::kSize * 2];
  while (state.KeepRunning())
  {
    id.ToLowerBase16(buf);
    benchmark::DoNotOptimize(buf);
  }
}
BENCHMARK(BM_SpanIdToLowerBase16);

void BM_SpanIdIsValid(benchmark::State &state)
{
  SpanId id(bytes);
  while (state.KeepRunning())
  {
    benchmark::DoNotOptimize(id.IsValid());
  }
}
BENCHMARK(BM_SpanIdIsValid);

}  // namespace
BENCHMARK_MAIN();
