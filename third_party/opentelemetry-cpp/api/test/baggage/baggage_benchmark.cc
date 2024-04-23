// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/baggage/baggage.h"
#include "opentelemetry/nostd/string_view.h"

#include <benchmark/benchmark.h>
#include <cstdint>

using namespace opentelemetry::baggage;
namespace nostd = opentelemetry::nostd;

namespace
{

const size_t kNumEntries = 10;

std::string header_with_custom_entries(size_t num_entries)
{
  std::string header;
  for (size_t i = 0; i < num_entries; i++)
  {
    std::string key   = "ADecentlyLargekey" + std::to_string(i);
    std::string value = "ADecentlyLargeValue" + std::to_string(i);
    header += key + "=" + value;
    if (i != num_entries - 1)
    {
      header += ",";
    }
  }
  return header;
}

void BM_CreateBaggageFromTenEntries(benchmark::State &state)
{
  std::string header = header_with_custom_entries(kNumEntries);
  while (state.KeepRunning())
  {
    auto baggage = Baggage::FromHeader(header);
  }
}
BENCHMARK(BM_CreateBaggageFromTenEntries);

void BM_ExtractBaggageHavingTenEntries(benchmark::State &state)
{
  auto baggage = Baggage::FromHeader(header_with_custom_entries(kNumEntries));
  while (state.KeepRunning())
  {
    baggage->GetAllEntries(
        [](nostd::string_view /* key */, nostd::string_view /* value */) { return true; });
  }
}
BENCHMARK(BM_ExtractBaggageHavingTenEntries);

void BM_CreateBaggageFrom180Entries(benchmark::State &state)
{
  std::string header = header_with_custom_entries(Baggage::kMaxKeyValuePairs);
  while (state.KeepRunning())
  {
    auto baggage = Baggage::FromHeader(header);
  }
}
BENCHMARK(BM_CreateBaggageFrom180Entries);

void BM_ExtractBaggageWith180Entries(benchmark::State &state)
{
  auto baggage = Baggage::FromHeader(header_with_custom_entries(Baggage::kMaxKeyValuePairs));
  while (state.KeepRunning())
  {
    baggage->GetAllEntries(
        [](nostd::string_view /* key */, nostd::string_view /* value */) { return true; });
  }
}
BENCHMARK(BM_ExtractBaggageWith180Entries);

void BM_SetValueBaggageWithTenEntries(benchmark::State &state)
{
  auto baggage = Baggage::FromHeader(
      header_with_custom_entries(kNumEntries - 1));  // 9 entries, and add one new
  while (state.KeepRunning())
  {
    auto new_baggage = baggage->Set("new_key", "new_value");
  }
}
BENCHMARK(BM_SetValueBaggageWithTenEntries);

void BM_SetValueBaggageWith180Entries(benchmark::State &state)
{
  auto baggage = Baggage::FromHeader(header_with_custom_entries(
      Baggage::kMaxKeyValuePairs - 1));  // keep 179 entries, and add one new
  while (state.KeepRunning())
  {
    auto new_baggage = baggage->Set("new_key", "new_value");
  }
}
BENCHMARK(BM_SetValueBaggageWith180Entries);

void BM_BaggageToHeaderTenEntries(benchmark::State &state)
{
  auto baggage = Baggage::FromHeader(header_with_custom_entries(kNumEntries));
  while (state.KeepRunning())
  {
    auto new_baggage = baggage->ToHeader();
  }
}
BENCHMARK(BM_BaggageToHeaderTenEntries);

void BM_BaggageToHeader180Entries(benchmark::State &state)
{
  auto baggage = Baggage::FromHeader(header_with_custom_entries(Baggage::kMaxKeyValuePairs));
  while (state.KeepRunning())
  {
    auto new_baggage = baggage->ToHeader();
  }
}
BENCHMARK(BM_BaggageToHeader180Entries);
}  // namespace

BENCHMARK_MAIN();
