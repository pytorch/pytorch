/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "benchmark/benchmark.h"

#include <c10/util/Logging.h>

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

NOINLINE int call(int id) {
  C10_LOG_API_USAGE_ONCE("bla");
  return id%2;
}

NOINLINE int call_no_logging(int id) {
  return id%2;
}

static void BM_APILogging(benchmark::State& state) {
  int id = 0;
  while (state.KeepRunning()) {
    for (int i = 0; i < 1000; ++i) {
      id += 1 + call(id);
    }
  }
  benchmark::DoNotOptimize(id);
}
BENCHMARK(BM_APILogging);

static void BM_NoAPILogging(benchmark::State& state) {
  int id = 0;
  while (state.KeepRunning()) {
    for (int i = 0; i < 1000; ++i) {
      id += 1 + call_no_logging(id);
    }
  }
  benchmark::DoNotOptimize(id);
}
BENCHMARK(BM_NoAPILogging);

BENCHMARK_MAIN();
