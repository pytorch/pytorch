/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

#include <benchmark/benchmark.h>

static void add_nc_q8(benchmark::State& state) {
  const size_t batchSize = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  std::vector<uint8_t> a(batchSize * channels);
  std::vector<uint8_t> b(batchSize * channels);
  std::vector<uint8_t> y(batchSize * channels);
  std::generate(a.begin(), a.end(), std::ref(u8rng));
  std::generate(b.begin(), b.end(), std::ref(u8rng));

  pytorch_qnnp_status status = pytorch_qnnp_initialize();
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to initialize QNNPACK");
  }

  pytorch_qnnp_operator_t addOperator = nullptr;
  status = pytorch_qnnp_create_add_nc_q8(
      channels,
      127 /* a:zero point */,
      1.0f /* a:scale */,
      127 /* b:zero point */,
      1.0f /* b:scale */,
      127 /* y:zero point */,
      1.0f /* y:scale */,
      1 /* y:min */,
      254 /* y:max */,
      0 /* flags */,
      &addOperator);
  if (status != pytorch_qnnp_status_success || addOperator == nullptr) {
    state.SkipWithError("failed to create Q8 Add operator");
  }

  status = pytorch_qnnp_setup_add_nc_q8(
      addOperator,
      batchSize,
      a.data(),
      channels /* a:stride */,
      b.data(),
      channels /* b:stride */,
      y.data(),
      channels /* y:stride */);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup Q8 Add operator");
  }

  for (auto _ : state) {
    status = pytorch_qnnp_run_operator(addOperator, nullptr /* thread pool */);
    if (status != pytorch_qnnp_status_success) {
      state.SkipWithError("failed to run Q8 Add operator");
    }
  }

  const size_t itemsPerIteration = batchSize * channels;
  state.SetItemsProcessed(
      int64_t(state.iterations()) * int64_t(itemsPerIteration));

  const size_t bytesPerIteration = 3 * itemsPerIteration * sizeof(uint8_t);
  state.SetBytesProcessed(
      int64_t(state.iterations()) * int64_t(bytesPerIteration));

  status = pytorch_qnnp_delete_operator(addOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete Q8 Add operator");
  }
}

static void add_nc_q8_inplace(benchmark::State& state) {
  const size_t batchSize = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  std::vector<uint8_t> a(batchSize * channels);
  std::vector<uint8_t> y(batchSize * channels);
  std::generate(a.begin(), a.end(), std::ref(u8rng));

  pytorch_qnnp_status status = pytorch_qnnp_initialize();
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to initialize QNNPACK");
  }

  pytorch_qnnp_operator_t addOperator = nullptr;
  status = pytorch_qnnp_create_add_nc_q8(
      channels,
      127 /* a:zero point */,
      1.0f /* a:scale */,
      127 /* b:zero point */,
      1.0f /* b:scale */,
      127 /* y:zero point */,
      1.0f /* y:scale */,
      1 /* y:min */,
      254 /* y:max */,
      0 /* flags */,
      &addOperator);
  if (status != pytorch_qnnp_status_success || addOperator == nullptr) {
    state.SkipWithError("failed to create Q8 Add operator");
  }

  status = pytorch_qnnp_setup_add_nc_q8(
      addOperator,
      batchSize,
      a.data(),
      channels /* a:stride */,
      y.data(),
      channels /* b:stride */,
      y.data(),
      channels /* y:stride */);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup Q8 Add operator");
  }

  for (auto _ : state) {
    status = pytorch_qnnp_run_operator(addOperator, nullptr /* thread pool */);
    if (status != pytorch_qnnp_status_success) {
      state.SkipWithError("failed to run Q8 Add operator");
    }
  }

  const size_t itemsPerIteration = batchSize * channels;
  state.SetItemsProcessed(
      int64_t(state.iterations()) * int64_t(itemsPerIteration));

  const size_t bytesPerIteration = 3 * itemsPerIteration * sizeof(uint8_t);
  state.SetBytesProcessed(
      int64_t(state.iterations()) * int64_t(bytesPerIteration));

  status = pytorch_qnnp_delete_operator(addOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete Q8 Add operator");
  }
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "C"});

  int32_t c = 16;
  for (int32_t n = 224; n >= 7; n /= 2) {
    b->Args({n * n, c});
    c *= 2;
  }
}

BENCHMARK(add_nc_q8)->Apply(CharacteristicArguments);
BENCHMARK(add_nc_q8_inplace)->Apply(CharacteristicArguments);

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
