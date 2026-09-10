/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

#include <benchmark/benchmark.h>

static void global_average_pooling_q8(benchmark::State& state) {
  const size_t batchSize = state.range(0);
  const size_t inputHeight = state.range(1);
  const size_t inputWidth = state.range(2);
  const size_t channels = state.range(3);

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  const size_t inputPixelStride = channels;
  const size_t outputPixelStride = channels;

  std::vector<uint8_t> input(
      batchSize * inputHeight * inputWidth * inputPixelStride);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> output(batchSize * outputPixelStride);

  pytorch_qnnp_status status = pytorch_qnnp_initialize();
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to initialize QNNPACK");
  }

  pytorch_qnnp_operator_t globalPoolingOperator = nullptr;
  status = pytorch_qnnp_create_global_average_pooling_nwc_q8(
      channels,
      127 /* input zero point */,
      0.75f /* input scale */,
      127 /* output zero point */,
      1.25f /* output scale */,
      0,
      255,
      0 /* flags */,
      &globalPoolingOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to create Global Average Pooling operator");
  }

  status = pytorch_qnnp_setup_global_average_pooling_nwc_q8(
      globalPoolingOperator,
      batchSize,
      inputHeight * inputWidth,
      input.data(),
      inputPixelStride,
      output.data(),
      outputPixelStride);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup Global Average Pooling operator");
  }

  for (auto _ : state) {
    pytorch_qnnp_run_operator(globalPoolingOperator, nullptr /* thread pool */);
  }

  status = pytorch_qnnp_delete_operator(globalPoolingOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete Global Average Pooling operator");
  }
  globalPoolingOperator = nullptr;

  state.SetBytesProcessed(
      uint64_t(state.iterations()) * batchSize *
      (inputHeight * inputWidth + 1) * channels * sizeof(uint8_t));
}

static void ImageNetArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "C"});

  /*       N  IH  IW    C */
  b->Args({1, 7, 7, 1000});
  b->Args({1, 13, 13, 1000});
}

BENCHMARK(global_average_pooling_q8)->Apply(ImageNetArguments);

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
