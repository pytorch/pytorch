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

static void average_pooling_q8(benchmark::State& state, const char* net) {
  const size_t batchSize = state.range(0);
  const size_t inputHeight = state.range(1);
  const size_t inputWidth = state.range(2);
  const size_t poolingSize = state.range(3);
  const size_t paddingSize = state.range(4);
  const size_t stride = state.range(5);
  const size_t channels = state.range(6);

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  const size_t inputPixelStride = channels;
  const size_t outputPixelStride = channels;
  const size_t outputHeight =
      (2 * paddingSize + inputHeight - poolingSize) / stride + 1;
  const size_t outputWidth =
      (2 * paddingSize + inputWidth - poolingSize) / stride + 1;

  std::vector<uint8_t> input(
      batchSize * inputHeight * inputWidth * inputPixelStride);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> output(
      batchSize * outputHeight * outputWidth * outputPixelStride);
  std::fill(output.begin(), output.end(), 0xA5);

  pytorch_qnnp_status status = pytorch_qnnp_initialize();
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to initialize QNNPACK");
  }

  pytorch_qnnp_operator_t poolingOperator = nullptr;
  status = pytorch_qnnp_create_average_pooling2d_nhwc_q8(
      paddingSize,
      paddingSize,
      poolingSize,
      poolingSize,
      stride,
      stride,
      channels,
      127 /* input zero point */,
      0.75f /* input scale */,
      127 /* output zero point */,
      1.25f /* output scale */,
      0,
      255,
      0 /* flags */,
      &poolingOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to create Average Pooling operator");
  }

  status = pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
      poolingOperator,
      batchSize,
      inputHeight,
      inputWidth,
      input.data(),
      inputPixelStride,
      output.data(),
      outputPixelStride,
      nullptr /* thread pool */);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup Average Pooling operator");
  }

  for (auto _ : state) {
    status =
        pytorch_qnnp_run_operator(poolingOperator, nullptr /* thread pool */);
    if (status != pytorch_qnnp_status_success) {
      state.SkipWithError("failed to run Average Pooling operator");
    }
  }

  status = pytorch_qnnp_delete_operator(poolingOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete Average Pooling operator");
  }
  poolingOperator = nullptr;

  state.SetBytesProcessed(
      uint64_t(state.iterations()) * batchSize *
      (inputHeight * inputWidth + outputHeight * outputWidth) * channels *
      sizeof(uint8_t));
}

/* ShuffleNet v1 with 1 group */
static void ShuffleNetV1G1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S   C */
  b->Args({1, 56, 56, 3, 1, 2, 24});
  b->Args({1, 28, 28, 3, 1, 2, 144});
  b->Args({1, 14, 14, 3, 1, 2, 288});
  b->Args({1, 7, 7, 3, 1, 2, 576});
}

/* ShuffleNet v1 with 2 groups */
static void ShuffleNetV1G2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S   C */
  b->Args({1, 56, 56, 3, 1, 2, 24});
  b->Args({1, 28, 28, 3, 1, 2, 200});
  b->Args({1, 14, 14, 3, 1, 2, 400});
  b->Args({1, 7, 7, 3, 1, 2, 800});
}

/* ShuffleNet v1 with 3 groups */
static void ShuffleNetV1G3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S   C */
  b->Args({1, 56, 56, 3, 1, 2, 24});
  b->Args({1, 28, 28, 3, 1, 2, 240});
  b->Args({1, 14, 14, 3, 1, 2, 480});
  b->Args({1, 7, 7, 3, 1, 2, 960});
}

/* ShuffleNet v1 with 4 groups */
static void ShuffleNetV1G4(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S    C */
  b->Args({1, 56, 56, 3, 1, 2, 24});
  b->Args({1, 28, 28, 3, 1, 2, 272});
  b->Args({1, 14, 14, 3, 1, 2, 576});
  b->Args({1, 7, 7, 3, 1, 2, 1088});
}

/* ShuffleNet v1 with 8 groups */
static void ShuffleNetV1G8(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W  K  P  S    C */
  b->Args({1, 56, 56, 3, 1, 2, 24});
  b->Args({1, 28, 28, 3, 1, 2, 384});
  b->Args({1, 14, 14, 3, 1, 2, 768});
  b->Args({1, 7, 7, 3, 1, 2, 1536});
}

BENCHMARK_CAPTURE(
    average_pooling_q8,
    shufflenet_v1_g1,
    "ShuffleNet v1 (1 group)")
    ->Apply(ShuffleNetV1G1);
BENCHMARK_CAPTURE(
    average_pooling_q8,
    shufflenet_v1_g2,
    "ShuffleNet v1 (2 groups)")
    ->Apply(ShuffleNetV1G2);
BENCHMARK_CAPTURE(
    average_pooling_q8,
    shufflenet_v1_g3,
    "ShuffleNet v1 (3 groups)")
    ->Apply(ShuffleNetV1G3);
BENCHMARK_CAPTURE(
    average_pooling_q8,
    shufflenet_v1_g4,
    "ShuffleNet v1 (4 groups)")
    ->Apply(ShuffleNetV1G4);
BENCHMARK_CAPTURE(
    average_pooling_q8,
    shufflenet_v1_g8,
    "ShuffleNet v1 (8 groups)")
    ->Apply(ShuffleNetV1G8);

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
