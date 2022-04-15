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

static void max_pooling_u8(benchmark::State& state, const char* net) {
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
  status = pytorch_qnnp_create_max_pooling2d_nhwc_u8(
      paddingSize,
      paddingSize,
      poolingSize,
      poolingSize,
      stride,
      stride,
      1 /* dilation height */,
      1 /* dilation width */,
      channels,
      0,
      255,
      0 /* flags */,
      &poolingOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to create Max Pooling operator");
  }

  status = pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
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
    state.SkipWithError("failed to setup Max Pooling operator");
  }

  for (auto _ : state) {
    status =
        pytorch_qnnp_run_operator(poolingOperator, nullptr /* thread pool */);
    if (status != pytorch_qnnp_status_success) {
      state.SkipWithError("failed to run Max Pooling operator");
    }
  }

  status = pytorch_qnnp_delete_operator(poolingOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete Max Pooling operator");
  }
  poolingOperator = nullptr;

  state.SetBytesProcessed(
      uint64_t(state.iterations()) * batchSize *
      (inputHeight * inputWidth + outputHeight * outputWidth) * channels *
      sizeof(uint8_t));
}

/* ShuffleNet v1/v2 */
static void ShuffleNet(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H   W    K  P  S   C */
  b->Args({1, 112, 112, 3, 1, 2, 24});
}

/* SqueezeNet 1.0 */
static void SqueezeNetV10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*********** MaxPool 1 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1, 111, 111, 3, 0, 2, 96});
  /*********** MaxPool 4 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1, 27, 27, 3, 0, 2, 256});
  /*********** MaxPool 8 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1, 13, 13, 3, 0, 2, 512});
}

/* SqueezeNet 1.1 */
static void SqueezeNetV11(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*********** MaxPool 1 ***********/
  /*       N   H    W   K  P  S   C */
  b->Args({1, 111, 111, 3, 0, 2, 64});
  /*********** MaxPool 3 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1, 55, 55, 3, 0, 2, 128});
  /*********** MaxPool 5 ************/
  /*       N   H    W   K  P  S   C */
  b->Args({1, 13, 13, 3, 0, 2, 256});
}

static void VGG(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "K", "P", "S", "C"});

  /*       N   H    W   K  P  S   C */
  b->Args({1, 224, 224, 2, 1, 2, 64});
  b->Args({1, 112, 112, 2, 1, 2, 128});
  b->Args({1, 56, 56, 2, 1, 2, 256});
  b->Args({1, 28, 28, 2, 1, 2, 512});
  b->Args({1, 14, 14, 2, 1, 2, 512});
}

BENCHMARK_CAPTURE(max_pooling_u8, shufflenet, "ShuffleNet v1/v2")
    ->Apply(ShuffleNet);
BENCHMARK_CAPTURE(max_pooling_u8, squeezenet_v10, "SqueezeNet v1.0")
    ->Apply(SqueezeNetV10);
BENCHMARK_CAPTURE(max_pooling_u8, squeezenet_v11, "SqueezeNet v1.1")
    ->Apply(SqueezeNetV11);
BENCHMARK_CAPTURE(max_pooling_u8, vgg, "VGG")->Apply(VGG);

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
