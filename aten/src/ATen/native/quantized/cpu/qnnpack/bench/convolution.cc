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

static void convolution_q8(benchmark::State& state, const char* net, bool per_channel=false) {
  const size_t batchSize = state.range(0);
  const size_t inputHeight = state.range(1);
  const size_t inputWidth = state.range(2);
  const size_t kernelHeight = state.range(3);
  const size_t kernelWidth = state.range(4);
  const size_t subsampling = state.range(5);
  const size_t dilation = state.range(6);
  const size_t groups = state.range(7);
  const size_t groupInputChannels = state.range(8);
  const size_t groupOutputChannels = state.range(9);

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto s32rng =
      std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  const size_t outputPixelStride = groups * groupOutputChannels;
  const size_t inputPixelStride = groups * groupInputChannels;
  const size_t effectiveKernelHeight = (kernelHeight - 1) * dilation + 1;
  const size_t effectiveKernelWidth = (kernelWidth - 1) * dilation + 1;
  const size_t paddingWidth = effectiveKernelWidth / 2;
  const size_t paddingHeight = effectiveKernelHeight / 2;
  const size_t outputHeight =
      (inputHeight + paddingHeight * 2 - effectiveKernelHeight) / subsampling +
      1;
  const size_t outputWidth =
      (inputWidth + paddingWidth * 2 - effectiveKernelWidth) / subsampling + 1;

  std::vector<uint8_t> input(
      batchSize * inputHeight * inputWidth * inputPixelStride);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> kernel(
      groups * groupOutputChannels * kernelHeight * kernelWidth *
      groupInputChannels);
  std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
  std::vector<int32_t> bias(groups * groupOutputChannels);
  std::generate(bias.begin(), bias.end(), std::ref(s32rng));
  std::vector<uint8_t> output(
      batchSize * outputHeight * outputWidth * outputPixelStride);

  pytorch_qnnp_status status = pytorch_qnnp_initialize();
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to initialize QNNPACK");
  }

  pytorch_qnnp_operator_t convolutionObject = nullptr;
  size_t num_zero_points_padded =
    ((groups * groupOutputChannels + 7) / 8) * 8;
  std::vector<uint8_t> kernel_zero_points(num_zero_points_padded, 127);
  std::vector<float> requantization_scale(
      num_zero_points_padded, 0.5 * 0.5 / 0.5);
  status = pytorch_qnnp_create_convolution2d_nhwc_q8(
      paddingHeight,
      paddingWidth,
      kernelHeight,
      kernelWidth,
      subsampling,
      subsampling,
      dilation,
      dilation,
      groups,
      groupInputChannels,
      groupOutputChannels,
      127,
      kernel_zero_points.data(),
      kernel.data(),
      bias.data(),
      127,
      0,
      255,
      0 /* flags */,
      requantization_scale.data(),
      per_channel,
      &convolutionObject);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to create Convolution operator");
  }

  status = pytorch_qnnp_setup_convolution2d_nhwc_q8(
      convolutionObject,
      batchSize,
      inputHeight,
      inputWidth,
      input.data(),
      inputPixelStride,
      output.data(),
      outputPixelStride,
      nullptr /* thread pool */);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup Convolution operator");
  }

  for (auto _ : state) {
    pytorch_qnnp_run_operator(convolutionObject, nullptr /* thread pool */);
  }

  status = pytorch_qnnp_delete_operator(convolutionObject);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete Convolution operator");
  }
  convolutionObject = nullptr;

  state.SetItemsProcessed(
      uint64_t(state.iterations()) * 2 * batchSize * outputHeight *
      outputWidth * groups * groupInputChannels * groupOutputChannels *
      kernelHeight * kernelWidth);
}

/* ShuffleNet v1 with 1 group */
static void ShuffleNetV1G1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /*************** Stage 2: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 36});
  b->Args({1, 56, 56, 3, 3, 2, 1, 36, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 36, 120});
  /*************** Stage 2: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 144, 36});
  b->Args({1, 28, 28, 3, 3, 2, 1, 36, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 36, 144});
  /*************** Stage 3: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 144, 72});
  b->Args({1, 28, 28, 3, 3, 2, 1, 72, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 72, 144});
  /*************** Stage 3: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 288, 72});
  b->Args({1, 14, 14, 3, 3, 2, 1, 72, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 72, 288});
  /*************** Stage 4: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 288, 144});
  b->Args({1, 14, 14, 3, 3, 2, 1, 144, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 144, 288});
  /*************** Stage 4: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 576, 144});
  b->Args({1, 7, 7, 3, 3, 2, 1, 144, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 144, 576});
}

/* ShuffleNet v1 with 2 groups */
static void ShuffleNetV1G2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /*************** Stage 2: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 50});
  b->Args({1, 56, 56, 3, 3, 2, 1, 50, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 2, 25, 88});
  /*************** Stage 2: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 2, 100, 25});
  b->Args({1, 28, 28, 3, 3, 2, 1, 50, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 2, 25, 100});
  /*************** Stage 3: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 2, 100, 50});
  b->Args({1, 28, 28, 3, 3, 2, 1, 100, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 2, 50, 100});
  /*************** Stage 3: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 2, 200, 50});
  b->Args({1, 14, 14, 3, 3, 2, 1, 100, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 2, 50, 200});
  /*************** Stage 4: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 2, 200, 100});
  b->Args({1, 14, 14, 3, 3, 2, 1, 200, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 2, 100, 200});
  /*************** Stage 4: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 2, 400, 100});
  b->Args({1, 7, 7, 3, 3, 2, 1, 200, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 2, 100, 400});
}

/* ShuffleNet v1 with 3 groups */
static void ShuffleNetV1G3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /*************** Stage 2: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 60});
  b->Args({1, 56, 56, 3, 3, 2, 1, 60, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 3, 20, 72});
  /*************** Stage 2: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 3, 80, 20});
  b->Args({1, 28, 28, 3, 3, 2, 1, 60, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 3, 20, 80});
  /*************** Stage 3: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 3, 80, 40});
  b->Args({1, 28, 28, 3, 3, 2, 1, 120, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 3, 40, 80});
  /*************** Stage 3: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 3, 160, 40});
  b->Args({1, 14, 14, 3, 3, 2, 1, 120, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 3, 40, 160});
  /*************** Stage 4: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 3, 160, 80});
  b->Args({1, 14, 14, 3, 3, 2, 1, 240, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 3, 80, 160});
  /*************** Stage 4: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 3, 320, 80});
  b->Args({1, 7, 7, 3, 3, 2, 1, 240, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 3, 80, 320});
}

/* ShuffleNet v1 with 4 groups */
static void ShuffleNetV1G4(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /*************** Stage 2: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 68});
  b->Args({1, 56, 56, 3, 3, 2, 1, 68, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 4, 17, 62});
  /*************** Stage 2: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 4, 68, 17});
  b->Args({1, 28, 28, 3, 3, 2, 1, 68, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 4, 17, 68});
  /*************** Stage 3: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 4, 68, 34});
  b->Args({1, 28, 28, 3, 3, 2, 1, 136, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 4, 34, 68});
  /*************** Stage 3: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 4, 136, 34});
  b->Args({1, 14, 14, 3, 3, 2, 1, 136, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 4, 34, 136});
  /*************** Stage 4: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 4, 136, 68});
  b->Args({1, 14, 14, 3, 3, 2, 1, 272, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 4, 68, 136});
  /*************** Stage 4: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 4, 272, 68});
  b->Args({1, 7, 7, 3, 3, 2, 1, 272, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 4, 68, 272});
}

/* ShuffleNet v1 with 8 groups */
static void ShuffleNetV1G8(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /*************** Stage 2: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 96});
  b->Args({1, 56, 56, 3, 3, 2, 1, 96, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 8, 12, 45});
  /*************** Stage 2: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 8, 48, 12});
  b->Args({1, 28, 28, 3, 3, 2, 1, 96, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 8, 12, 48});
  /*************** Stage 3: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 8, 48, 24});
  b->Args({1, 28, 28, 3, 3, 2, 1, 192, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 8, 24, 48});
  /*************** Stage 3: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 8, 96, 24});
  b->Args({1, 14, 14, 3, 3, 2, 1, 192, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 8, 24, 96});
  /*************** Stage 4: stride-2 unit **************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 8, 96, 48});
  b->Args({1, 14, 14, 3, 3, 2, 1, 384, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 8, 48, 96});
  /*************** Stage 4: stride-1 units *************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 8, 192, 48});
  b->Args({1, 7, 7, 3, 3, 2, 1, 384, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 8, 48, 192});
}

/* ShuffleNet v2 (0.5X scale) */
static void ShuffleNetV2X05(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /********************** Stage 2 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 3, 3, 2, 1, 24, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 24, 24});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 24});
  b->Args({1, 28, 28, 3, 3, 1, 1, 24, 1, 1});
  /********************** Stage 3 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 3, 3, 2, 1, 48, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 48, 48});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 48, 48});
  b->Args({1, 14, 14, 3, 3, 1, 1, 48, 1, 1});
  /********************** Stage 4 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 3, 3, 2, 1, 96, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 96, 96});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 96, 96});
  b->Args({1, 7, 7, 3, 3, 1, 1, 96, 1, 1});
  /*********************** Conv 5 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 192, 1024});
}

/* ShuffleNet v2 (1.0X scale) */
static void ShuffleNetV2X10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /********************** Stage 2 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 3, 3, 2, 1, 24, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 24, 58});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 58});
  b->Args({1, 56, 56, 3, 3, 2, 1, 58, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 58, 58});
  b->Args({1, 28, 28, 3, 3, 1, 1, 58, 1, 1});
  /********************** Stage 3 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 3, 3, 2, 1, 116, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 116, 116});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 116, 116});
  b->Args({1, 14, 14, 3, 3, 1, 1, 116, 1, 1});
  /********************** Stage 4 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 3, 3, 2, 1, 232, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 232, 232});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 232, 232});
  b->Args({1, 7, 7, 3, 3, 1, 1, 232, 1, 1});
  /*********************** Conv 5 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 464, 1024});
}

/* ShuffleNet v2 (1.5X scale) */
static void ShuffleNetV2X15(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /********************** Stage 2 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 3, 3, 2, 1, 24, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 24, 88});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 88});
  b->Args({1, 56, 56, 3, 3, 2, 1, 88, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 88, 88});
  b->Args({1, 28, 28, 3, 3, 1, 1, 88, 1, 1});
  /********************** Stage 3 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 3, 3, 2, 1, 176, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 176, 176});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 176, 176});
  b->Args({1, 14, 14, 3, 3, 1, 1, 176, 1, 1});
  /********************** Stage 4 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 3, 3, 2, 1, 352, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 352, 352});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 352, 352});
  b->Args({1, 7, 7, 3, 3, 1, 1, 352, 1, 1});
  /*********************** Conv 5 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 704, 1024});
}

/* ShuffleNet v2 (2.0X scale) */
static void ShuffleNetV2X20(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Conv 1 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 24});
  /********************** Stage 2 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 56, 56, 3, 3, 2, 1, 24, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 24, 122});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 122});
  b->Args({1, 56, 56, 3, 3, 2, 1, 122, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 122, 122});
  b->Args({1, 28, 28, 3, 3, 1, 1, 122, 1, 1});
  /********************** Stage 3 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 28, 28, 3, 3, 2, 1, 244, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 244, 244});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 244, 244});
  b->Args({1, 14, 14, 3, 3, 1, 1, 244, 1, 1});
  /********************** Stage 4 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 14, 14, 3, 3, 2, 1, 488, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 488, 488});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 488, 488});
  b->Args({1, 7, 7, 3, 3, 1, 1, 488, 1, 1});
  /*********************** Conv 5 **********************/
  /*       N   H    W   KH  KW  S  D   G   GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 976, 2048});
}

static void MobileNetV1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 32});
  b->Args({1, 112, 112, 3, 3, 1, 1, 32, 1, 1});
  b->Args({1, 112, 112, 1, 1, 1, 1, 1, 32, 64});
  b->Args({1, 112, 112, 3, 3, 2, 1, 64, 1, 1});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 64, 128});
  b->Args({1, 56, 56, 3, 3, 1, 1, 128, 1, 1});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 128, 128});
  b->Args({1, 56, 56, 3, 3, 2, 1, 128, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 128, 256});
  b->Args({1, 28, 28, 3, 3, 1, 1, 256, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 256, 256});
  b->Args({1, 28, 28, 3, 3, 2, 1, 256, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 256, 512});
  b->Args({1, 14, 14, 3, 3, 1, 1, 512, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 512, 512});
  b->Args({1, 14, 14, 3, 3, 2, 1, 512, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 512, 1024});
  b->Args({1, 7, 7, 3, 3, 1, 1, 1024, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 1024, 1024});
}

static void MobileNetV2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 32});

  /******************** Bottleneck 1 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1, 112, 112, 3, 3, 1, 1, 32, 1, 1});
  b->Args({1, 112, 112, 1, 1, 1, 1, 1, 32, 16});

  /******************** Bottleneck 2 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1, 112, 112, 1, 1, 1, 1, 1, 16, 96});
  b->Args({1, 112, 112, 3, 3, 2, 1, 96, 1, 1});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 96, 24});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 24, 144});
  b->Args({1, 56, 56, 3, 3, 1, 1, 144, 1, 1});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 144, 24});

  /******************** Bottleneck 3 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  // b->Args({1,  56,  56,  1,  1, 1, 1,   1,   24,  144});
  b->Args({1, 56, 56, 3, 3, 2, 1, 144, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 144, 32});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 32, 192});
  b->Args({1, 28, 28, 3, 3, 1, 1, 192, 1, 1});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 192, 32});
  // b->Args({1,  28,  28,  1,  1, 1, 1,   1,   32,  192});
  // b->Args({1,  28,  28,  3,  3, 1, 1, 192,    1,    1});
  // b->Args({1,  28,  28,  1,  1, 1, 1,   1,  192,   32});

  /******************** Bottleneck 4 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  // b->Args({1,  28,  28,  1,  1, 1, 1,   1,   32,  192});
  b->Args({1, 28, 28, 3, 3, 2, 1, 192, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 192, 64});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 64, 384});
  b->Args({1, 14, 14, 3, 3, 1, 1, 384, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 384, 64});
  // b->Args({1,  14,  14,  1,  1, 1, 1,   1,   64,  384});
  // b->Args({1,  14,  14,  3,  3, 1, 1, 384,    1,    1});
  // b->Args({1,  14,  14,  1,  1, 1, 1,   1,  384,   64});
  // b->Args({1,  14,  14,  1,  1, 1, 1,   1,   64,  384});
  // b->Args({1,  14,  14,  3,  3, 1, 1, 384,    1,    1});
  // b->Args({1,  14,  14,  1,  1, 1, 1,   1,  384,   64});

  /******************** Bottleneck 5 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  // b->Args({1,  14,  14,  1,  1, 1, 1,   1,   64,  384});
  // b->Args({1,  14,  14,  3,  3, 1, 1, 384,    1,    1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 384, 96});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 96, 576});
  b->Args({1, 14, 14, 3, 3, 1, 1, 576, 1, 1});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 576, 96});
  // b->Args({1,  14,  14,  1,  1, 1, 1,   1,   96,  576});
  // b->Args({1,  14,  14,  3,  3, 1, 1, 576,    1,    1});
  // b->Args({1,  14,  14,  1,  1, 1, 1,   1,  576,   96});

  /******************** Bottleneck 6 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  // b->Args({1,  14,  14,  1,  1, 1, 1,   1,   96,  576});
  b->Args({1, 14, 14, 3, 3, 2, 1, 576, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 576, 160});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 160, 960});
  b->Args({1, 7, 7, 3, 3, 1, 1, 960, 1, 1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 960, 160});
  // b->Args({1,   7,   7,  1,  1, 1, 1,   1,  160,  960});
  // b->Args({1,   7,   7,  3,  3, 1, 1, 960,    1,    1});
  // b->Args({1,   7,   7,  1,  1, 1, 1,   1,  960,  160});

  /******************** Bottleneck 7 *******************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  // b->Args({1,   7,   7,  1,  1, 1, 1,   1,  160,  960});
  // b->Args({1,   7,   7,  3,  3, 1, 1, 960,    1,    1});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 960, 320});

  /**************** Pre-pooling Conv2D *****************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 320, 1280});
  /**************** Post-pooling Conv2D ****************/
  /*       N   H    W   KH  KW  S  D    G  GCin  GCout */
  b->Args({1, 1, 1, 1, 1, 1, 1, 1, 1280, 1000});
}

/* SqueezeNet 1.0 */
static void SqueezeNetV10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** Conv 1 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224, 7, 7, 2, 1, 1, 3, 96});
  /********************** Fire 2 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 55, 55, 1, 1, 1, 1, 1, 96, 16});
  b->Args({1, 55, 55, 1, 1, 1, 1, 1, 16, 64});
  b->Args({1, 55, 55, 3, 3, 1, 1, 1, 16, 64});
  /********************** Fire 3 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 56, 55, 1, 1, 1, 1, 1, 128, 16});
  /*b->Args({1,  55,  55,  1,  1, 1, 1, 1,   16,   64});*/
  /*b->Args({1,  55,  55,  3,  3, 1, 1, 1,   16,   64});*/
  /********************** Fire 4 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 55, 55, 1, 1, 1, 1, 1, 128, 32});
  b->Args({1, 55, 55, 1, 1, 1, 1, 1, 32, 128});
  b->Args({1, 55, 55, 3, 3, 1, 1, 1, 32, 128});
  /********************** Fire 5 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 256, 32});
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 32, 128});
  b->Args({1, 27, 27, 3, 3, 1, 1, 1, 32, 128});
  /********************** Fire 6 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 256, 48});
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 48, 192});
  b->Args({1, 27, 27, 3, 3, 1, 1, 1, 48, 192});
  /********************** Fire 7 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 384, 48});
  /*b->Args({1,  27,  27,  1,  1, 1, 1, 1,   48,  192});*/
  /*b->Args({1,  27,  27,  3,  3, 1, 1, 1,   48,  192});*/
  /********************** Fire 8 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 384, 64});
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 64, 256});
  b->Args({1, 27, 27, 3, 3, 1, 1, 1, 64, 256});
  /********************** Fire 9 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 512, 64});
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 64, 256});
  b->Args({1, 13, 13, 3, 3, 1, 1, 1, 64, 256});
  /********************* Conv 10 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 512, 1000});
}

/* SqueezeNet 1.1 */
static void SqueezeNetV11(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** Conv 1 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 2, 1, 1, 3, 64});
  /********************** Fire 2 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 55, 55, 1, 1, 1, 1, 1, 64, 16});
  b->Args({1, 55, 55, 1, 1, 1, 1, 1, 16, 64});
  b->Args({1, 55, 55, 3, 3, 1, 1, 1, 16, 64});
  /********************** Fire 3 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 55, 55, 1, 1, 1, 1, 1, 128, 16});
  /*b->Args({1,  55,  55,  1,  1, 1, 1, 1,   16,   64});*/
  /*b->Args({1,  55,  55,  3,  3, 1, 1, 1,   16,   64});*/
  /********************** Fire 4 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 128, 32});
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 32, 128});
  b->Args({1, 27, 27, 3, 3, 1, 1, 1, 32, 128});
  /********************** Fire 5 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 27, 27, 1, 1, 1, 1, 1, 256, 32});
  /*b->Args({1,  27,  27,  1,  1, 1, 1, 1,   32,  128});*/
  /*b->Args({1,  27,  27,  3,  3, 1, 1, 1,   32,  128});*/
  /********************** Fire 6 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 256, 48});
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 48, 192});
  b->Args({1, 13, 13, 3, 3, 1, 1, 1, 48, 192});
  /********************** Fire 7 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 384, 48});
  /*b->Args({1,  13,  13,  1,  1, 1, 1, 1,   48,  192});*/
  /*b->Args({1,  13,  13,  3,  3, 1, 1, 1,   48,  192});*/
  /********************** Fire 8 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 384, 64});
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 64, 256});
  b->Args({1, 13, 13, 3, 3, 1, 1, 1, 64, 256});
  /********************** Fire 9 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 512, 64});
  /*b->Args({1,  13,  13,  1,  1, 1, 1, 1,   64,  256});*/
  /*b->Args({1,  13,  13,  3,  3, 1, 1, 1,   64,  256});*/
  /********************* Conv 10 *********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 13, 13, 1, 1, 1, 1, 1, 512, 1000});
}

static void ResNet18(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************* Conv 1 *********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 224, 224, 7, 7, 2, 1, 1, 3, 64});
  /******************** Conv 2.X ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 56, 56, 3, 3, 1, 1, 1, 64, 64});
  /******************** Conv 3.X ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 56, 56, 3, 3, 2, 1, 1, 64, 128});
  b->Args({1, 28, 28, 3, 3, 1, 1, 1, 128, 128});
  b->Args({1, 56, 56, 1, 1, 2, 1, 1, 64, 128});
  /******************** Conv 4.X ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 28, 28, 3, 3, 2, 1, 1, 128, 256});
  b->Args({1, 14, 14, 3, 3, 1, 1, 1, 256, 256});
  b->Args({1, 28, 28, 1, 1, 2, 1, 1, 128, 256});
  /******************** Conv 5.X ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 14, 14, 3, 3, 2, 1, 1, 256, 512});
  b->Args({1, 7, 7, 3, 3, 1, 1, 1, 512, 512});
  b->Args({1, 14, 14, 1, 1, 2, 1, 1, 256, 512});
}

static void ResNet50(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************* Conv 1 *********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 224, 224, 7, 7, 2, 1, 1, 3, 64});
  /******************** Conv 2.1 ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 64, 64});
  b->Args({1, 56, 56, 3, 3, 1, 1, 1, 64, 64});
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 64, 256});
  /*b->Args({1,  56,  56,  1,  1, 1, 1, 1,   64,  256});*/
  /******************** Conv 2.X ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 256, 64});
  /*b->Args({1,  56,  56,  3,  3, 1, 1, 1,   64,   64});*/
  /*b->Args({1,  56,  56,  1,  1, 1, 1, 1,   64,  256});*/
  /******************** Conv 3.1 ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 256, 128});
  b->Args({1, 56, 56, 3, 3, 2, 1, 1, 128, 128});
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 128, 512});
  b->Args({1, 56, 56, 1, 1, 2, 1, 1, 256, 512});
  /******************** Conv 3.X ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 512, 128});
  b->Args({1, 28, 28, 3, 3, 1, 1, 1, 128, 128});
  /*b->Args({1,  28,  28,  1,  1, 1, 1, 1,  128,  512});*/
  /******************** Conv 4.1 ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 512, 256});
  b->Args({1, 28, 28, 3, 3, 2, 1, 1, 256, 256});
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 256, 1024});
  b->Args({1, 28, 28, 1, 1, 2, 1, 1, 512, 1024});
  /******************** Conv 4.X ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 1024, 256});
  b->Args({1, 14, 14, 3, 3, 1, 1, 1, 256, 256});
  /*b->Args({1,  14,  14,  1,  1, 1, 1, 1,  256, 1024});*/
  /******************** Conv 5.1 ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 1024, 512});
  b->Args({1, 14, 14, 3, 3, 2, 1, 1, 512, 512});
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 512, 2048});
  b->Args({1, 14, 14, 1, 1, 2, 1, 1, 1024, 2048});
  /******************** Conv 5.X ********************/
  /*       N   H    W   KH  KW  S  D  G GCin  GCout */
  b->Args({1, 7, 7, 1, 1, 1, 1, 1, 2048, 512});
  b->Args({1, 7, 7, 3, 3, 1, 1, 1, 512, 512});
  /*b->Args({1,   7,   7,  1,  1, 1, 1, 1,  512, 2048});*/
}

static void VGG(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************* Conv 1.1 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 1, 1, 1, 3, 64});
  /********************* Conv 1.2 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224, 3, 3, 1, 1, 1, 64, 64});

  /********************* Conv 2.1 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 112, 112, 3, 3, 1, 1, 1, 64, 128});
  /********************* Conv 2.2 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 112, 112, 3, 3, 1, 1, 1, 128, 128});

  /********************* Conv 3.1 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 56, 56, 3, 3, 1, 1, 1, 128, 256});
  /********************* Conv 3.2 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 56, 56, 3, 3, 1, 1, 1, 256, 256});
  /********************* Conv 3.3 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 56, 56, 1, 1, 1, 1, 1, 256, 256});

  /********************* Conv 4.1 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 28, 28, 3, 3, 1, 1, 1, 256, 512});
  /********************* Conv 4.2 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 28, 28, 3, 3, 1, 1, 1, 512, 512});
  /********************* Conv 4.3 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 28, 28, 1, 1, 1, 1, 1, 512, 512});

  /********************* Conv 5.X ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 14, 14, 3, 3, 1, 1, 1, 512, 512});
  /********************* Conv 5.3 ********************/
  /*       N   H    W   KH  KW  S  D  G  GCin  GCout */
  b->Args({1, 14, 14, 1, 1, 1, 1, 1, 512, 512});
}

static void DWConv3x3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** 96 x 96 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 96, 96, 3, 3, 1, 1, 512, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 1, 256, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 1, 128, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 1, 64, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 1, 48, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 1, 32, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 1, 24, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 1, 16, 1, 1});
  /********************** 32 x 32 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 32, 32, 3, 3, 1, 1, 768, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 1, 512, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 1, 256, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 1, 128, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 1, 64, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 1, 48, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 1, 32, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 1, 24, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 1, 16, 1, 1});
  /********************** 17 x 17 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 17, 17, 3, 3, 1, 1, 1024, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 1, 768, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 1, 512, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 1, 384, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 1, 256, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 1, 128, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 1, 64, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 1, 32, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 1, 16, 1, 1});
  /********************** 11 x 11 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 11, 11, 3, 3, 1, 1, 1024, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 768, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 512, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 384, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 256, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 192, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 128, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 64, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 32, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 1, 16, 1, 1});
  /*********************** 7 x 7 **********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 7, 7, 3, 3, 1, 1, 1024, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 1, 768, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 1, 512, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 1, 384, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 1, 256, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 1, 128, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 1, 64, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 1, 32, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 1, 16, 1, 1});
}

static void DWConv3x3d2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** 96 x 96 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 96, 96, 3, 3, 1, 2, 512, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 2, 256, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 2, 128, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 2, 64, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 2, 48, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 2, 32, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 2, 24, 1, 1});
  b->Args({1, 96, 96, 3, 3, 1, 2, 16, 1, 1});
  /********************** 32 x 32 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 32, 32, 3, 3, 1, 2, 768, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 2, 512, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 2, 256, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 2, 128, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 2, 64, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 2, 48, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 2, 32, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 2, 24, 1, 1});
  b->Args({1, 32, 32, 3, 3, 1, 2, 16, 1, 1});
  /********************** 17 x 17 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 17, 17, 3, 3, 1, 2, 1024, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 2, 768, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 2, 512, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 2, 384, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 2, 256, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 2, 128, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 2, 64, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 2, 32, 1, 1});
  b->Args({1, 17, 17, 3, 3, 1, 2, 16, 1, 1});
  /********************** 11 x 11 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 11, 11, 3, 3, 1, 2, 1024, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 768, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 512, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 384, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 256, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 192, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 128, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 64, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 32, 1, 1});
  b->Args({1, 11, 11, 3, 3, 1, 2, 16, 1, 1});
  /*********************** 7 x 7 **********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 7, 7, 3, 3, 1, 2, 1024, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 2, 768, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 2, 512, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 2, 384, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 2, 256, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 2, 128, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 2, 64, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 2, 32, 1, 1});
  b->Args({1, 7, 7, 3, 3, 1, 2, 16, 1, 1});
}

static void DWConv5x5(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "S", "D", "G", "GCin", "GCout"});

  /********************** 96 x 96 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 96, 96, 5, 5, 1, 1, 512, 1, 1});
  b->Args({1, 96, 96, 5, 5, 1, 1, 256, 1, 1});
  b->Args({1, 96, 96, 5, 5, 1, 1, 128, 1, 1});
  b->Args({1, 96, 96, 5, 5, 1, 1, 64, 1, 1});
  b->Args({1, 96, 96, 5, 5, 1, 1, 48, 1, 1});
  b->Args({1, 96, 96, 5, 5, 1, 1, 32, 1, 1});
  b->Args({1, 96, 96, 5, 5, 1, 1, 24, 1, 1});
  b->Args({1, 96, 96, 5, 5, 1, 1, 16, 1, 1});
  /********************** 32 x 32 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 32, 32, 5, 5, 1, 1, 768, 1, 1});
  b->Args({1, 32, 32, 5, 5, 1, 1, 512, 1, 1});
  b->Args({1, 32, 32, 5, 5, 1, 1, 256, 1, 1});
  b->Args({1, 32, 32, 5, 5, 1, 1, 128, 1, 1});
  b->Args({1, 32, 32, 5, 5, 1, 1, 64, 1, 1});
  b->Args({1, 32, 32, 5, 5, 1, 1, 48, 1, 1});
  b->Args({1, 32, 32, 5, 5, 1, 1, 32, 1, 1});
  b->Args({1, 32, 32, 5, 5, 1, 1, 24, 1, 1});
  b->Args({1, 32, 32, 5, 5, 1, 1, 16, 1, 1});
  /********************** 17 x 17 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 17, 17, 5, 5, 1, 1, 1024, 1, 1});
  b->Args({1, 17, 17, 5, 5, 1, 1, 768, 1, 1});
  b->Args({1, 17, 17, 5, 5, 1, 1, 512, 1, 1});
  b->Args({1, 17, 17, 5, 5, 1, 1, 384, 1, 1});
  b->Args({1, 17, 17, 5, 5, 1, 1, 256, 1, 1});
  b->Args({1, 17, 17, 5, 5, 1, 1, 128, 1, 1});
  b->Args({1, 17, 17, 5, 5, 1, 1, 64, 1, 1});
  b->Args({1, 17, 17, 5, 5, 1, 1, 32, 1, 1});
  b->Args({1, 17, 17, 5, 5, 1, 1, 16, 1, 1});
  /********************** 11 x 11 *********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 11, 11, 5, 5, 1, 1, 1024, 1, 1});
  b->Args({1, 11, 11, 5, 5, 1, 1, 768, 1, 1});
  b->Args({1, 11, 11, 5, 5, 1, 1, 512, 1, 1});
  b->Args({1, 11, 11, 5, 5, 1, 1, 384, 1, 1});
  b->Args({1, 11, 11, 5, 5, 1, 1, 256, 1, 1});
  b->Args({1, 11, 11, 5, 5, 1, 1, 128, 1, 1});
  b->Args({1, 11, 11, 5, 5, 1, 1, 64, 1, 1});
  b->Args({1, 11, 11, 5, 5, 1, 1, 32, 1, 1});
  b->Args({1, 11, 11, 5, 5, 1, 1, 16, 1, 1});
  /*********************** 7 x 7 **********************/
  /*       N   H   W  KH  KW  S  D    G   GCin  GCout */
  b->Args({1, 7, 7, 5, 5, 1, 1, 1024, 1, 1});
  b->Args({1, 7, 7, 5, 5, 1, 1, 768, 1, 1});
  b->Args({1, 7, 7, 5, 5, 1, 1, 512, 1, 1});
  b->Args({1, 7, 7, 5, 5, 1, 1, 384, 1, 1});
  b->Args({1, 7, 7, 5, 5, 1, 1, 256, 1, 1});
  b->Args({1, 7, 7, 5, 5, 1, 1, 128, 1, 1});
  b->Args({1, 7, 7, 5, 5, 1, 1, 64, 1, 1});
  b->Args({1, 7, 7, 5, 5, 1, 1, 32, 1, 1});
  b->Args({1, 7, 7, 5, 5, 1, 1, 16, 1, 1});
}

BENCHMARK_CAPTURE(convolution_q8, mobilenet_v1, "MobileNet v1")
    ->Apply(MobileNetV1);
BENCHMARK_CAPTURE(convolution_q8, mobilenet_v2, "MobileNet v2")
    ->Apply(MobileNetV2);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")
    ->Apply(ShuffleNetV1G1);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")
    ->Apply(ShuffleNetV1G2);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")
    ->Apply(ShuffleNetV1G3);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")
    ->Apply(ShuffleNetV1G4);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")
    ->Apply(ShuffleNetV1G8);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v2_x05, "ShuffleNet v2 0.5X")
    ->Apply(ShuffleNetV2X05);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v2_x10, "ShuffleNet v2 1.0X")
    ->Apply(ShuffleNetV2X10);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v2_x15, "ShuffleNet v2 1.5X")
    ->Apply(ShuffleNetV2X15);
BENCHMARK_CAPTURE(convolution_q8, shufflenet_v2_x20, "ShuffleNet v2 2.0X")
    ->Apply(ShuffleNetV2X20);
BENCHMARK_CAPTURE(convolution_q8, squeezenet_v10, "SqueezeNet 1.0")
    ->Apply(SqueezeNetV10);
BENCHMARK_CAPTURE(convolution_q8, squeezenet_v11, "SqueezeNet 1.1")
    ->Apply(SqueezeNetV11);
BENCHMARK_CAPTURE(convolution_q8, resnet18, "ResNet-18")->Apply(ResNet18);
BENCHMARK_CAPTURE(convolution_q8, resnet50, "ResNet-50")->Apply(ResNet50);
BENCHMARK_CAPTURE(convolution_q8, vgg, "VGG")->Apply(VGG);
BENCHMARK_CAPTURE(convolution_q8, dwconv3x3, "3x3 DW Convolutions")
    ->Apply(DWConv3x3);
BENCHMARK_CAPTURE(
    convolution_q8,
    dwconv3x3d2,
    "3x3 DW Convolutions (dilation 2)")
    ->Apply(DWConv3x3d2);
BENCHMARK_CAPTURE(convolution_q8, dwconv5x5, "5x5 DW Convolutions")
    ->Apply(DWConv5x5);
BENCHMARK_CAPTURE(convolution_q8, dwconv3x3_per_channel, "3x3 DW Convolutions", true)
    ->Apply(DWConv3x3);
BENCHMARK_CAPTURE(
    convolution_q8,
    dwconv3x3d2_per_channel,
    "3x3 DW Convolutions (dilation 2)", true)
    ->Apply(DWConv3x3d2);
BENCHMARK_CAPTURE(convolution_q8, dwconv5x5_per_channel, "5x5 DW Convolutions", true)
    ->Apply(DWConv5x5);

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
