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

static void channel_shuffle_x8(benchmark::State& state, const char* net) {
  const size_t batchSize = static_cast<size_t>(state.range(0));
  const size_t groups = static_cast<size_t>(state.range(1));
  const size_t groupChannels = static_cast<size_t>(state.range(2));

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  std::vector<uint8_t> input(batchSize * groups * groupChannels);
  std::vector<uint8_t> output(batchSize * groups * groupChannels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));

  pytorch_qnnp_status status = pytorch_qnnp_initialize();
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to initialize QNNPACK");
  }

  pytorch_qnnp_operator_t channelShuffleOperator = nullptr;
  status = pytorch_qnnp_create_channel_shuffle_nc_x8(
      groups, groupChannels, 0 /* flags */, &channelShuffleOperator);
  if (status != pytorch_qnnp_status_success ||
      channelShuffleOperator == nullptr) {
    state.SkipWithError("failed to create X8 Channel Shuffle operator");
  }

  status = pytorch_qnnp_setup_channel_shuffle_nc_x8(
      channelShuffleOperator,
      batchSize,
      input.data(),
      groups * groupChannels /* input:stride */,
      output.data(),
      groups * groupChannels /* output:stride */);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup X8 Channel Shuffle operator");
  }

  for (auto _ : state) {
    status = pytorch_qnnp_run_operator(
        channelShuffleOperator, nullptr /* thread pool */);
    if (status != pytorch_qnnp_status_success) {
      state.SkipWithError("failed to run X8 Channel Shuffle operator");
    }
  }

  const size_t itemsPerIteration = batchSize * groups * groupChannels;
  state.SetItemsProcessed(
      int64_t(state.iterations()) * int64_t(itemsPerIteration));

  const size_t bytesPerIteration = 2 * itemsPerIteration * sizeof(uint8_t);
  state.SetBytesProcessed(
      int64_t(state.iterations()) * int64_t(bytesPerIteration));

  status = pytorch_qnnp_delete_operator(channelShuffleOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete X8 Channel Shuffle operator");
  }
}

static void ShuffleNetV1G2Arguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 ********/
  /*        H    W  G   CG */
  b->Args({56 * 56, 2, 25});
  b->Args({28 * 28, 2, 25});

  /******** Stage 3 ********/
  /*        H    W  G   CG */
  b->Args({28 * 28, 2, 50});
  b->Args({14 * 14, 2, 50});

  /******** Stage 4 ********/
  /*        H    W  G   CG */
  b->Args({14 * 14, 2, 100});
  b->Args({7 * 7, 2, 100});
}

static void ShuffleNetV1G3Arguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 *******/
  /*        H    W  G  CG */
  b->Args({56 * 56, 3, 20});
  b->Args({28 * 28, 3, 20});

  /******** Stage 3 *******/
  /*        H    W  G  CG */
  b->Args({28 * 28, 3, 40});
  b->Args({14 * 14, 3, 40});

  /******** Stage 4 *******/
  /*        H    W  G  CG */
  b->Args({14 * 14, 3, 80});
  b->Args({7 * 7, 3, 80});
}

static void ShuffleNetV1G4Arguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 *******/
  /*        H    W  G  CG */
  b->Args({56 * 56, 4, 17});
  b->Args({28 * 28, 4, 17});

  /******** Stage 3 *******/
  /*        H    W  G  CG */
  b->Args({28 * 28, 4, 34});
  b->Args({14 * 14, 4, 34});

  /******** Stage 4 *******/
  /*        H    W  G  CG */
  b->Args({14 * 14, 4, 68});
  b->Args({7 * 7, 4, 68});
}

static void ShuffleNetV1G8Arguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 *******/
  /*        H    W  G  CG */
  b->Args({56 * 56, 8, 12});
  b->Args({28 * 28, 8, 12});

  /******** Stage 3 *******/
  /*        H    W  G  CG */
  b->Args({28 * 28, 8, 24});
  b->Args({14 * 14, 8, 24});

  /******** Stage 4 *******/
  /*        H    W  G  CG */
  b->Args({14 * 14, 8, 48});
  b->Args({7 * 7, 8, 48});
}

static void ShuffleNetV2x0_5Arguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 *******/
  /*        H    W  G  CG */
  b->Args({28 * 28, 2, 24});

  /******** Stage 3 *******/
  /*        H    W  G  CG */
  b->Args({14 * 14, 2, 48});

  /******** Stage 4 *******/
  /*        H    W  G  CG */
  b->Args({7 * 7, 2, 96});
}

static void ShuffleNetV2x1_0Arguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 ********/
  /*        H    W  G   CG */
  b->Args({28 * 28, 2, 58});

  /******** Stage 3 ********/
  /*        H    W  G   CG */
  b->Args({14 * 14, 2, 116});

  /******** Stage 4 ********/
  /*        H    W  G   CG */
  b->Args({7 * 7, 2, 232});
}

static void ShuffleNetV2x1_5Arguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 ********/
  /*        H    W  G   CG */
  b->Args({28 * 28, 2, 88});

  /******** Stage 3 ********/
  /*        H    W  G   CG */
  b->Args({14 * 14, 2, 176});

  /******** Stage 4 ********/
  /*        H    W  G   CG */
  b->Args({7 * 7, 2, 352});
}

static void ShuffleNetV2x2_0Arguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "G", "GC"});

  /******** Stage 2 ********/
  /*        H    W  G   CG */
  b->Args({28 * 28, 2, 122});

  /******** Stage 3 ********/
  /*        H    W  G   CG */
  b->Args({14 * 14, 2, 244});

  /******** Stage 4 ********/
  /*        H    W  G   CG */
  b->Args({7 * 7, 2, 488});
}

BENCHMARK_CAPTURE(
    channel_shuffle_x8,
    shufflenet_v1_g2,
    "ShuffleNet v1 (2 groups)")
    ->Apply(ShuffleNetV1G2Arguments);
BENCHMARK_CAPTURE(
    channel_shuffle_x8,
    shufflenet_v1_g3,
    "ShuffleNet v1 (3 groups)")
    ->Apply(ShuffleNetV1G3Arguments);
BENCHMARK_CAPTURE(
    channel_shuffle_x8,
    shufflenet_v1_g4,
    "ShuffleNet v1 (4 groups)")
    ->Apply(ShuffleNetV1G4Arguments);
BENCHMARK_CAPTURE(
    channel_shuffle_x8,
    shufflenet_v1_g8,
    "ShuffleNet v1 (8 groups)")
    ->Apply(ShuffleNetV1G8Arguments);
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x05, "ShuffleNet v2 x0.5")
    ->Apply(ShuffleNetV2x0_5Arguments);
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x10, "ShuffleNet v2 x1.0")
    ->Apply(ShuffleNetV2x1_0Arguments);
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x15, "ShuffleNet v2 x1.5")
    ->Apply(ShuffleNetV2x1_5Arguments);
BENCHMARK_CAPTURE(channel_shuffle_x8, shufflenet_v2_x20, "ShuffleNet v2 x2.0")
    ->Apply(ShuffleNetV2x2_0Arguments);

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
