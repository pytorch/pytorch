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

#include <cpuinfo.h>
#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/sgemm.h>

#include <benchmark/benchmark.h>

inline uint32_t divideRoundUp(uint32_t x, uint32_t q) {
  return x / q + uint32_t(x % q != 0);
}

inline uint32_t roundUp(uint32_t x, uint32_t q) {
  return q * divideRoundUp(x, q);
}

static void sgemmBenchmark(
    benchmark::State& state,
    pytorch_sgemm_ukernel_function sgemm,
    uint32_t mc,
    uint32_t nc,
    uint32_t kc,
    uint32_t mr,
    uint32_t nr,
    uint32_t np,
    uint32_t kr) {
  const size_t ncStride = roundUp(nc, np);
  const size_t kcStride = roundUp(kc, kr);

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

  std::vector<float> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(nc);
  std::generate(b.begin(), b.end(), std::ref(f32rng));
  std::vector<float, AlignedAllocator<float, 32>> w(
      ncStride * kcStride + ncStride);
  std::fill(w.begin(), w.end(), 0.0f);
  pytorch_pack_sgemm_w(nc, kc, nr, kr, k.data(), b.data(), w.data());
  std::vector<float> c(mc * nc);
  std::fill(c.begin(), c.end(), std::nanf(""));

  pytorch_qnnp_fp32_clamping_params clampingParams{
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity()};

  for (auto _ : state) {
    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        sgemm(
            mb,
            nb,
            kc,
            a.data() + m * kc,
            kc * sizeof(float),
            w.data() + n * (kcStride + 1),
            c.data() + m * nc + n,
            nc * sizeof(float),
            &clampingParams);
      }
    }
  }

  state.SetItemsProcessed(uint64_t(state.iterations()) * 2 * mc * nc * kc);
}

static void sgemm_in_l1(
    benchmark::State& state,
    pytorch_sgemm_ukernel_function sgemm,
    uint32_t mr,
    uint32_t nr,
    uint32_t np,
    uint32_t kr) {
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
  }

  const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
  const size_t l1d_reserve = 512;
  const size_t kc = roundUp(
      ((l1d_size - l1d_reserve) / sizeof(float) - mr * nr) / (mr + nr),
      np * kr);

  sgemmBenchmark(state, sgemm, mr /* mc */, nr /* nc */, kc, mr, nr, np, kr);
}

static void sgemm(
    benchmark::State& state,
    pytorch_sgemm_ukernel_function sgemm,
    uint32_t mr,
    uint32_t nr,
    uint32_t np,
    uint32_t kr) {
  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  sgemmBenchmark(state, sgemm, mc, nc, kc, mr, nr, np, kr);
}

/* ShuffleNet v1 with 1 group */
static void ShuffleNetV1G1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 36, 24 * 1 * 1});
  b->Args({28 * 28, 120, 36 * 1 * 1});
  b->Args({28 * 28, 36, 144 * 1 * 1});
  b->Args({28 * 28, 144, 36 * 1 * 1});
  b->Args({28 * 28, 72, 144 * 1 * 1});
  b->Args({14 * 14, 144, 72 * 1 * 1});
  b->Args({14 * 14, 72, 288 * 1 * 1});
  b->Args({14 * 14, 288, 72 * 1 * 1});
  b->Args({14 * 14, 144, 288 * 1 * 1});
  b->Args({7 * 7, 288, 144 * 1 * 1});
  b->Args({7 * 7, 144, 576 * 1 * 1});
  b->Args({7 * 7, 576, 144 * 1 * 1});
}

/* ShuffleNet v1 with 2 groups */
static void ShuffleNetV1G2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 50, 24 * 1 * 1});
  b->Args({28 * 28, 88, 25 * 1 * 1});
  b->Args({28 * 28, 25, 100 * 1 * 1});
  b->Args({28 * 28, 100, 25 * 1 * 1});
  b->Args({28 * 28, 50, 100 * 1 * 1});
  b->Args({14 * 14, 100, 50 * 1 * 1});
  b->Args({14 * 14, 50, 200 * 1 * 1});
  b->Args({14 * 14, 200, 50 * 1 * 1});
  b->Args({14 * 14, 100, 200 * 1 * 1});
  b->Args({7 * 7, 200, 100 * 1 * 1});
  b->Args({7 * 7, 100, 400 * 1 * 1});
  b->Args({7 * 7, 400, 100 * 1 * 1});
}

/* ShuffleNet v1 with 3 groups */
static void ShuffleNetV1G3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 60, 24 * 1 * 1});
  b->Args({28 * 28, 72, 20 * 1 * 1});
  b->Args({28 * 28, 20, 80 * 1 * 1});
  b->Args({28 * 28, 80, 20 * 1 * 1});
  b->Args({28 * 28, 40, 80 * 1 * 1});
  b->Args({14 * 14, 80, 40 * 1 * 1});
  b->Args({14 * 14, 40, 160 * 1 * 1});
  b->Args({14 * 14, 160, 40 * 1 * 1});
  b->Args({14 * 14, 80, 160 * 1 * 1});
  b->Args({7 * 7, 160, 80 * 1 * 1});
  b->Args({7 * 7, 80, 320 * 1 * 1});
  b->Args({7 * 7, 320, 80 * 1 * 1});
}

/* ShuffleNet v1 with 4 groups */
static void ShuffleNetV1G4(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 68, 24 * 1 * 1});
  b->Args({28 * 28, 62, 17 * 1 * 1});
  b->Args({28 * 28, 17, 68 * 1 * 1});
  b->Args({28 * 28, 68, 17 * 1 * 1});
  b->Args({28 * 28, 34, 68 * 1 * 1});
  b->Args({14 * 14, 68, 34 * 1 * 1});
  b->Args({14 * 14, 34, 136 * 1 * 1});
  b->Args({14 * 14, 136, 34 * 1 * 1});
  b->Args({14 * 14, 68, 136 * 1 * 1});
  b->Args({7 * 7, 136, 68 * 1 * 1});
  b->Args({7 * 7, 68, 272 * 1 * 1});
  b->Args({7 * 7, 272, 68 * 1 * 1});
}

/* ShuffleNet v1 with 8 groups */
static void ShuffleNetV1G8(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M       N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 96, 24 * 1 * 1});
  b->Args({28 * 28, 45, 12 * 1 * 1});
  b->Args({28 * 28, 12, 48 * 1 * 1});
  b->Args({28 * 28, 48, 12 * 1 * 1});
  b->Args({28 * 28, 24, 48 * 1 * 1});
  b->Args({14 * 14, 48, 24 * 1 * 1});
  b->Args({14 * 14, 24, 96 * 1 * 1});
  b->Args({14 * 14, 96, 24 * 1 * 1});
  b->Args({14 * 14, 48, 96 * 1 * 1});
  b->Args({7 * 7, 96, 48 * 1 * 1});
  b->Args({7 * 7, 48, 192 * 1 * 1});
  b->Args({7 * 7, 192, 48 * 1 * 1});
}

/* ShuffleNet v2 (0.5X scale) */
static void ShuffleNetV2X05(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 24, 24 * 1 * 1});
  b->Args({28 * 28, 24, 24 * 1 * 1});
  b->Args({28 * 28, 48, 48 * 1 * 1});
  b->Args({14 * 14, 48, 48 * 1 * 1});
  b->Args({14 * 14, 96, 96 * 1 * 1});
  b->Args({7 * 7, 96, 96 * 1 * 1});
  b->Args({7 * 7, 1024, 192 * 1 * 1});
}

/* ShuffleNet v2 (1.0X scale) */
static void ShuffleNetV2X10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 58, 24 * 1 * 1});
  b->Args({28 * 28, 58, 24 * 1 * 1});
  b->Args({28 * 28, 58, 58 * 1 * 1});
  b->Args({14 * 14, 116, 116 * 1 * 1});
  b->Args({14 * 14, 116, 116 * 1 * 1});
  b->Args({14 * 14, 232, 232 * 1 * 1});
  b->Args({7 * 7, 232, 232 * 1 * 1});
  b->Args({7 * 7, 1024, 464 * 1 * 1});
}

/* ShuffleNet v2 (1.5X scale) */
static void ShuffleNetV2X15(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 88, 24 * 1 * 1});
  b->Args({28 * 28, 88, 24 * 1 * 1});
  b->Args({28 * 28, 88, 88 * 1 * 1});
  b->Args({28 * 28, 176, 176 * 1 * 1});
  b->Args({14 * 14, 176, 176 * 1 * 1});
  b->Args({14 * 14, 352, 352 * 1 * 1});
  b->Args({7 * 7, 352, 352 * 1 * 1});
  b->Args({7 * 7, 1024, 704 * 1 * 1});
}

/* ShuffleNet v2 (2.0X scale) */
static void ShuffleNetV2X20(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112, 24, 3 * 3 * 3});
  b->Args({56 * 56, 122, 24 * 1 * 1});
  b->Args({28 * 28, 122, 24 * 1 * 1});
  b->Args({28 * 28, 122, 122 * 1 * 1});
  b->Args({28 * 28, 244, 244 * 1 * 1});
  b->Args({14 * 14, 244, 244 * 1 * 1});
  b->Args({14 * 14, 488, 488 * 1 * 1});
  b->Args({7 * 7, 488, 488 * 1 * 1});
  b->Args({7 * 7, 2048, 976 * 1 * 1});
}

static void MobileNetV1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N          K    */
  b->Args({112 * 112, 32, 3 * 3 * 3});
  b->Args({112 * 112, 64, 32 * 1 * 1});
  b->Args({56 * 56, 128, 64 * 1 * 1});
  b->Args({56 * 56, 128, 128 * 1 * 1});
  b->Args({28 * 28, 256, 128 * 1 * 1});
  b->Args({28 * 28, 256, 256 * 1 * 1});
  b->Args({14 * 14, 512, 256 * 1 * 1});
  b->Args({14 * 14, 512, 512 * 1 * 1});
  b->Args({7 * 7, 1024, 512 * 1 * 1});
  b->Args({7 * 7, 1024, 1024 * 1 * 1});
}

static void MobileNetV2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N          K    */
  b->Args({112 * 112, 32, 3 * 3 * 3});
  /************ Bottleneck 1 ************/
  b->Args({112 * 112, 16, 32 * 1 * 1});
  /************ Bottleneck 2 ************/
  b->Args({112 * 112, 96, 16 * 1 * 1});
  b->Args({56 * 56, 24, 96 * 1 * 1});
  b->Args({56 * 56, 144, 24 * 1 * 1});
  b->Args({56 * 56, 24, 144 * 1 * 1});
  /************ Bottleneck 3 ************/
  b->Args({28 * 28, 32, 144 * 1 * 1});
  b->Args({28 * 28, 192, 32 * 1 * 1});
  b->Args({28 * 28, 32, 192 * 1 * 1});
  /************ Bottleneck 4 ************/
  b->Args({14 * 14, 64, 192 * 1 * 1});
  b->Args({14 * 14, 192, 64 * 1 * 1});
  b->Args({14 * 14, 64, 384 * 1 * 1});
  /************ Bottleneck 5 ************/
  b->Args({14 * 14, 96, 384 * 1 * 1});
  b->Args({14 * 14, 576, 96 * 1 * 1});
  b->Args({14 * 14, 96, 576 * 1 * 1});
  /************ Bottleneck 6 ************/
  b->Args({7 * 7, 160, 576 * 1 * 1});
  b->Args({7 * 7, 960, 160 * 1 * 1});
  b->Args({7 * 7, 160, 960 * 1 * 1});
  /************ Bottleneck 7 ************/
  b->Args({7 * 7, 320, 960 * 1 * 1});
  /********* Pre-pooling Conv2D *********/
  b->Args({7 * 7, 1280, 320 * 1 * 1});
  /******** Post-pooling Conv2D *********/
  b->Args({1 * 1, 1000, 1280 * 1 * 1});
}

/* SqueezeNet 1.0 */
static void SqueezeNetV10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  /*************** Conv 1 ***************/
  b->Args({111 * 111, 96, 3 * 7 * 7});
  /*************** Fire 2 ***************/
  b->Args({55 * 55, 16, 96 * 1 * 1});
  b->Args({55 * 55, 64, 16 * 1 * 1});
  b->Args({55 * 55, 64, 16 * 3 * 3});
  /*************** Fire 3 ***************/
  b->Args({55 * 55, 16, 128 * 1 * 1});
  /*************** Fire 4 ***************/
  b->Args({55 * 55, 32, 128 * 1 * 1});
  b->Args({55 * 55, 128, 32 * 1 * 1});
  b->Args({55 * 55, 128, 32 * 3 * 3});
  /*************** Fire 5 ***************/
  b->Args({27 * 27, 32, 256 * 1 * 1});
  b->Args({27 * 27, 128, 32 * 1 * 1});
  b->Args({27 * 27, 128, 32 * 3 * 3});
  /*************** Fire 6 ***************/
  b->Args({27 * 27, 48, 256 * 1 * 1});
  b->Args({27 * 27, 192, 48 * 1 * 1});
  b->Args({27 * 27, 192, 48 * 3 * 3});
  /*************** Fire 7 ***************/
  b->Args({27 * 27, 48, 384 * 1 * 1});
  /*************** Fire 8 ***************/
  b->Args({27 * 27, 64, 384 * 1 * 1});
  b->Args({27 * 27, 256, 64 * 1 * 1});
  b->Args({27 * 27, 256, 64 * 3 * 3});
  /*************** Fire 9 ***************/
  b->Args({13 * 13, 64, 512 * 1 * 1});
  b->Args({13 * 13, 256, 64 * 1 * 1});
  b->Args({13 * 13, 256, 64 * 3 * 3});
  /*************** Conv 10 **************/
  b->Args({13 * 13, 1000, 512 * 1 * 1});
}

/* SqueezeNet 1.1 */
static void SqueezeNetV11(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  /*************** Conv 1 ***************/
  b->Args({111 * 111, 64, 3 * 3 * 3});
  /*************** Fire 2 ***************/
  b->Args({55 * 55, 16, 64 * 1 * 1});
  b->Args({55 * 55, 64, 16 * 1 * 1});
  b->Args({55 * 55, 64, 16 * 3 * 3});
  /*************** Fire 3 ***************/
  b->Args({55 * 55, 16, 128 * 1 * 1});
  /*************** Fire 4 ***************/
  b->Args({27 * 27, 32, 128 * 1 * 1});
  b->Args({27 * 27, 128, 32 * 1 * 1});
  b->Args({27 * 27, 128, 32 * 3 * 3});
  /*************** Fire 5 ***************/
  b->Args({27 * 27, 32, 256 * 1 * 1});
  /*************** Fire 6 ***************/
  b->Args({13 * 13, 48, 256 * 1 * 1});
  b->Args({13 * 13, 192, 48 * 1 * 1});
  b->Args({13 * 13, 192, 48 * 3 * 3});
  /*************** Fire 7 ***************/
  b->Args({13 * 13, 48, 384 * 1 * 1});
  /*************** Fire 8 ***************/
  b->Args({13 * 13, 64, 384 * 1 * 1});
  b->Args({13 * 13, 256, 64 * 1 * 1});
  b->Args({13 * 13, 256, 64 * 3 * 3});
  /*************** Fire 9 ***************/
  b->Args({13 * 13, 64, 512 * 1 * 1});
  /*************** Conv 10 **************/
  b->Args({13 * 13, 1000, 512 * 1 * 1});
}

static void ResNet18(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K    */
  b->Args({112 * 112, 64, 3 * 7 * 7});
  b->Args({56 * 56, 64, 64 * 3 * 3});
  b->Args({28 * 28, 128, 64 * 3 * 3});
  b->Args({28 * 28, 128, 128 * 3 * 3});
  b->Args({28 * 28, 128, 64 * 1 * 1});
  b->Args({14 * 14, 256, 128 * 3 * 3});
  b->Args({14 * 14, 256, 256 * 3 * 3});
  b->Args({14 * 14, 256, 128 * 1 * 1});
  b->Args({7 * 7, 512, 256 * 3 * 3});
  b->Args({7 * 7, 512, 512 * 3 * 3});
  b->Args({7 * 7, 512, 256 * 1 * 1});
}

static void ResNet50(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N         K     */
  /**************** Conv 1 ***************/
  b->Args({112 * 112, 64, 3 * 7 * 7});
  /*           M        N          K     */
  /*************** Conv 2.X **************/
  b->Args({56 * 56, 64, 64 * 1 * 1});
  b->Args({56 * 56, 64, 64 * 3 * 3});
  b->Args({56 * 56, 256, 64 * 1 * 1});
  b->Args({56 * 56, 64, 256 * 1 * 1});
  /*           M        N          K     */
  /*************** Conv 3.X **************/
  b->Args({56 * 56, 128, 256 * 1 * 1});
  b->Args({28 * 28, 128, 128 * 3 * 3});
  b->Args({28 * 28, 512, 128 * 1 * 1});
  b->Args({28 * 28, 512, 256 * 1 * 1});
  b->Args({28 * 28, 128, 512 * 1 * 1});
  /*           M        N          K     */
  /*************** Conv 4.X **************/
  b->Args({28 * 28, 256, 512 * 1 * 1});
  b->Args({14 * 14, 256, 256 * 3 * 3});
  b->Args({14 * 14, 1024, 256 * 1 * 1});
  b->Args({14 * 14, 1024, 512 * 1 * 1});
  b->Args({14 * 14, 256, 1024 * 1 * 1});
  /*           M        N          K     */
  /*************** Conv 5.X **************/
  b->Args({14 * 14, 512, 1024 * 1 * 1});
  b->Args({7 * 7, 512, 512 * 3 * 3});
  b->Args({7 * 7, 2048, 512 * 1 * 1});
  b->Args({7 * 7, 2048, 1024 * 1 * 1});
  b->Args({7 * 7, 512, 2048 * 1 * 1});
}

static void VGG(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /*           M        N        K     */
  /************** Conv 1.1 *************/
  b->Args({224 * 224, 64, 3 * 3 * 3});
  /************** Conv 1.2 *************/
  b->Args({224 * 224, 64, 64 * 3 * 3});
  /************** Conv 2.1 *************/
  b->Args({112 * 112, 128, 64 * 3 * 3});
  /************** Conv 2.2 *************/
  b->Args({112 * 112, 128, 128 * 3 * 3});
  /************** Conv 3.1 *************/
  b->Args({56 * 56, 256, 128 * 3 * 3});
  /************** Conv 3.3 *************/
  b->Args({56 * 56, 256, 256 * 1 * 1});
  /************** Conv 4.1 *************/
  b->Args({28 * 28, 512, 256 * 3 * 3});
  /************** Conv 4.2 *************/
  b->Args({28 * 28, 512, 512 * 3 * 3});
  /************** Conv 4.3 *************/
  b->Args({28 * 28, 512, 512 * 1 * 1});
  /************** Conv 5.X *************/
  b->Args({14 * 14, 512, 512 * 3 * 3});
  /************** Conv 5.3 *************/
  b->Args({14 * 14, 512, 512 * 1 * 1});
}

BENCHMARK_CAPTURE(
    sgemm_in_l1,
    6x8__psimd,
    pytorch_sgemm_ukernel_6x8__psimd,
    6,
    8,
    8,
    1);
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
BENCHMARK_CAPTURE(sgemm_in_l1, 5x8__neon, pytorch_sgemm_ukernel_5x8__neon, 5, 8, 8, 1);
BENCHMARK_CAPTURE(sgemm_in_l1, 6x8__neon, pytorch_sgemm_ukernel_6x8__neon, 6, 8, 8, 1);
#endif

static void sgemm_6x8__psimd(benchmark::State& state, const char* net) {
  sgemm(state, pytorch_sgemm_ukernel_6x8__psimd, 6, 8, 8, 1);
}

BENCHMARK_CAPTURE(sgemm_6x8__psimd, mobilenet_v1, "MobileNet v1")
    ->Apply(MobileNetV1);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, mobilenet_v2, "MobileNet v2")
    ->Apply(MobileNetV2);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")
    ->Apply(ShuffleNetV1G1);
BENCHMARK_CAPTURE(
    sgemm_6x8__psimd,
    shufflenet_v1_g2,
    "ShuffleNet v1 (2 groups)")
    ->Apply(ShuffleNetV1G2);
BENCHMARK_CAPTURE(
    sgemm_6x8__psimd,
    shufflenet_v1_g3,
    "ShuffleNet v1 (3 groups)")
    ->Apply(ShuffleNetV1G3);
BENCHMARK_CAPTURE(
    sgemm_6x8__psimd,
    shufflenet_v1_g4,
    "ShuffleNet v1 (4 groups)")
    ->Apply(ShuffleNetV1G4);
BENCHMARK_CAPTURE(
    sgemm_6x8__psimd,
    shufflenet_v1_g8,
    "ShuffleNet v1 (8 groups)")
    ->Apply(ShuffleNetV1G8);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, shufflenet_v2_x05, "ShuffleNet v2 0.5X")
    ->Apply(ShuffleNetV2X05);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, shufflenet_v2_x10, "ShuffleNet v2 1.0X")
    ->Apply(ShuffleNetV2X10);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, shufflenet_v2_x15, "ShuffleNet v2 1.5X")
    ->Apply(ShuffleNetV2X15);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, shufflenet_v2_x20, "ShuffleNet v2 2.0X")
    ->Apply(ShuffleNetV2X20);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, resnet18, "ResNet-18")->Apply(ResNet18);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, resnet50, "ResNet-50")->Apply(ResNet50);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, squeezenet_v10, "SqueezeNet 1.0")
    ->Apply(SqueezeNetV10);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, squeezenet_v11, "SqueezeNet 1.1")
    ->Apply(SqueezeNetV11);
BENCHMARK_CAPTURE(sgemm_6x8__psimd, vgg, "VGG")->Apply(VGG);

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
static void sgemm_5x8__neon(benchmark::State& state, const char* net) {
  sgemm(state, pytorch_sgemm_ukernel_5x8__neon, 5, 8, 8, 1);
}

static void sgemm_6x8__neon(benchmark::State& state, const char* net) {
  sgemm(state, pytorch_sgemm_ukernel_6x8__neon, 6, 8, 8, 1);
}

BENCHMARK_CAPTURE(sgemm_5x8__neon, mobilenet_v1, "MobileNet v1")
    ->Apply(MobileNetV1);
BENCHMARK_CAPTURE(sgemm_5x8__neon, mobilenet_v2, "MobileNet v2")
    ->Apply(MobileNetV2);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")
    ->Apply(ShuffleNetV1G1);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")
    ->Apply(ShuffleNetV1G2);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")
    ->Apply(ShuffleNetV1G3);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")
    ->Apply(ShuffleNetV1G4);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")
    ->Apply(ShuffleNetV1G8);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v2_x05, "ShuffleNet v2 0.5X")
    ->Apply(ShuffleNetV2X05);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v2_x10, "ShuffleNet v2 1.0X")
    ->Apply(ShuffleNetV2X10);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v2_x15, "ShuffleNet v2 1.5X")
    ->Apply(ShuffleNetV2X15);
BENCHMARK_CAPTURE(sgemm_5x8__neon, shufflenet_v2_x20, "ShuffleNet v2 2.0X")
    ->Apply(ShuffleNetV2X20);
BENCHMARK_CAPTURE(sgemm_5x8__neon, resnet18, "ResNet-18")->Apply(ResNet18);
BENCHMARK_CAPTURE(sgemm_5x8__neon, resnet50, "ResNet-50")->Apply(ResNet50);
BENCHMARK_CAPTURE(sgemm_5x8__neon, squeezenet_v10, "SqueezeNet 1.0")
    ->Apply(SqueezeNetV10);
BENCHMARK_CAPTURE(sgemm_5x8__neon, squeezenet_v11, "SqueezeNet 1.1")
    ->Apply(SqueezeNetV11);
BENCHMARK_CAPTURE(sgemm_5x8__neon, vgg, "VGG")->Apply(VGG);

BENCHMARK_CAPTURE(sgemm_6x8__neon, mobilenet_v1, "MobileNet v1")
    ->Apply(MobileNetV1);
BENCHMARK_CAPTURE(sgemm_6x8__neon, mobilenet_v2, "MobileNet v2")
    ->Apply(MobileNetV2);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")
    ->Apply(ShuffleNetV1G1);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")
    ->Apply(ShuffleNetV1G2);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")
    ->Apply(ShuffleNetV1G3);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")
    ->Apply(ShuffleNetV1G4);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")
    ->Apply(ShuffleNetV1G8);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v2_x05, "ShuffleNet v2 0.5X")
    ->Apply(ShuffleNetV2X05);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v2_x10, "ShuffleNet v2 1.0X")
    ->Apply(ShuffleNetV2X10);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v2_x15, "ShuffleNet v2 1.5X")
    ->Apply(ShuffleNetV2X15);
BENCHMARK_CAPTURE(sgemm_6x8__neon, shufflenet_v2_x20, "ShuffleNet v2 2.0X")
    ->Apply(ShuffleNetV2X20);
BENCHMARK_CAPTURE(sgemm_6x8__neon, resnet18, "ResNet-18")->Apply(ResNet18);
BENCHMARK_CAPTURE(sgemm_6x8__neon, resnet50, "ResNet-50")->Apply(ResNet50);
BENCHMARK_CAPTURE(sgemm_6x8__neon, squeezenet_v10, "SqueezeNet 1.0")
    ->Apply(SqueezeNetV10);
BENCHMARK_CAPTURE(sgemm_6x8__neon, squeezenet_v11, "SqueezeNet 1.1")
    ->Apply(SqueezeNetV11);
BENCHMARK_CAPTURE(sgemm_6x8__neon, vgg, "VGG")->Apply(VGG);
#endif

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
