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
#include <fp16.h>
#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/hgemm.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

#include <benchmark/benchmark.h>

inline uint32_t divideRoundUp(uint32_t x, uint32_t q) {
  return x / q + uint32_t(x % q != 0);
}

inline uint32_t roundUp(uint32_t x, uint32_t q) {
  return q * divideRoundUp(x, q);
}

class HGEMM : public benchmark::Fixture {
 public:
  inline HGEMM(uint32_t mr, uint32_t nr, uint32_t kr)
      : mr_(mr), nr_(nr), kr_(kr), mc_(mr), nc_(nr), kc_(kr) {}

  virtual void SetUp(const benchmark::State&) override {
    const uint_fast32_t seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    auto rng = std::bind(
        fp16_ieee_from_fp32_value,
        std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed)));

    a_.resize(mc() * kc());
    std::generate(a_.begin(), a_.end(), std::ref(rng));
    k_.resize(nc() * kc());
    std::generate(k_.begin(), k_.end(), std::ref(rng));
    b_.resize(nc());
    std::generate(b_.begin(), b_.end(), std::ref(rng));
    w_.resize(ncStride() * kcStride() + ncStride());
    std::fill(w_.begin(), w_.end(), 0);
    pytorch_pack_hgemm_w(nc(), kc(), nr(), kr(), k(), b(), w());
    c_.resize(mc() * nc());
    std::fill(c_.begin(), c_.end(), UINT16_C(0x7E00) /* NaN */);
  }

  virtual void TearDown(benchmark::State& state) override {
    state.SetItemsProcessed(
        uint64_t(state.iterations()) * 2 * mc() * nc() * kc());
    a_.clear();
    k_.clear();
    b_.clear();
    w_.clear();
    c_.clear();
  }

  inline const uint16_t* a() const {
    return a_.data();
  }

  inline const uint16_t* k() const {
    return k_.data();
  }

  inline const uint16_t* b() const {
    return b_.data();
  }

  inline uint16_t* w() {
    return w_.data();
  }

  inline const uint16_t* w() const {
    return w_.data();
  }

  inline uint16_t* c() {
    return c_.data();
  }

  inline uint32_t mr() const {
    return mr_;
  }

  inline uint32_t mc() const {
    return mc_;
  }

  inline uint32_t nr() const {
    return nr_;
  }

  inline uint32_t nc() const {
    return nc_;
  }

  inline uint32_t ncStride() const {
    return roundUp(nc(), nr());
  }

  inline uint32_t kr() const {
    return kr_;
  }

  inline uint32_t kc() const {
    return kc_;
  }

  inline uint32_t kcStride() const {
    return roundUp(kc(), kr());
  }

  inline const pytorch_qnnp_fp16_clamping_params* clampingParams() const {
    return &clampingParams_;
  }

 protected:
  std::vector<uint16_t> a_;
  std::vector<uint16_t> k_;
  std::vector<uint16_t> b_;
  std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> w_;
  std::vector<uint16_t> c_;
  uint32_t mr_{0};
  uint32_t nr_{0};
  uint32_t kr_{0};
  uint32_t mc_{mr_};
  uint32_t nc_{nr_};
  uint32_t kc_{kr_};
  pytorch_qnnp_fp16_clamping_params clampingParams_{0x3C00, 0x7C00, 0xFC00};
};

template <uint32_t MR, uint32_t NR, uint32_t KR>
class HGEMM_L1 : public HGEMM {
 public:
  inline HGEMM_L1() : HGEMM(MR, NR, KR) {
    cpuinfo_initialize();
    const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
    const size_t l1d_reserve = 512;
    kc_ = ((l1d_size - l1d_reserve) / sizeof(uint16_t) - mr() * nr()) /
        (mr() + nr());
    if (kr() != 1) {
      kc_ = kc_ / kr() * kr();
    } else {
      kc_ = kc_ / nr() * nr();
    }
  }
};

template <uint32_t MR, uint32_t NR, uint32_t KR>
class HGEMM_Op : public HGEMM {
 public:
  inline HGEMM_Op() : HGEMM(MR, NR, KR) {}

  virtual void SetUp(const benchmark::State& state) override {
    mc_ = state.range(0);
    nc_ = state.range(1);
    kc_ = state.range(2);

    HGEMM::SetUp(state);
  }
};

static void ShuffleNetV1G1GemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /* group = 1 */
  b->Args({56 * 56, 30, 24});
  b->Args({28 * 28, 120, 30});
  b->Args({28 * 28, 36, 144});
  b->Args({28 * 28, 144, 36});
  b->Args({14 * 14, 144, 36});
  b->Args({14 * 14, 72, 288});
  b->Args({14 * 14, 288, 72});
  b->Args({7 * 7, 288, 72});
  b->Args({7 * 7, 144, 576});
  b->Args({7 * 7, 576, 144});
}

static void ShuffleNetV1G2GemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /* group = 2 */
  b->Args({56 * 56, 22, 12});
  b->Args({28 * 28, 88, 22});
  b->Args({28 * 28, 25, 100});
  b->Args({28 * 28, 100, 25});
  b->Args({14 * 14, 100, 25});
  b->Args({14 * 14, 50, 200});
  b->Args({14 * 14, 200, 50});
  b->Args({7 * 7, 200, 50});
  b->Args({7 * 7, 100, 400});
  b->Args({7 * 7, 400, 100});
}

static void ShuffleNetV1G3GemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /* group = 3 */
  b->Args({56 * 56, 18, 8});
  b->Args({28 * 28, 72, 18});
  b->Args({28 * 28, 20, 80});
  b->Args({28 * 28, 80, 20});
  b->Args({14 * 14, 80, 20});
  b->Args({14 * 14, 40, 160});
  b->Args({14 * 14, 160, 40});
  b->Args({7 * 7, 160, 40});
  b->Args({7 * 7, 80, 320});
  b->Args({7 * 7, 320, 80});
}

static void ShuffleNetV1G4GemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /* group = 4 */
  b->Args({56 * 56, 15, 6});
  b->Args({28 * 28, 62, 15});
  b->Args({28 * 28, 17, 68});
  b->Args({28 * 28, 68, 17});
  b->Args({14 * 14, 68, 17});
  b->Args({14 * 14, 34, 136});
  b->Args({14 * 14, 136, 34});
  b->Args({7 * 7, 136, 34});
  b->Args({7 * 7, 68, 272});
  b->Args({7 * 7, 272, 68});
}

static void ShuffleNetV1G8GemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /* group = 8 */
  b->Args({56 * 56, 11, 3});
  b->Args({28 * 28, 45, 11});
  b->Args({28 * 28, 12, 48});
  b->Args({28 * 28, 48, 12});
  b->Args({14 * 14, 48, 12});
  b->Args({14 * 14, 24, 96});
  b->Args({14 * 14, 96, 24});
  b->Args({7 * 7, 96, 24});
  b->Args({7 * 7, 48, 192});
  b->Args({7 * 7, 192, 48});
}

static void MobileNetV1GemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  b->Args({112 * 112, 32, 3 * 3 * 3});
  b->Args({112 * 112, 64, 32});
  b->Args({56 * 56, 128, 64});
  b->Args({56 * 56, 128, 128});
  b->Args({28 * 28, 256, 128});
  b->Args({28 * 28, 256, 256});
  b->Args({14 * 14, 512, 256});
  b->Args({14 * 14, 512, 512});
  b->Args({7 * 7, 1024, 512});
  b->Args({7 * 7, 1024, 1024});
}

static void SqueezeNetV10GemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  /* Conv 1 */
  b->Args({111 * 111, 96, 7 * 7 * 3});
  /* Fire 2 */
  b->Args({55 * 55, 16, 96});
  b->Args({55 * 55, 64, 16});
  b->Args({55 * 55, 64, 3 * 3 * 16});
  /* Fire 3 */
  b->Args({55 * 55, 16, 128});
  b->Args({55 * 55, 64, 16});
  b->Args({55 * 55, 64, 3 * 3 * 16});
  /* Fire 4 */
  b->Args({55 * 55, 32, 128});
  b->Args({55 * 55, 128, 32});
  b->Args({55 * 55, 128, 3 * 3 * 32});
  /* Fire 5 */
  b->Args({27 * 27, 32, 256});
  b->Args({27 * 27, 128, 32});
  b->Args({27 * 27, 128, 3 * 3 * 32});
  /* Fire 6 */
  b->Args({27 * 27, 48, 256});
  b->Args({27 * 27, 192, 48});
  b->Args({27 * 27, 192, 3 * 3 * 48});
  /* Fire 7 */
  b->Args({27 * 27, 48, 384});
  b->Args({27 * 27, 192, 48});
  b->Args({27 * 27, 192, 3 * 3 * 48});
  /* Fire 8 */
  b->Args({27 * 27, 64, 384});
  b->Args({27 * 27, 256, 64});
  b->Args({27 * 27, 256, 3 * 3 * 64});
  /* Fire 9 */
  b->Args({13 * 13, 64, 512});
  b->Args({13 * 13, 256, 64});
  b->Args({13 * 13, 256, 3 * 3 * 64});
  /* Conv 10 */
  b->Args({13 * 13, 1000, 512});
}

static void GemmArguments(benchmark::internal::Benchmark* b) {
  for (auto S = 15; S <= 128; S *= 2) {
    for (int K = 8; K <= 1024; K *= 2) {
      b->Args({S * S, K, K});
    }
  }
}

#if CPUINFO_ARCH_ARM
BENCHMARK_TEMPLATE_F(HGEMM_L1, 8x8__aarch32_neonfp16arith, 8, 8, 1)
(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) {
    state.SkipWithError("NEON FP16 compute is not supported");
  }
  for (auto _ : state) {
    pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith(
        mr(),
        nr(),
        kc(),
        a(),
        kc() * sizeof(uint16_t),
        w() + nc() * (kcStride() + 1),
        c(),
        mr() * sizeof(uint16_t),
        clampingParams());
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(HGEMM_Op, 8x8__aarch32_neonfp16arith, 8, 8, 1)
(benchmark::State& state) {
  if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) {
    state.SkipWithError("NEON FP16 compute is not supported");
  }
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint16_t),
            w() + n * (kcStride() + 1),
            c() + m * nc() + n,
            nc() * sizeof(uint16_t),
            clampingParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(HGEMM_Op, 8x8__aarch32_neonfp16arith)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(HGEMM_Op, 8x8__aarch32_neonfp16arith)
    ->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(HGEMM_Op, 8x8__aarch32_neonfp16arith)
    ->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(HGEMM_Op, 8x8__aarch32_neonfp16arith)
    ->Apply(GemmArguments);
#endif

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
