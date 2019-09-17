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
#include <qnnpack/q8gemm.h>
#include <qnnpack/requantization.h>

#include <benchmark/benchmark.h>

#if PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP
#include <gemmlowp/public/gemmlowp.h>
#endif

inline uint32_t divideRoundUp(uint32_t x, uint32_t q) {
  return x / q + uint32_t(x % q != 0);
}

inline uint32_t roundUp(uint32_t x, uint32_t q) {
  return q * divideRoundUp(x, q);
}

#if PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP
struct GemmlowpOutputPipeline {
  typedef gemmlowp::VectorMap<const int32_t, gemmlowp::VectorShape::Col>
      ColVectorMap;
  typedef std::tuple<
      gemmlowp::OutputStageBiasAddition<ColVectorMap>,
      gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint,
      gemmlowp::OutputStageClamp,
      gemmlowp::OutputStageSaturatingCastToUint8>
      Pipeline;

  static Pipeline Make(
      const int32_t* bias_data,
      int output_rows,
      int32_t output_offset,
      int32_t output_multiplier,
      int output_shift,
      int32_t output_activation_min,
      int32_t output_activation_max) {
    ColVectorMap bias_vector(bias_data, output_rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint
        quantize_down_stage;
    quantize_down_stage.result_offset_after_shift = output_offset;
    quantize_down_stage.result_fixedpoint_multiplier = output_multiplier;
    quantize_down_stage.result_shift = output_shift;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = output_activation_min;
    clamp_stage.max = output_activation_max;
    gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
    return std::make_tuple(
        bias_addition_stage,
        quantize_down_stage,
        clamp_stage,
        saturating_cast_stage);
  }
};
#endif

class Q8GEMM : public benchmark::Fixture {
 public:
  inline Q8GEMM(uint32_t mr, uint32_t nr, uint32_t np, uint32_t kr)
      : mr_(mr), nr_(nr), np_(np), kr_(kr), mc_(mr), nc_(nr), kc_(kr) {}

  virtual void SetUp(const benchmark::State&) override {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    a_.resize(mc() * kc());
    std::generate(a_.begin(), a_.end(), std::ref(u8rng));
    k_.resize(nc() * kc());
    std::generate(k_.begin(), k_.end(), std::ref(u8rng));
    b_.resize(nc());
    std::generate(b_.begin(), b_.end(), std::ref(s32rng));
    w_.resize(
        kcStride() * ncStride() +
        ncStride() * sizeof(int32_t) / sizeof(uint8_t));
    std::fill(w_.begin(), w_.end(), 127);
    pytorch_pack_q8gemm_w(
        nc(),
        kc(),
        nr(),
        np(),
        kr(),
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        127,
        127,
#endif
        k(),
        b(),
        w());
    c_.resize(mc() * nc());
    std::fill(c_.begin(), c_.end(), 0xA5);

    quantizationParams_ = pytorch_qnnp_compute_conv_quantization_params(
        127, 127, 0.75f, 127, 1, 254);
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

  inline const uint8_t* a() const {
    return a_.data();
  }

  inline const uint8_t* k() const {
    return k_.data();
  }

  inline const int32_t* b() const {
    return b_.data();
  }

  inline uint8_t* w() {
    return w_.data();
  }

  inline const uint8_t* w() const {
    return w_.data();
  }

  inline uint8_t* c() {
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

  inline uint32_t np() const {
    return np_;
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

  inline const pytorch_qnnp_conv_quantization_params* quantizationParams()
      const {
    return &quantizationParams_;
  }

 protected:
  std::vector<uint8_t> a_;
  std::vector<uint8_t> k_;
  std::vector<int32_t> b_;
  std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> w_;
  std::vector<uint8_t> c_;
  uint32_t mr_{0};
  uint32_t nr_{0};
  uint32_t np_{0};
  uint32_t kr_{0};
  uint32_t mc_{mr_};
  uint32_t nc_{nr_};
  uint32_t kc_{kr_};
  pytorch_qnnp_conv_quantization_params quantizationParams_;
};

template <uint32_t MR, uint32_t NR, uint32_t NP, uint32_t KR>
class Q8GEMM_L1 : public Q8GEMM {
 public:
  inline Q8GEMM_L1() : Q8GEMM(MR, NR, NP, KR) {
    cpuinfo_initialize();
    const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
    const size_t l1d_reserve = 512;
    kc_ = ((l1d_size - l1d_reserve) / sizeof(uint8_t) - mr() * nr()) /
        (mr() + nr());
    if (kr() != 1) {
      kc_ = kc_ / kr() * kr();
    } else {
      kc_ = kc_ / nr() * nr();
    }
  }
};

template <uint32_t MR, uint32_t NR, uint32_t NP, uint32_t KR>
class Q8GEMM_Op : public Q8GEMM {
 public:
  inline Q8GEMM_Op() : Q8GEMM(MR, NR, NP, KR) {}

  virtual void SetUp(const benchmark::State& state) override {
    mc_ = state.range(0);
    nc_ = state.range(1);
    kc_ = state.range(2);

    Q8GEMM::SetUp(state);
  }
};

class Q8GEMM_XZP : public Q8GEMM {
 public:
  inline Q8GEMM_XZP(uint32_t mr, uint32_t nr, uint32_t np, uint32_t kr)
      : Q8GEMM(mr, nr, np, kr) {}
  virtual void SetUp(const benchmark::State&) override {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    a_.resize(mc() * kc());
    std::generate(a_.begin(), a_.end(), std::ref(u8rng));
    k_.resize(ncStride() * kcStride());
    std::generate(k_.begin(), k_.end(), std::ref(u8rng));
    b_.resize(roundUp(nc(), nr()));
    std::generate(b_.begin(), b_.end(), std::ref(s32rng));
    w_.resize(ncStride() * (kcStride() + sizeof(int32_t) / sizeof(uint8_t)));
    std::fill(w_.begin(), w_.end(), 127);
    pytorch_pack_swizzle_q8gemm_b(
        nc(),
        kc(),
        np(),
        kr(),
        8,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        127,
        127,
#endif
        k(),
        b(),
        w());
    c_.resize(mc() * nc());
    std::fill(c_.begin(), c_.end(), 0xA5);
    aRowSums_.resize(roundUp(mc(), mr()));
    std::fill(aRowSums_.begin(), aRowSums_.end(), 0xFE01);

    requantizationParams_ =
        pytorch_qnnp_compute_requantization_params(0.75f, 127, 1, 254);
  }

  virtual void TearDown(benchmark::State& state) override {
    state.SetItemsProcessed(
        uint64_t(state.iterations()) * 2 * mc() * nc() * kc());
    a_.clear();
    k_.clear();
    c_.clear();
    aRowSums_.clear();
  }

  inline int32_t* aRowSums() {
    return aRowSums_.data();
  }

  inline const int32_t* aRowSums() const {
    return aRowSums_.data();
  }

  inline const pytorch_qnnp_q31_requantization_params* requantizationParams()
      const {
    return &requantizationParams_;
  }

 protected:
  std::vector<int32_t> aRowSums_;
  pytorch_qnnp_q31_requantization_params requantizationParams_;
};

template <uint32_t MR, uint32_t NR, uint32_t NP, uint32_t KR>
class Q8GEMM_XZP_L1 : public Q8GEMM_XZP {
 public:
  inline Q8GEMM_XZP_L1() : Q8GEMM_XZP(MR, NR, NP, KR) {
    cpuinfo_initialize();
    const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
    const size_t l1d_reserve = 512;
    kc_ = ((l1d_size - l1d_reserve) / sizeof(uint8_t) - mr() * nr()) /
        (mr() + nr());
    if (kr() != 1) {
      kc_ = kc_ / kr() * kr();
    } else {
      kc_ = kc_ / nr() * nr();
    }
  }
};

template <uint32_t MR, uint32_t NR, uint32_t NP, uint32_t KR>
class Q8GEMM_XZP_Op : public Q8GEMM_XZP {
 public:
  inline Q8GEMM_XZP_Op() : Q8GEMM_XZP(MR, NR, NP, KR) {}

  virtual void SetUp(const benchmark::State& state) override {
    mc_ = state.range(0);
    nc_ = state.range(1);
    kc_ = state.range(2);

    Q8GEMM_XZP::SetUp(state);
  }
};

template <uint32_t MR, uint32_t NR, uint32_t NP, uint32_t KR>
class COMPUTE_ROW_SUM_Op : public Q8GEMM_XZP {
 public:
  inline COMPUTE_ROW_SUM_Op() : Q8GEMM_XZP(MR, NR, NP, KR) {}

  virtual void SetUp(const benchmark::State& state) override {
    mc_ = state.range(0);
    nc_ = state.range(1);
    kc_ = state.range(2);

    Q8GEMM_XZP::SetUp(state);
  }

  virtual void TearDown(benchmark::State& state) override {
    state.SetItemsProcessed(uint64_t(state.iterations()) * (mc() * kc()));
    a_.clear();
    k_.clear();
    b_.clear();
    c_.clear();
    aRowSums_.clear();
  }
};

#if PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP
class GEMMLOWP : public benchmark::Fixture {
 public:
  virtual void SetUp(const benchmark::State& state) override {
    const uint_fast32_t seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    auto rng =
        std::bind(std::uniform_int_distribution<uint8_t>(), std::mt19937(seed));

    mc_ = state.range(0);
    nc_ = state.range(1);
    kc_ = state.range(2);

    a_.resize(mc() * kc());
    std::generate(a_.begin(), a_.end(), std::ref(rng));
    k_.resize(nc() * kc());
    std::generate(k_.begin(), k_.end(), std::ref(rng));
    b_.resize(nc());
    std::generate(b_.begin(), b_.end(), std::ref(rng));
    c_.resize(mc() * nc());
    std::fill(c_.begin(), c_.end(), 0xA5);

    threadingContext.set_max_num_threads(1);
  }

  virtual void TearDown(benchmark::State& state) override {
    state.SetItemsProcessed(
        uint64_t(state.iterations()) * 2 * mc() * nc() * kc());
    a_.clear();
    k_.clear();
    c_.clear();
  }

  inline const uint8_t* a() const {
    return a_.data();
  }

  inline const uint8_t* k() const {
    return k_.data();
  }

  inline const int32_t* b() const {
    return b_.data();
  }

  inline uint8_t* c() {
    return c_.data();
  }

  inline uint32_t mc() const {
    return mc_;
  }

  inline uint32_t nc() const {
    return nc_;
  }

  inline uint32_t kc() const {
    return kc_;
  }

 protected:
  gemmlowp::MultiThreadGemmContext threadingContext;

 private:
  std::vector<uint8_t> a_;
  std::vector<uint8_t> k_;
  std::vector<int32_t> b_;
  std::vector<uint8_t> c_;
  uint32_t mc_;
  uint32_t nc_;
  uint32_t kc_;
};
#endif

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
  b->ArgNames({"M", "N", "K"});

  for (auto S = 15; S <= 128; S *= 2) {
    for (int K = 8; K <= 1024; K *= 2) {
      b->Args({S * S, K, K});
    }
  }
}

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
static void q8gemm_compute_row_sum(
    const uint8_t* a,
    size_t m,
    size_t k,
    size_t stride,
    const int32_t multiplier,
    int32_t* row_sum) {
  const size_t block_size = 4;
  for (size_t block_start = 0; block_start < m; block_start += block_size) {
    pytorch_q8sumrows_ukernel_4x__neon(
        a + block_start * stride,
        std::min(block_size, m - block_start),
        k,
        stride,
        multiplier,
        row_sum + block_start);
  }
}
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_ARM
BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 4x8__aarch32_neon, 4, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    pytorch_q8gemm_ukernel_4x8__aarch32_neon(
        mr(),
        nr(),
        kc(),
        a(),
        kc() * sizeof(uint8_t),
        w(),
        c(),
        mr() * sizeof(uint8_t),
        quantizationParams());
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 4x8__aarch32_neon, 4, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_ukernel_4x8__aarch32_neon(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            w() + n * (kcStride() * sizeof(uint8_t) + sizeof(int32_t)),
            c() + m * nc() + n,
            nc() * sizeof(uint8_t),
            quantizationParams());
      }
    }
  }
}
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)
    ->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)
    ->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)->Apply(GemmArguments);

BENCHMARK_TEMPLATE_F(Q8GEMM_XZP_L1, 4x8c2__aarch32_neon, 4, 8, 8, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    q8gemm_compute_row_sum(a(), mr(), kc(), kc(), -64, aRowSums());
    pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon(
        mr(),
        nr(),
        kc(),
        a(),
        kc(),
        aRowSums(),
        w(),
        c(),
        mr(),
        requantizationParams());
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon, 4, 8, 8, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    q8gemm_compute_row_sum(a(), mc(), kc(), kc(), -64, aRowSums());
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc(),
            aRowSums() + m,
            w() + n * (kcStride() + sizeof(int32_t) / sizeof(uint8_t)),
            c() + m * nc() + n,
            nc(),
            requantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon)
    ->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon)
    ->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2__aarch32_neon)->Apply(GemmArguments);
#endif

#if CPUINFO_ARCH_ARM64
BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 8x8__aarch64_neon, 8, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    pytorch_q8gemm_ukernel_8x8__aarch64_neon(
        mr(),
        nr(),
        kc(),
        a(),
        kc() * sizeof(uint8_t),
        w(),
        c(),
        mr() * sizeof(uint8_t),
        quantizationParams());
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 8x8__aarch64_neon, 8, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_ukernel_8x8__aarch64_neon(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            w() + n * (kcStride() * sizeof(uint8_t) + sizeof(int32_t)),
            c() + m * nc() + n,
            nc() * sizeof(uint8_t),
            quantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)
    ->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)
    ->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)->Apply(GemmArguments);
#endif

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 4x8__neon, 4, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    pytorch_q8gemm_ukernel_4x8__neon(
        mr(),
        nr(),
        kc(),
        a(),
        kc() * sizeof(uint8_t),
        w(),
        c(),
        mr() * sizeof(uint8_t),
        quantizationParams());
  }
}

BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 8x8__neon, 8, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    pytorch_q8gemm_ukernel_8x8__neon(
        mr(),
        nr(),
        kc(),
        a(),
        kc() * sizeof(uint8_t),
        w(),
        c(),
        mr() * sizeof(uint8_t),
        quantizationParams());
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 4x8__neon, 4, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_ukernel_4x8__neon(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            w() + n * (kcStride() * sizeof(uint8_t) + sizeof(int32_t)),
            c() + m * nc() + n,
            nc() * sizeof(uint8_t),
            quantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__neon)->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__neon)->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__neon)->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__neon)->Apply(GemmArguments);

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 8x8__neon, 8, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_ukernel_8x8__neon(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            w() + n * (kcStride() * sizeof(uint8_t) + sizeof(int32_t)),
            c() + m * nc() + n,
            nc() * sizeof(uint8_t),
            quantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__neon)->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__neon)->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__neon)->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__neon)->Apply(GemmArguments);

BENCHMARK_TEMPLATE_F(Q8GEMM_XZP_L1, 4x8c2_neon, 4, 8, 8, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    q8gemm_compute_row_sum(a(), mr(), kc(), kc(), -64, aRowSums());
    pytorch_q8gemm_xzp_ukernel_4x8c2__neon(
        mr(),
        nr(),
        kc(),
        a(),
        kc(),
        aRowSums(),
        w(),
        c(),
        mr(),
        requantizationParams());
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_XZP_Op, 4x8c2_neon, 4, 8, 8, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    q8gemm_compute_row_sum(a(), mc(), kc(), kc(), -64, aRowSums());
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_xzp_ukernel_4x8c2__neon(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc(),
            aRowSums() + m,
            w() + n * (kcStride() + sizeof(int32_t) / sizeof(uint8_t)),
            c() + m * nc() + n,
            nc(),
            requantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2_neon)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2_neon)
    ->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2_neon)
    ->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_XZP_Op, 4x8c2_neon)->Apply(GemmArguments);

BENCHMARK_TEMPLATE_DEFINE_F(
    COMPUTE_ROW_SUM_Op,
    compute_row_sum_neon,
    4,
    8,
    8,
    2)
(benchmark::State& state) {
  for (auto _ : state) {
    const size_t block_size = 4;
    for (size_t block_start = 0; block_start < mc();
         block_start += block_size) {
      pytorch_q8sumrows_ukernel_4x__neon(
          a() + block_start * kc(),
          min(block_size, mc() - block_start),
          kc(),
          kc(),
          0x11,
          aRowSums() + block_start);
    }
  }
}

BENCHMARK_REGISTER_F(COMPUTE_ROW_SUM_Op, compute_row_sum_neon)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(COMPUTE_ROW_SUM_Op, compute_row_sum_neon)
    ->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(COMPUTE_ROW_SUM_Op, compute_row_sum_neon)
    ->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(COMPUTE_ROW_SUM_Op, compute_row_sum_neon)
    ->Apply(GemmArguments);

#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 2x4c8__sse2, 2, 4, 1, 8)
(benchmark::State& state) {
  for (auto _ : state) {
    pytorch_q8gemm_ukernel_2x4c8__sse2(
        mr(),
        nr(),
        kc(),
        a(),
        kc() * sizeof(uint8_t),
        w(),
        c(),
        mr() * sizeof(uint8_t),
        quantizationParams());
  }
}

BENCHMARK_TEMPLATE_F(Q8GEMM_L1, 4x4c2__sse2, 4, 4, 4, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    pytorch_q8gemm_ukernel_4x4c2__sse2(
        mr(),
        nr(),
        kc(),
        a(),
        kc() * sizeof(uint8_t),
        w(),
        c(),
        mr() * sizeof(uint8_t),
        quantizationParams());
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 2x4c8__sse2, 2, 4, 1, 8)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_ukernel_2x4c8__sse2(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            w() + n * (kcStride() * sizeof(uint8_t) + sizeof(int32_t)),
            c() + m * nc() + n,
            nc() * sizeof(uint8_t),
            quantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_Op, 2x4c8__sse2)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 2x4c8__sse2)->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 2x4c8__sse2)->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 2x4c8__sse2)->Apply(GemmArguments);

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 4x4c2__sse2, 4, 4, 4, 2)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0; n < nc(); n += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_ukernel_4x4c2__sse2(
            mrr,
            nrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            w() + n * (kcStride() * sizeof(uint8_t) + sizeof(int32_t)),
            c() + m * nc() + n,
            nc() * sizeof(uint8_t),
            quantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x4c2__sse2)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x4c2__sse2)->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x4c2__sse2)->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x4c2__sse2)->Apply(GemmArguments);
#endif

#if PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP
BENCHMARK_DEFINE_F(GEMMLOWP, single_threaded)(benchmark::State& state) {
  for (auto _ : state) {
    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor> AM(
        a(), mc(), kc(), kc());
    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor> BM(
        k(), kc(), nc(), kc());
    gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::RowMajor> CM(
        c(), mc(), nc(), nc());
    const auto& output_pipeline =
        GemmlowpOutputPipeline::Make(b(), nc(), 127, 1, 2, 0, 255);
    gemmlowp::GemmWithOutputPipeline<
        uint8_t,
        uint8_t,
        gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
        &threadingContext, AM, BM, &CM, 2, 1, output_pipeline);
  }
}

BENCHMARK_REGISTER_F(GEMMLOWP, single_threaded)
    ->Apply(ShuffleNetV1G1GemmArguments);
BENCHMARK_REGISTER_F(GEMMLOWP, single_threaded)
    ->Apply(MobileNetV1GemmArguments);
BENCHMARK_REGISTER_F(GEMMLOWP, single_threaded)
    ->Apply(SqueezeNetV10GemmArguments);
BENCHMARK_REGISTER_F(GEMMLOWP, single_threaded)->Apply(GemmArguments);
#endif

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
