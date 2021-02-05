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

#include <pack_block_sparse.h>
#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/q8gemm.h>
#include <qnnpack/q8gemm_sparse.h>
#include <qnnpack/requantization.h>

#include <benchmark/benchmark.h>

namespace {
  inline uint32_t divideRoundUp(uint32_t x, uint32_t q) {
    return x / q + uint32_t(x % q != 0);
  }

  inline uint32_t roundUp(uint32_t x, uint32_t q) {
    return q * divideRoundUp(x, q);
  }

  void fillBlockSparseWeights(
      uint8_t* b,
      size_t N,
      size_t K,
      size_t block_size,
      float sparsity,
      const uint8_t* zero_points) {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    std::bernoulli_distribution dist{sparsity};
    for (uint32_t n = 0; n < N ; ++n) {
      for (uint32_t k = 0; k < K; k += block_size) {
        if (dist(rng)) {
          for (uint32_t l = 0; (l < block_size) && (k + l < K); ++l) {
            *(b + n * K + k + l) = zero_points[n];
          }
        }
      }
    }
  }

}

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
    size_t num_zero_points_kernel = (nc_ + (nr_ -1)) & -nr_;
    std::vector<uint8_t> kernel_zero_points(num_zero_points_kernel, 127);
    std::vector<float> requantization_scales(num_zero_points_kernel, 0.75f);
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
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        kernel_zero_points.data(),
#endif
        w());
    c_.resize(mc() * nc());
    std::fill(c_.begin(), c_.end(), 0xA5);

    quantizationParams_ = pytorch_qnnp_compute_conv_quantization_params(
        127, kernel_zero_points.data(),
        requantization_scales.data(), 127, 1, 254);
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

class Q8GEMMSparse : public benchmark::Fixture {
 public:
  inline Q8GEMMSparse(uint32_t mr, uint32_t nr, uint32_t kr)
      : mr_(mr), nr_(nr), kr_(kr), mc_(mr), nc_(nr), kc_(kr) {}

  virtual void SetUp(const benchmark::State&) override {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    auto f32rng =
        std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    a_.resize(mc() * kc());
    std::generate(a_.begin(), a_.end(), std::ref(u8rng));
    k_.resize(nc() * kc());
    b_.resize(nc());
    std::generate(b_.begin(), b_.end(), std::ref(f32rng));
    size_t num_zero_points_kernel = (nc_ + (nr_ -1)) & -nr_;
    std::vector<uint8_t> kernel_zero_points(num_zero_points_kernel, 127);

    std::generate(k_.begin(), k_.end(), std::ref(u8rng));
    fillBlockSparseWeights(
        k_.data(), nc(), kc(), blockSize(), sparsity(), kernel_zero_points.data());
    bcsr_matrix_ =
      qnnpack::generateBlockCSRMatrix(
          k_.data(),
          nc(),
          kc(),
          blockSize(),
          kernel_zero_points.data());
    std::vector<float> dequantization_scales(num_zero_points_kernel, 0.75f);
    c_.resize(mc() * nc());
    std::fill(c_.begin(), c_.end(), 0xA5);

    quantizationParams_ = pytorch_qnnp_conv_dynamic_quantization_params{
      127,
      kernel_zero_points.data(),
      dequantization_scales.data(),
    };
  }

  virtual void TearDown(benchmark::State& state) override {
    state.SetItemsProcessed(
        uint64_t(state.iterations()) * 2 * mc() * nc() * kc());
    a_.clear();
    k_.clear();
    b_.clear();
    c_.clear();
  }

  inline const uint8_t* a() const {
    return a_.data();
  }

  inline const uint8_t* k() const {
    return k_.data();
  }

  inline const float* b() const {
    return b_.data();
  }

  inline float* c() {
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

  inline size_t blockSize() const {
    return this->block_size_;
  }

  inline float sparsity() const {
    return this->sparsity_;
  }

  inline const pytorch_qnnp_conv_dynamic_quantization_params* quantizationParams()
      const {
    return &quantizationParams_;
  }

 protected:
  std::vector<uint8_t> a_;
  std::vector<uint8_t> k_;
  std::vector<float> b_;
  std::unique_ptr<qnnpack::BCSRMatrix> bcsr_matrix_;
  std::vector<float> c_;
  uint32_t mr_{0};
  uint32_t nr_{0};
  uint32_t kr_{0};
  uint32_t mc_{mr_};
  uint32_t nc_{nr_};
  uint32_t kc_{kr_};
  uint32_t block_size_{4};
  float sparsity_{0.7f};
  pytorch_qnnp_conv_dynamic_quantization_params quantizationParams_;
};

template <uint32_t MR, uint32_t NR, uint32_t KR>
class Q8GEMMSparse_Op : public Q8GEMMSparse {
 public:
  inline Q8GEMMSparse_Op() : Q8GEMMSparse(MR, NR, KR) {}

  virtual void SetUp(const benchmark::State& state) override {
    mc_ = state.range(0);
    nc_ = state.range(1);
    kc_ = state.range(2);

    Q8GEMMSparse::SetUp(state);
  }
};

static void SparseGEMMBenchGemmArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K"});

  b->Args({5, 4096, 640});
  b->Args({20, 4096, 640});
  b->Args({4, 4096, 1024});
  b->Args({3, 4096, 1024});
  b->Args({5, 1024, 640});
  b->Args({5, 4096, 1280});
  b->Args({20, 4096, 880});
  b->Args({10, 4096, 640});
  b->Args({10, 4096, 1280});
  b->Args({5, 4096, 1024});
  b->Args({6, 4096, 1024});
  b->Args({7, 4096, 1024});
  b->Args({8, 4096, 1024});
  b->Args({9, 4096, 1024});
  b->Args({7, 4096, 640});
  b->Args({4, 4096, 640});
  b->Args({28, 4096, 640});
  b->Args({16, 4096, 640});
  b->Args({10, 4096, 1024});
  b->Args({8, 4096, 640});
  b->Args({8, 4096, 1280});
  b->Args({7, 1024, 640});
  b->Args({7, 4096, 1280});
  b->Args({4, 1024, 640});
  b->Args({4, 4096, 1280});
  b->Args({28, 4096, 880});
  b->Args({16, 4096, 880});
  b->Args({14, 4096, 640});
  b->Args({14, 4096, 1280});
}

#if CPUINFO_ARCH_ARM
BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 4x8__aarch32_neon, 4, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
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
            channel_offset,
            quantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_Op, 4x8__aarch32_neon)
    ->Apply(SparseGEMMBenchGemmArguments);

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMMSparse_Op, 8x4c1x4__aarch32_neon, 8, 4, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4__aarch32_neon(
            mrr,
            nrr,
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            bcsr_matrix_->values.data(),
            bcsr_matrix_->row_values.data() + n,
            bcsr_matrix_->col_indices.data(),
            b() + n,
            c() + m * nc() + n,
            nc(),
            channel_offset,
            quantizationParams());
      }
    }
  }
}
BENCHMARK_REGISTER_F(Q8GEMMSparse_Op, 8x4c1x4__aarch32_neon)
    ->Apply(SparseGEMMBenchGemmArguments);

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMMSparse_Op, 8x4c1x4_prepacked__aarch32_neon, 8, 4, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto m_blocks = (mc() + mr()  - 1) / mr();
    auto k_blocks = (kc() + 4  - 1) / 4;
    std::vector<uint8_t> a_packed(m_blocks * k_blocks * mr() * 4 + 8);
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon(
            mrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            a_packed.data() + (m >> 3) * (k_blocks << 2) * mr()
            );
      }
    }
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA__aarch32_neon(
            mrr,
            nrr,
            a_packed.data() + (m >> 3) * (k_blocks << 2) * mr(),
            bcsr_matrix_->values.data(),
            bcsr_matrix_->row_values.data(),
            bcsr_matrix_->col_indices.data(),
            b() + n,
            c() + m * nc() + n,
            nc(),
            channel_offset,
            quantizationParams());
      }
    }
  }
}
BENCHMARK_REGISTER_F(Q8GEMMSparse_Op, 8x4c1x4_prepacked__aarch32_neon)
    ->Apply(SparseGEMMBenchGemmArguments);

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMMSparse_Op, 4x8c1x4_prepacked__aarch32_neon, 4, 8, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto m_blocks = (mc() + mr()  - 1) / mr();
    auto k_blocks = (kc() + 4  - 1) / 4;
    std::vector<uint8_t> a_packed(m_blocks * k_blocks * mr() * 4 + 8);
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon(
            mrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            a_packed.data() + (m >> 2) * (k_blocks << 2) * mr()
            );
      }
    }
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA__aarch32_neon(
            mrr,
            nrr,
            a_packed.data() + (m >> 2) * (k_blocks << 2) * mr(),
            bcsr_matrix_->values.data(),
            bcsr_matrix_->row_values.data(),
            bcsr_matrix_->col_indices.data(),
            b() + n,
            c() + m * nc() + n,
            nc(),
            channel_offset,
            quantizationParams());
      }
    }
  }
}
BENCHMARK_REGISTER_F(Q8GEMMSparse_Op, 4x8c1x4_prepacked__aarch32_neon)
    ->Apply(SparseGEMMBenchGemmArguments);

#endif

#if CPUINFO_ARCH_ARM64
BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMM_Op, 8x8__aarch64_neon, 8, 8, 8, 1)
(benchmark::State& state) {
  for (auto _ : state) {
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
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
            channel_offset,
            quantizationParams());
      }
    }
  }
}

BENCHMARK_REGISTER_F(Q8GEMM_Op, 8x8__aarch64_neon)
    ->Apply(SparseGEMMBenchGemmArguments);

BENCHMARK_TEMPLATE_DEFINE_F(Q8GEMMSparse_Op, 8x8c1x4_prepacked__aarch64_neon, 8, 8, 4)
(benchmark::State& state) {
  for (auto _ : state) {
    auto m_blocks = (mc() + mr()  - 1) / mr();
    auto k_blocks = (kc() + 4  - 1) / 4;
    std::vector<uint8_t> a_packed(m_blocks * k_blocks * mr() * 4 + 8);
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon(
            mrr,
            kc(),
            a() + m * kc(),
            kc() * sizeof(uint8_t),
            a_packed.data() + (m >> 3) * (k_blocks << 2) * mr()
            );
      }
    }
    for (uint32_t m = 0; m < mc(); m += mr()) {
      const uint32_t mrr = min(mc() - m, mr());
      for (uint32_t n = 0, channel_offset = 0; n < nc();
          n += nr(), channel_offset += nr()) {
        const uint32_t nrr = min(nc() - n, nr());
        pytorch_q8gemm_dq_sparse_1x4_ukernel_8x8_packedA__aarch64_neon(
            mrr,
            nrr,
            a_packed.data() + (m >> 3) * (k_blocks << 2) * mr(),
            bcsr_matrix_->values.data(),
            bcsr_matrix_->row_values.data(),
            bcsr_matrix_->col_indices.data(),
            b() + n,
            c() + m * nc() + n,
            nc(),
            channel_offset,
            quantizationParams());
      }
    }
  }
}
BENCHMARK_REGISTER_F(Q8GEMMSparse_Op, 8x8c1x4_prepacked__aarch64_neon)
    ->Apply(SparseGEMMBenchGemmArguments);

#endif

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
