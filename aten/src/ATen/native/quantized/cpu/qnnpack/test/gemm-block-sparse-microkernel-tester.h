/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <fp16.h>

#include <pack_block_sparse.h>
#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

#define MAYBE_UNUSED __attribute__((unused))

namespace {
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

  // Temp Debug utils that will be removed later
  MAYBE_UNUSED void printMatrix(const char* name, const uint8_t* a, const size_t M, const size_t N) {
    std::cout << "Matrix START:" << name << "...\n";
    for (uint32_t m = 0; m < M ; ++m) {
      for (uint32_t n = 0; n < N; n++) {
        std::cout << (const uint32_t)(*(a + m * N + n)) << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << "Matrix END...\n\n";
  }

  MAYBE_UNUSED void printMatrix(const char* name, const float* a, const size_t M, const size_t N) {
    std::cout << "Matrix START:" << name << "...\n";
    for (uint32_t m = 0; m < M ; ++m) {
      for (uint32_t n = 0; n < N; n++) {
        std::cout << (*(a + m * N + n)) << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << "Matrix END...\n\n";
  }

}

class GemmBlockSparseMicrokernelTester {
 public:
  inline GemmBlockSparseMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline GemmBlockSparseMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline GemmBlockSparseMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GemmBlockSparseMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GemmBlockSparseMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline GemmBlockSparseMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  inline GemmBlockSparseMicrokernelTester& blockSize(size_t block_size) {
    this->blockSize_ = block_size;
    return *this;
  }

  inline GemmBlockSparseMicrokernelTester& sparsity(float s) {
    this->sparsity_ = s;
    return *this;
  }

  inline size_t ks() const {
    return this->ks_;
  }

  inline size_t blockSize() const {
    return this->blockSize_;
  }

  inline float sparsity() const {
    return this->sparsity_;
  }

  inline size_t biasN() const {
    return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
  }

  inline GemmBlockSparseMicrokernelTester& aStride(size_t aStride) {
    this->aStride_ = aStride;
    return *this;
  }

  inline size_t aStride() const {
    return this->aStride_ == 0 ? k() : this->aStride_;
  }

  inline GemmBlockSparseMicrokernelTester& cStride(size_t cStride) {
    this->cStride_ = cStride;
    return *this;
  }

  inline size_t cStride() const {
    return this->cStride_ == 0 ? n() : this->cStride_;
  }

  inline GemmBlockSparseMicrokernelTester& aZeroPoint(uint8_t aZeroPoint) {
    this->aZeroPoint_ = aZeroPoint;
    return *this;
  }

  inline uint8_t aZeroPoint() const {
    return this->aZeroPoint_;
  }

  inline GemmBlockSparseMicrokernelTester& bZeroPoint(uint8_t bZeroPoint) {
    this->bZeroPoint_ = bZeroPoint;
    return *this;
  }

  inline uint8_t bZeroPoint() const {
    return this->bZeroPoint_;
  }

  inline GemmBlockSparseMicrokernelTester& multiplier(const float multiplier) {
    this->multiplier_ = multiplier;
    return *this;
  }

  inline float multiplier() const {
    return this->multiplier_;
  }

  inline GemmBlockSparseMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GemmBlockSparseMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GemmBlockSparseMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_q8gemm_dq_sparse_ukernel_function qgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<float, AlignedAllocator<float, 32>> bias(std::max<size_t>(8, n()));
    std::vector<float> c((m() - 1) * cStride() + n());
    std::vector<float> acc(m() * n());

    const uint8_t* aPtr = a.data();

    for (size_t iteration = 0; iteration < 1; iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0.0f);
      size_t num_zero_points_padded = n() + 8;
      std::vector<uint8_t> kernel_zero_points
        (num_zero_points_padded, bZeroPoint());
      std::generate(kernel_zero_points.begin(), kernel_zero_points.end(), std::ref(u8rng));

      // This loop to ensure the assert_ne on b mat does not fire.
      uint8_t max_elem, min_elem;
      do {
        std::generate(b.begin(), b.end(), std::ref(u8rng));
        fillBlockSparseWeights(
            b.data(), n(), k(), blockSize(), sparsity(), kernel_zero_points.data());
        max_elem = *std::max_element(b.cbegin(), b.cend());
        min_elem = *std::min_element(b.cbegin(), b.cend());
      } while (max_elem == min_elem);

      std::unique_ptr<qnnpack::BCSRMatrix> bcsr_matrix =
        qnnpack::generateBlockCSRMatrix(
            b.data(), n(), k(), blockSize(), kernel_zero_points.data());

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      auto f32rng =
          std::bind(std::uniform_real_distribution<float>(1, 5), rng);
      std::vector<float> dequantization_scales(num_zero_points_padded);
      std::generate(
          dequantization_scales.begin(),
          dequantization_scales.end(),
          std::ref(f32rng));
      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kIndex = 0; kIndex < k(); kIndex++) {
            ASSERT_LT(mIndex * n() + nIndex, acc.size());
            ASSERT_LT(mIndex * k() + kIndex, a.size());
            acc[mIndex * n() + nIndex] +=
                (int32_t(aPtr[mIndex * aStride() + kIndex]) -
                 int32_t(aZeroPoint())) *
                (int32_t(b[nIndex * k() + kIndex]) - int32_t(kernel_zero_points[nIndex]));
          }
          acc[mIndex * n() + nIndex] =
            acc[mIndex * n() + nIndex] *
            dequantization_scales[nIndex] +
            bias[nIndex];
        }
      }

      const struct pytorch_qnnp_conv_dynamic_quantization_params quantizationParams{
        aZeroPoint(),
        kernel_zero_points.data(),
        dequantization_scales.data(),
      };

      qgemm(
          m(),
          n(),
          aPtr,
          aStride() * sizeof(uint8_t),
          bcsr_matrix->values.data(),
          bcsr_matrix->row_values.data(),
          bcsr_matrix->col_indices.data(),
          bias.data(),
          c.data(),
          cStride(),
          0,
          &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_EQ(
              c[mIndex * cStride() + nIndex],
              acc[mIndex * n() + nIndex])
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << acc[mIndex * n() + nIndex]
              << ", optimized = " << c[mIndex * cStride() + nIndex]
              << ", Mr x Nr = " << mr() << " x " << nr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t aStride_{0};
  size_t cStride_{0};
  size_t blockSize_{4};
  uint8_t aZeroPoint_{127};
  uint8_t bZeroPoint_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
  float multiplier_{2.0f};
  float sparsity_{0.7f};
};
