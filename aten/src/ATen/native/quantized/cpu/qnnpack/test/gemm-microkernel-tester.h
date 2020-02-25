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

#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

class GemmMicrokernelTester {
 public:
  inline GemmMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline GemmMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline GemmMicrokernelTester& np(size_t np) {
    this->np_ = np;
    return *this;
  }

  inline size_t np() const {
    return this->np_;
  }

  inline GemmMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  inline size_t kr() const {
    return this->kr_;
  }

  inline GemmMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GemmMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GemmMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline GemmMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  inline size_t ks() const {
    return this->ks_;
  }

  inline size_t packedK() const {
    return k() % kr() == 0 ? k() : (k() / kr() + 1) * kr();
  }

  inline size_t packedN() const {
    return n() % np() == 0 ? n() : (n() / np() + 1) * np();
  }

  inline size_t biasN() const {
    return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
  }

  inline GemmMicrokernelTester& aStride(size_t aStride) {
    this->aStride_ = aStride;
    return *this;
  }

  inline size_t aStride() const {
    return this->aStride_ == 0 ? k() : this->aStride_;
  }

  inline GemmMicrokernelTester& cStride(size_t cStride) {
    this->cStride_ = cStride;
    return *this;
  }

  inline size_t cStride() const {
    return this->cStride_ == 0 ? n() : this->cStride_;
  }

  inline GemmMicrokernelTester& aZeroPoint(uint8_t aZeroPoint) {
    this->aZeroPoint_ = aZeroPoint;
    return *this;
  }

  inline uint8_t aZeroPoint() const {
    return this->aZeroPoint_;
  }

  inline GemmMicrokernelTester& bZeroPoint(uint8_t bZeroPoint) {
    this->bZeroPoint_ = bZeroPoint;
    return *this;
  }

  inline uint8_t bZeroPoint() const {
    return this->bZeroPoint_;
  }

  inline GemmMicrokernelTester& multiplier(const float multiplier) {
    this->multiplier_ = multiplier;
    return *this;
  }

  inline float multiplier() const {
    return this->multiplier_;
  }

  inline GemmMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GemmMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GemmMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_q8gemm_ukernel_function qgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<int32_t> bias(n());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedW(
        packedN() * packedK() + biasN() * sizeof(uint32_t) / sizeof(uint8_t));
    std::vector<uint8_t> c((m() - 1) * cStride() + n());
    std::vector<int32_t> acc(m() * n());
    std::vector<uint8_t> cRef(m() * n());

    const uint8_t* aPtr = a.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0xA5);

      std::fill(packedW.begin(), packedW.end(), bZeroPoint());

      pytorch_pack_q8gemm_w(
          n(),
          k(),
          nr(),
          np(),
          kr(),
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          aZeroPoint(),
          bZeroPoint(),
#endif
          b.data(),
          bias.data(),
          packedW.data());

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kIndex = 0; kIndex < k(); kIndex++) {
            ASSERT_LE(n(), packedN());
            ASSERT_LT(mIndex * n() + nIndex, acc.size());
            ASSERT_LT(mIndex * k() + kIndex, a.size());
            acc[mIndex * n() + nIndex] +=
                (int32_t(aPtr[mIndex * aStride() + kIndex]) -
                 int32_t(aZeroPoint())) *
                (int32_t(b[nIndex * k() + kIndex]) - int32_t(bZeroPoint()));
          }
          acc[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      const int32_t accMin = *std::min_element(acc.cbegin(), acc.cend());
      const int32_t accMax = *std::max_element(acc.cbegin(), acc.cend());
      if (m() * n() >= 3) {
        ASSERT_NE(accMax, accMin)
            << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }

      const double cScale = uint32_t(accMax - accMin) >= 256
          ? double(uint32_t(accMax - accMin)) / 255.0
          : 1.00001;
      const uint8_t cZeroPoint = uint8_t(std::max(
          std::min(
              lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(cScale);
      const union pytorch_qnnp_conv_quantization_params quantizationParams =
          pytorch_qnnp_compute_conv_quantization_params(
              aZeroPoint(),
              bZeroPoint(),
              requantizationScale,
              cZeroPoint,
              qmin(),
              qmax());
      const union pytorch_qnnp_q31_requantization_params
          scalarRequantizationParams =
              pytorch_qnnp_compute_scalar_requantization_params(
                  requantizationScale, cZeroPoint, qmin(), qmax());

      qgemm(
          m(),
          n(),
          k(),
          aPtr,
          aStride() * sizeof(uint8_t),
          packedW.data(),
          c.data(),
          cStride() * sizeof(uint8_t),
          &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] = pytorch_qnnp_q31_requantize(
              acc[mIndex * n() + nIndex], scalarRequantizationParams);
        }
      }

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmin()));
          ASSERT_EQ(
              uint32_t(c[mIndex * cStride() + nIndex]),
              uint32_t(cRef[mIndex * n() + nIndex]))
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << (uint32_t)cRef[mIndex * n() + nIndex]
              << " (accumulator = " << acc[mIndex * n() + nIndex]
              << "), optimized = " << (uint32_t)c[mIndex * cStride() + nIndex]
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k()
              << ", requantization scale = " << requantizationScale
              << ", output zero point = " << int32_t(cZeroPoint);
        }
      }
    }
  }

  void test(pytorch_q8gemm_dq_ukernel_function qgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<float, AlignedAllocator<float, 32>> bias(std::max<size_t>(8, n()));
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedW(
        packedN() * packedK() + biasN() * sizeof(uint32_t) / sizeof(uint8_t));
    std::vector<float> c((m() - 1) * cStride() + n());
    std::vector<float> acc(m() * n());

    const uint8_t* aPtr = a.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0.0f);

      std::fill(packedW.begin(), packedW.end(), bZeroPoint());

      pytorch_pack_q8gemm_w(
          n(),
          k(),
          nr(),
          np(),
          kr(),
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          aZeroPoint(),
          bZeroPoint(),
#endif
          b.data(),
          nullptr,
          packedW.data());

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kIndex = 0; kIndex < k(); kIndex++) {
            ASSERT_LE(n(), packedN());
            ASSERT_LT(mIndex * n() + nIndex, acc.size());
            ASSERT_LT(mIndex * k() + kIndex, a.size());
            acc[mIndex * n() + nIndex] +=
                (int32_t(aPtr[mIndex * aStride() + kIndex]) -
                 int32_t(aZeroPoint())) *
                (int32_t(b[nIndex * k() + kIndex]) - int32_t(bZeroPoint()));
          }
          acc[mIndex * n() + nIndex] = acc[mIndex * n() + nIndex] * multiplier() + bias[nIndex];
        }
      }

      const struct pytorch_qnnp_conv_dynamic_quantization_params quantizationParams{
        aZeroPoint(),
        bZeroPoint(),
        multiplier(),
      };

      qgemm(
          m(),
          n(),
          k(),
          aPtr,
          aStride() * sizeof(uint8_t),
          packedW.data(),
          bias.data(),
          c.data(),
          cStride(),
          &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_EQ(
              c[mIndex * cStride() + nIndex],
              acc[mIndex * n() + nIndex])
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << acc[mIndex * n() + nIndex]
              << ", optimized = " << c[mIndex * cStride() + nIndex]
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void test(pytorch_q8conv_ukernel_function qconv) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a((mr() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * ks() * k());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedW(
        ks() * packedN() * packedK() +
        biasN() * sizeof(uint32_t) / sizeof(uint8_t));
    std::vector<int32_t> bias(n());
    std::vector<uint8_t> c((m() - 1) * cStride() + n());
    std::vector<int32_t> acc(m() * n());
    std::vector<uint8_t> cRef(m() * n());
    std::vector<const uint8_t*> im2col(mr() * ks());

    const uint8_t* aPtr = a.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(c.begin(), c.end(), 0xA5);

      std::fill(packedW.begin(), packedW.end(), bZeroPoint());

      pytorch_pack_q8conv_w(
          n(),
          ks(),
          k(),
          np(),
          kr(),
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          aZeroPoint(),
          bZeroPoint(),
#endif
          b.data(),
          bias.data(),
          packedW.data());

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
        for (size_t mIndex = 0; mIndex < mr(); mIndex++) {
          im2col[ksIndex * mr() + mIndex] = aPtr + aStride() * mIndex;
        }
      }
      std::shuffle(im2col.begin(), im2col.end(), rng);
      for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
        for (size_t mIndex = m(); mIndex < mr(); mIndex++) {
          im2col[ksIndex * mr() + mIndex] = im2col[ksIndex * mr() + m() - 1];
        }
      }

      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
            for (size_t kBlockStart = 0; kBlockStart < k();
                 kBlockStart += kr()) {
              for (size_t kBlockOffset = 0;
                   kBlockOffset < std::min(k() - kBlockStart, kr());
                   kBlockOffset++) {
                ASSERT_LT(ksIndex * mr() + mIndex, im2col.size());
                ASSERT_LT(kBlockStart + kBlockOffset, k());
                ASSERT_LT(kBlockStart + kBlockOffset, aStride());

                acc[mIndex * n() + nIndex] +=
                    (int32_t(im2col[ksIndex * mr() + mIndex]
                                   [kBlockStart + kBlockOffset]) -
                     int32_t(aZeroPoint())) *
                    (int32_t(
                         b[(nIndex * ks() + ksIndex) * k() + kBlockStart +
                           kBlockOffset]) -
                     int32_t(bZeroPoint()));
              }
            }
          }
          acc[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      const int32_t accMin = *std::min_element(acc.cbegin(), acc.cend());
      const int32_t accMax = *std::max_element(acc.cbegin(), acc.cend());
      if (m() * n() >= 3) {
        ASSERT_NE(accMax, accMin)
            << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }

      const double cScale = uint32_t(accMax - accMin) >= 256
          ? double(uint32_t(accMax - accMin)) / 255.0
          : 1.00001;
      const uint8_t cZeroPoint = uint8_t(std::max(
          std::min(
              lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(cScale);
      const union pytorch_qnnp_conv_quantization_params quantizationParams =
          pytorch_qnnp_compute_conv_quantization_params(
              aZeroPoint(),
              bZeroPoint(),
              requantizationScale,
              cZeroPoint,
              qmin(),
              qmax());
      const union pytorch_qnnp_q31_requantization_params
          scalarRequantizationParams =
              pytorch_qnnp_compute_scalar_requantization_params(
                  requantizationScale, cZeroPoint, qmin(), qmax());

      qconv(
          m(),
          n(),
          k(),
          ks(),
          im2col.data(),
          packedW.data(),
          c.data(),
          cStride() * sizeof(uint8_t),
          &quantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] = pytorch_qnnp_q31_requantize(
              acc[mIndex * n() + nIndex], scalarRequantizationParams);
        }
      }

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmin()));
          ASSERT_EQ(
              uint32_t(c[mIndex * cStride() + nIndex]),
              uint32_t(cRef[mIndex * n() + nIndex]))
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << uint32_t(cRef[mIndex * n() + nIndex])
              << " (accumulator = " << acc[mIndex * n() + nIndex]
              << "), optimized = " << uint32_t(c[mIndex * cStride() + nIndex])
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k()
              << ", requantization scale = " << requantizationScale
              << ", output zero point = " << int32_t(cZeroPoint);
        }
      }
    }
  }

  static void q8gemm_compute_row_sum(
      const uint8_t* a,
      size_t m,
      size_t k,
      size_t stride,
      const int32_t multiplier,
      int32_t* row_sum,
      pytorch_q8sum_rows_ukernel_function q8sum_rows) {
    const size_t block_size = 4;
    for (size_t block_start = 0; block_start < m; block_start += block_size) {
      q8sum_rows(
          a + block_start * stride,
          std::min(block_size, m - block_start),
          k,
          stride,
          multiplier,
          row_sum + block_start);
    }
  }

  void test(pytorch_q8gemm_xzp_ukernel_function qgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a((m() - 1) * aStride() + k() + 8);
    std::vector<uint8_t> b(n() * k());
    std::vector<int32_t> bias(n());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedW(
        packedN() * packedK() + biasN() * sizeof(uint32_t) / sizeof(uint8_t));
    std::vector<int32_t> aRowSums(m());
    std::vector<uint8_t> c((m() - 1) * cStride() + n());
    std::vector<int32_t> acc(m() * n());
    std::vector<uint8_t> cRef(m() * n());

    const uint8_t* aPtr = a.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));

      std::fill(packedW.begin(), packedW.end(), 0);
      pytorch_pack_swizzle_q8gemm_b(
          n(),
          k(),
          np(),
          kr(),
          8,
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          aZeroPoint(),
          bZeroPoint(),
#endif
          b.data(),
          bias.data(),
          packedW.data());

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      std::fill(aRowSums.begin(), aRowSums.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        int32_t sum = 0;
        for (size_t kIndex = 0; kIndex < k(); kIndex++) {
          sum += int32_t(aPtr[mIndex * aStride() + kIndex]);
        }
        aRowSums[mIndex] = -sum * int32_t(bZeroPoint());
      }

      /* Compute 32-bit results and output quantization arguments */
      std::fill(acc.begin(), acc.end(), 0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kIndex = 0; kIndex < k(); kIndex++) {
            ASSERT_LE(n(), packedN());
            ASSERT_LT(mIndex * n() + nIndex, acc.size());
            ASSERT_LT(mIndex * k() + kIndex, a.size());
            acc[mIndex * n() + nIndex] +=
                (int32_t(aPtr[mIndex * aStride() + kIndex]) -
                 int32_t(aZeroPoint())) *
                (int32_t(b[nIndex * k() + kIndex]) - int32_t(bZeroPoint()));
          }
          acc[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      const int32_t accMin = *std::min_element(acc.cbegin(), acc.cend());
      const int32_t accMax = *std::max_element(acc.cbegin(), acc.cend());
      if (m() * n() >= 3) {
        ASSERT_NE(accMax, accMin)
            << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }

      const double cScale = uint32_t(accMax - accMin) >= 256
          ? double(uint32_t(accMax - accMin)) / 255.0
          : 1.00001;
      const uint8_t cZeroPoint = uint8_t(std::max(
          std::min(
              lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(cScale);
      const union pytorch_qnnp_q31_requantization_params requantizationParams =
          pytorch_qnnp_compute_requantization_params(
              requantizationScale, cZeroPoint, qmin(), qmax());
      const union pytorch_qnnp_q31_requantization_params
          scalarRequantizationParams =
              pytorch_qnnp_compute_scalar_requantization_params(
                  requantizationScale, cZeroPoint, qmin(), qmax());

      std::fill(c.begin(), c.end(), 0xA5);
      qgemm(
          m(),
          n(),
          k(),
          aPtr,
          aStride(),
          aRowSums.data(),
          packedW.data(),
          c.data(),
          cStride(),
          &requantizationParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] = pytorch_qnnp_q31_requantize(
              acc[mIndex * n() + nIndex], scalarRequantizationParams);
        }
      }

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(c[mIndex * cStride() + nIndex]), uint32_t(qmin()));
          ASSERT_EQ(c[mIndex * cStride() + nIndex], cRef[mIndex * n() + nIndex])
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << (uint32_t)cRef[mIndex * n() + nIndex]
              << ", optimized = " << (uint32_t)c[mIndex * cStride() + nIndex]
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
    }
  }

  void test(pytorch_hgemm_ukernel_function hgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());
    ASSERT_GE(aStride(), k());
    ASSERT_GE(cStride(), n());

    std::random_device randomDevice;
    auto rng = std::bind(
        fp16_ieee_from_fp32_value,
        std::bind(
            std::uniform_real_distribution<float>(),
            std::mt19937(randomDevice())));

    std::vector<uint16_t> a((m() - 1) * aStride() + k() + 4);
    std::vector<uint16_t> b(n() * k());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> packedW(
        packedN() * packedK() + biasN());
    std::vector<uint16_t> bias(n());
    std::vector<uint16_t> c((mr() - 1) * cStride() + nr());
    std::vector<float> cRef(m() * n());

    const uint16_t* aPtr = a.data() + 4;

    struct pytorch_qnnp_fp16_clamping_params clampingParams;
    clampingParams.scale = UINT16_C(0x3C00) /* 1.0 */;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(rng));
      std::generate(b.begin(), b.end(), std::ref(rng));
      std::generate(bias.begin(), bias.end(), std::ref(rng));
      std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);
      std::fill(cRef.begin(), cRef.end(), 0.0f);

      std::fill(packedW.begin(), packedW.end(), 0);
      pytorch_pack_hgemm_w(n(), k(), np(), kr(), b.data(), bias.data(), packedW.data());

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kBlockStart = 0; kBlockStart < k(); kBlockStart += kr()) {
            for (size_t kBlockOffset = 0;
                 kBlockOffset < std::min(k() - kBlockStart, kr());
                 kBlockOffset++) {
              ASSERT_LE(n(), packedN());
              ASSERT_LT(mIndex * n() + nIndex, cRef.size());
              ASSERT_LT(mIndex * k() + kBlockStart + kBlockOffset, a.size());

              cRef[mIndex * n() + nIndex] +=
                  fp16_ieee_to_fp32_value(
                      aPtr[mIndex * aStride() + kBlockStart + kBlockOffset]) *
                  fp16_ieee_to_fp32_value(
                      b[nIndex * k() + kBlockStart + kBlockOffset]);
            }
          }
          cRef[mIndex * n() + nIndex] += fp16_ieee_to_fp32_value(bias[nIndex]);
        }
      }

      const float accMin = *std::min_element(cRef.cbegin(), cRef.cend());
      const float accMax = *std::max_element(cRef.cbegin(), cRef.cend());
      const float cMin = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(
          accMin + (accMax - accMin) / 255.0f * float(qmin())));
      const float cMax = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(
          accMax - (accMax - accMin) / 255.0f * float(255 - qmax())));
      clampingParams.max = fp16_ieee_from_fp32_value(cMax);
      clampingParams.min = fp16_ieee_from_fp32_value(cMin);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] =
              std::max(std::min(cRef[mIndex * n() + nIndex], cMax), cMin);
        }
      }

      hgemm(
          m(),
          n(),
          k(),
          aPtr,
          aStride() * sizeof(uint16_t),
          packedW.data(),
          c.data(),
          cStride() * sizeof(uint16_t),
          &clampingParams);

      /* Validate micro-kernel outputs */
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_NEAR(
              fp16_ieee_to_fp32_value(c[mIndex * cStride() + nIndex]),
              cRef[mIndex * n() + nIndex],
              std::abs(cRef[mIndex * n() + nIndex]) * 1.0e-2f)
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << cRef[mIndex * n() + nIndex]
              << ", optimized = "
              << fp16_ieee_to_fp32_value(c[mIndex * cStride() + nIndex])
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
      /* Check that micro-kernel did not overwrite data beyond bounds */
      for (size_t mIndex = 0; mIndex < m() - 1; mIndex++) {
        for (size_t nIndex = n(); nIndex < cStride(); nIndex++) {
          ASSERT_EQ(UINT16_C(0x7E00) /* NaN */, c[mIndex * cStride() + nIndex])
              << "at " << mIndex << ", " << nIndex
              << ": Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
      for (size_t i = (m() - 1) * cStride() + n(); i < c.size(); i++) {
        ASSERT_EQ(UINT16_C(0x7E00) /* NaN */, c[i])
            << "at i = " << i << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x "
            << k();
      }
    }
  }

  void test(pytorch_sgemm_ukernel_function sgemm) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());
    ASSERT_GE(aStride(), k());
    ASSERT_GE(cStride(), n());

    std::random_device randomDevice;
    auto rng = std::bind(
        std::uniform_real_distribution<float>(), std::mt19937(randomDevice()));

    std::vector<float> a((m() - 1) * aStride() + k());
    std::vector<float> b(n() * k());
    std::vector<float> bias(n());
    std::vector<float, AlignedAllocator<float, 32>> packedW(
        packedN() * packedK() + biasN());
    std::vector<float> c((mr() - 1) * cStride() + nr());
    std::vector<float> cRef(m() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(rng));
      std::generate(b.begin(), b.end(), std::ref(rng));
      std::generate(bias.begin(), bias.end(), std::ref(rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(cRef.begin(), cRef.end(), 0.0f);

      std::fill(packedW.begin(), packedW.end(), 0.0f);
      pytorch_pack_sgemm_w(n(), k(), np(), kr(), b.data(), bias.data(), packedW.data());

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t kIndex = 0; kIndex < k(); kIndex++) {
            ASSERT_LE(n(), packedN());
            ASSERT_LT(mIndex * n() + nIndex, cRef.size());
            cRef[mIndex * n() + nIndex] +=
                a[mIndex * aStride() + kIndex] * b[nIndex * k() + kIndex];
          }
          cRef[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      const float accMin = *std::min_element(cRef.cbegin(), cRef.cend());
      const float accMax = *std::max_element(cRef.cbegin(), cRef.cend());
      const float cMin = accMin + (accMax - accMin) / 255.0f * float(qmin());
      const float cMax =
          accMax - (accMax - accMin) / 255.0f * float(255 - qmax());
      struct pytorch_qnnp_fp32_clamping_params clampingParams = {
          .max = cMax,
          .min = cMin,
      };

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] =
              std::max(std::min(cRef[mIndex * n() + nIndex], cMax), cMin);
        }
      }

      sgemm(
          m(),
          n(),
          k(),
          a.data(),
          aStride() * sizeof(float),
          packedW.data(),
          c.data(),
          cStride() * sizeof(float),
          &clampingParams);

      /* Validate micro-kernel outputs */
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_NEAR(
              c[mIndex * cStride() + nIndex],
              cRef[mIndex * n() + nIndex],
              std::abs(cRef[mIndex * n() + nIndex]) * 1.0e-6f)
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << cRef[mIndex * n() + nIndex]
              << ", optimized = " << c[mIndex * cStride() + nIndex]
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
      /* Check that micro-kernel did not overwrite data beyond bounds */
      for (size_t mIndex = 0; mIndex < m() - 1; mIndex++) {
        for (size_t nIndex = n(); nIndex < cStride(); nIndex++) {
          ASSERT_TRUE(std::isnan(c[mIndex * cStride() + nIndex]))
              << "at " << mIndex << ", " << nIndex
              << ": Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x K = " << m() << " x " << n() << " x " << k();
        }
      }
      for (size_t i = (m() - 1) * cStride() + n(); i < c.size(); i++) {
        ASSERT_TRUE(std::isnan(c[i]))
            << "at i = " << i << ", Mr x Nr x Kr = " << mr() << " x " << nr()
            << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x "
            << k();
      }
    }
  }

  void test(pytorch_sconv_ukernel_function sconv) const {
    ASSERT_LE(m(), mr());
    ASSERT_LE(n(), nr());
    ASSERT_GE(k(), kr());

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto f32rng = std::bind(
        std::uniform_real_distribution<float>(), std::mt19937(randomDevice()));

    std::vector<float> a((mr() - 1) * aStride() + k() + 8);
    std::vector<float> b(n() * ks() * k());
    std::vector<float, AlignedAllocator<float, 32>> packedW(
        ks() * packedK() * packedN() + biasN());
    std::vector<float> bias(n());
    std::vector<float> c((m() - 1) * cStride() + n());
    std::vector<float> cRef(m() * n());
    std::vector<const float*> im2col(mr() * ks());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(c.begin(), c.end(), nanf(""));
      std::fill(cRef.begin(), cRef.end(), 0.0f);

      std::fill(packedW.begin(), packedW.end(), 0.0f);
      pytorch_pack_sconv_w(
          n(), ks(), k(), np(), kr(), b.data(), bias.data(), packedW.data());

      ASSERT_NE(
          *std::max_element(a.cbegin(), a.cend()),
          *std::min_element(a.cbegin(), a.cend()));
      ASSERT_NE(
          *std::max_element(b.cbegin(), b.cend()),
          *std::min_element(b.cbegin(), b.cend()));

      for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
        for (size_t mIndex = 0; mIndex < mr(); mIndex++) {
          im2col[ksIndex * mr() + mIndex] = a.data() + aStride() * mIndex;
        }
      }
      std::shuffle(im2col.begin(), im2col.end(), rng);
      for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
        for (size_t mIndex = m(); mIndex < mr(); mIndex++) {
          im2col[ksIndex * mr() + mIndex] = im2col[ksIndex * mr() + m() - 1];
        }
      }

      std::fill(cRef.begin(), cRef.end(), 0.0);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          for (size_t ksIndex = 0; ksIndex < ks(); ksIndex++) {
            for (size_t kBlockStart = 0; kBlockStart < k();
                 kBlockStart += kr()) {
              for (size_t kBlockOffset = 0;
                   kBlockOffset < std::min(k() - kBlockStart, kr());
                   kBlockOffset++) {
                ASSERT_LT(ksIndex * mr() + mIndex, im2col.size());
                ASSERT_LT(kBlockStart + kBlockOffset, k());
                ASSERT_LT(kBlockStart + kBlockOffset, aStride());

                cRef[mIndex * n() + nIndex] +=
                    double(im2col[ksIndex * mr() + mIndex]
                                 [kBlockStart + kBlockOffset]) *
                    double(
                        b[(nIndex * ks() + ksIndex) * k() + kBlockStart +
                          kBlockOffset]);
              }
            }
          }
          cRef[mIndex * n() + nIndex] += bias[nIndex];
        }
      }

      const float accMin = *std::min_element(cRef.cbegin(), cRef.cend());
      const float accMax = *std::max_element(cRef.cbegin(), cRef.cend());
      if (m() * n() >= 3) {
        ASSERT_NE(accMax, accMin)
            << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
            << ", M x N x K = " << m() << " x " << n() << " x " << k();
      }

      const float cRefMin = accMin + float(qmin()) / 255.0f * (accMax - accMin);
      const float cRefMax =
          accMax - float(255 - qmax()) / 255.0f * (accMax - accMin);
      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          cRef[mIndex * n() + nIndex] =
              std::min(cRef[mIndex * n() + nIndex], cRefMax);
          cRef[mIndex * n() + nIndex] =
              std::max(cRef[mIndex * n() + nIndex], cRefMin);
        }
      }

      const struct pytorch_qnnp_fp32_clamping_params clampingParams {
        cRefMax, cRefMin
      };

      sconv(
          m(),
          n(),
          k(),
          ks(),
          im2col.data(),
          packedW.data(),
          c.data(),
          cStride() * sizeof(float),
          &clampingParams);

      for (size_t mIndex = 0; mIndex < m(); mIndex++) {
        for (size_t nIndex = 0; nIndex < n(); nIndex++) {
          ASSERT_LE(c[mIndex * cStride() + nIndex], cRefMax);
          ASSERT_GE(c[mIndex * cStride() + nIndex], cRefMin);
          ASSERT_NEAR(
              c[mIndex * cStride() + nIndex],
              cRef[mIndex * n() + nIndex],
              std::abs(cRef[mIndex * n() + nIndex]) * 1.0e-6f)
              << "at " << mIndex << ", " << nIndex
              << ": reference = " << cRef[mIndex * n() + nIndex]
              << ", optimized = " << c[mIndex * cStride() + nIndex]
              << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
              << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
              << " x " << ks();
        }
      }
    }
  }

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t np_{1};
  size_t kr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t aStride_{0};
  size_t cStride_{0};
  uint8_t aZeroPoint_{127};
  uint8_t bZeroPoint_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
  float multiplier_{2.0f};
};
