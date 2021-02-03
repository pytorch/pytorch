/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>
#include <memory>

#include <pack_block_sparse.h>
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <qnnpack/AlignedAllocator.h>

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

class FullyConnectedSparseOperatorTester {
 public:
  inline FullyConnectedSparseOperatorTester& inputChannels(size_t inputChannels) {
    assert(inputChannels >= 1);
    this->inputChannels_ = inputChannels;
    return *this;
  }

  inline size_t inputChannels() const {
    return this->inputChannels_;
  }

  inline FullyConnectedSparseOperatorTester& outputChannels(size_t outputChannels) {
    assert(outputChannels >= 1);
    this->outputChannels_ = outputChannels;
    return *this;
  }

  inline size_t outputChannels() const {
    return this->outputChannels_;
  }

  inline FullyConnectedSparseOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline FullyConnectedSparseOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride >= 1);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return inputChannels();
    } else {
      assert(this->inputStride_ >= inputChannels());
      return this->inputStride_;
    }
  }

  inline FullyConnectedSparseOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride >= 1);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return outputChannels();
    } else {
      assert(this->outputStride_ >= outputChannels());
      return this->outputStride_;
    }
  }

  inline FullyConnectedSparseOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline FullyConnectedSparseOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline FullyConnectedSparseOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  inline FullyConnectedSparseOperatorTester& blockSize(size_t block_size) {
    this->blockSize_ = block_size;
    return *this;
  }

  inline FullyConnectedSparseOperatorTester& sparsity(float s) {
    this->sparsity_ = s;
    return *this;
  }

  inline size_t blockSize() const {
    return this->blockSize_;
  }

  inline float sparsity() const {
    return this->sparsity_;
  }

  enum class Mode {
    Dynamic,
    Runtime,
  };

  void testQ8(const Mode mode) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    auto f32rng =
        std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    std::vector<uint8_t> input(
        (batchSize() - 1) * inputStride() + inputChannels() + 8);
    std::vector<uint8_t> kernel(outputChannels() * inputChannels());
    std::vector<int32_t> bias(outputChannels());
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + outputChannels());
    std::vector<float> output_dynamic(output.size());
    std::vector<int32_t> accumulators(batchSize() * outputChannels());
    std::vector<float> accumulators_float(batchSize() * outputChannels());

    const uint8_t* const inputPtr = input.data();
    const uint8_t inputZeroPoint = 127;
    // Make number of output channels multiple of 8.
    // This is the least common denominator for SSE/ARM kernels we have.
    size_t num_zero_points_padded = outputChannels() + 8;
    std::vector<uint8_t> kernelZeroPoints(num_zero_points_padded, 127);

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::generate(kernelZeroPoints.begin(), kernelZeroPoints.end(), std::ref(u8rng));

      uint8_t max_elem, min_elem;
      do {
        std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
        fillBlockSparseWeights(
            kernel.data(),
            outputChannels(),
            inputChannels(),
            blockSize(),
            sparsity(),
            kernelZeroPoints.data());
        max_elem = *std::max_element(kernel.cbegin(), kernel.cend());
        min_elem = *std::min_element(kernel.cbegin(), kernel.cend());
      } while (max_elem == min_elem);

      std::unique_ptr<qnnpack::BCSRMatrix> bcsr_matrix =
        qnnpack::generateBlockCSRMatrix(
            kernel.data(),
            outputChannels(),
            inputChannels(),
            blockSize(),
            kernelZeroPoints.data());

      std::fill(output.begin(), output.end(), 0xA5);
      std::fill(output_dynamic.begin(), output_dynamic.end(), 0.0f);
      std::fill(accumulators.begin(), accumulators.end(), 0);

      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oc = 0; oc < outputChannels(); oc++) {
          accumulators[i * outputChannels() + oc] = bias[oc];
        }
      }
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oc = 0; oc < outputChannels(); oc++) {
          for (size_t ic = 0; ic < inputChannels(); ic++) {
            accumulators[i * outputChannels() + oc] +=
                (int32_t(inputPtr[i * inputStride() + ic]) -
                 int32_t(inputZeroPoint)) *
                (int32_t(kernel[oc * inputChannels() + ic]) -
                 int32_t(kernelZeroPoints[oc]));
          }
        }
      }

      // Create dummy min/max for empty inputs.
      // These are only used to compute scale and zero point,
      // and real callers will just pull those values from the model.
      const int32_t accumulatorsMin = accumulators.empty()
          ? 0
          : *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulatorsMax = accumulators.empty()
          ? 900
          : *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double outputScale =
          double(uint32_t(accumulatorsMax - accumulatorsMin)) / 255.0;
      const uint8_t outputZeroPoint = uint8_t(std::max(
          std::min(
              lrint(
                  127.5 -
                  0.5 * double(accumulatorsMin + accumulatorsMax) /
                      outputScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      // 1 bcz input_scale and kernel_scale are both 1.
      std::vector<float>
        requantization_scales(num_zero_points_padded, 1.0 * 1.0 / outputScale);
      auto scale_generator = [&]() -> float {return (f32rng()/outputScale);};
      std::generate(
          requantization_scales.begin(),
          requantization_scales.end(),
          std::ref(scale_generator));

      switch(mode) {
        case Mode::Runtime:
          break;
        case Mode::Dynamic: {
            // Attention! Bias size must be a multiple of 8.
            constexpr size_t kBiasSizeMultiple = 8u;
            std::vector<float, AlignedAllocator<float, 32>> bias_float(
              (bias.size() + (kBiasSizeMultiple - 1)) & -kBiasSizeMultiple);
            std::copy(bias.cbegin(), bias.cend(), bias_float.begin());

            pytorch_qnnp_operator_t sparse_gemm = nullptr;

            ASSERT_EQ(
                pytorch_qnnp_status_success,
                pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
                    inputChannels(),
                    outputChannels(),
                    inputZeroPoint,
                    kernelZeroPoints.data(),
                    bcsr_matrix->col_indices.data(),
                    bcsr_matrix->row_values.data(),
                    bcsr_matrix->values.data(),
                    bcsr_matrix->col_block_size,
                    outputZeroPoint,
                    qmin(),
                    qmax(),
                    0,
                    requantization_scales.data(),
                    &sparse_gemm));

            ASSERT_EQ(
                pytorch_qnnp_status_success,
                pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
                    sparse_gemm,
                    batchSize(),
                    inputPtr,
                    inputStride(),
                    bias_float.data(),
                    output_dynamic.data(),
                    outputStride()));

            ASSERT_EQ(
                pytorch_qnnp_status_success,
                pytorch_qnnp_run_operator(sparse_gemm, nullptr /* thread pool */));

            ASSERT_EQ(
                pytorch_qnnp_status_success,
                pytorch_qnnp_delete_operator(sparse_gemm));
            sparse_gemm = nullptr;

            break;
          }
        default:
          // Undefined!
          ASSERT_TRUE(false);
      }

      switch (mode) {
        case Mode::Runtime:
          break;
        case Mode::Dynamic:
        {
          // Bias is added post scaling, as float.
          for (size_t i = 0; i < batchSize(); i++) {
            for (size_t oc = 0; oc < outputChannels(); oc++) {
              accumulators[i * outputChannels() + oc] -= bias[oc];
              accumulators_float[i * outputChannels() + oc] =
                (float)accumulators[i * outputChannels() + oc] *
                  requantization_scales[oc] + float(bias[oc]);
            }
          }
          for (size_t i = 0; i < batchSize(); i++) {
            for (size_t c = 0; c < outputChannels(); c++) {
              ASSERT_EQ(
                  output_dynamic[i * outputChannels() + c],
                  accumulators_float[i * outputChannels() + c])
                  << "at " << i << ", " << c
                  << ": reference = " <<
                  accumulators_float[i * outputChannels() + c]
                  << ", optimized = " << output_dynamic[i * outputChannels() + c];
            }
          }
        }
        break;

        default:
          // Undefined!
          ASSERT_TRUE(false);
      }
    }
  }

 private:
  size_t inputChannels_{1};
  size_t inputStride_{0};
  size_t outputChannels_{1};
  size_t outputStride_{0};
  size_t batchSize_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
  float sparsity_{0.7f};
  size_t blockSize_{4};
};
