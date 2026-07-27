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
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

class ChannelShuffleOperatorTester {
 public:
  inline ChannelShuffleOperatorTester& groups(size_t groups) {
    assert(groups != 0);
    this->groups_ = groups;
    return *this;
  }

  inline size_t groups() const {
    return this->groups_;
  }

  inline ChannelShuffleOperatorTester& groupChannels(size_t groupChannels) {
    assert(groupChannels != 0);
    this->groupChannels_ = groupChannels;
    return *this;
  }

  inline size_t groupChannels() const {
    return this->groupChannels_;
  }

  inline size_t channels() const {
    return groups() * groupChannels();
  }

  inline ChannelShuffleOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return channels();
    } else {
      assert(this->inputStride_ >= channels());
      return this->inputStride_;
    }
  }

  inline ChannelShuffleOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return channels();
    } else {
      assert(this->outputStride_ >= channels());
      return this->outputStride_;
    }
  }

  inline ChannelShuffleOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline ChannelShuffleOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testX8() const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Create, setup, run, and destroy Channel Shuffle operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t channel_shuffle_op = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_channel_shuffle_nc_x8(
              groups(), groupChannels(), 0, &channel_shuffle_op));
      ASSERT_NE(nullptr, channel_shuffle_op);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_channel_shuffle_nc_x8(
              channel_shuffle_op,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(
              channel_shuffle_op, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(channel_shuffle_op));
      channel_shuffle_op = nullptr;

      /* Verify results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t g = 0; g < groups(); g++) {
          for (size_t c = 0; c < groupChannels(); c++) {
            ASSERT_EQ(
                uint32_t(input[i * inputStride() + g * groupChannels() + c]),
                uint32_t(output[i * outputStride() + c * groups() + g]));
          }
        }
      }
    }
  }

 private:
  size_t groups_{1};
  size_t groupChannels_{1};
  size_t batchSize_{1};
  size_t inputStride_{0};
  size_t outputStride_{0};
  size_t iterations_{15};
};
