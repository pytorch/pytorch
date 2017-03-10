/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <thread>
#include <vector>

#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/hash_store.h"
#include "gloo/transport/tcp/device.h"

namespace gloo {
namespace test {

class BaseTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    device_ = ::gloo::transport::tcp::CreateDevice("localhost");
    store_ = std::unique_ptr<::gloo::rendezvous::Store>(
        new ::gloo::rendezvous::HashStore);
  }

  void spawnThreads(int size, std::function<void(int)> fn) {
    std::vector<std::thread> threads;
    std::vector<std::string> errors;
    for (int rank = 0; rank < size; rank++) {
      threads.push_back(std::thread([&, rank]() {
        try {
          fn(rank);
        } catch (const std::exception& ex) {
          errors.push_back(ex.what());
        }
      }));
    }

    // Wait for threads to complete
    for (auto& thread : threads) {
      thread.join();
    }

    // Re-throw first exception if there is one
    if (errors.size() > 0) {
      throw(std::runtime_error(errors[0]));
    }
  }

  void spawn(
      int size,
      std::function<void(std::shared_ptr<Context>)> fn) {
    spawnThreads(size, [&](int rank) {
        auto context =
          std::make_shared<::gloo::rendezvous::Context>(rank, size);
        context->connectFullMesh(*store_, device_);
        fn(std::move(context));
      });
  }

  std::shared_ptr<::gloo::transport::Device> device_;
  std::unique_ptr<::gloo::rendezvous::Store> store_;
};

class Fixture {
 public:
  Fixture(const std::shared_ptr<Context> context, int ptrs, int count)
      : context(context),
        count(count) {
    for (int i = 0; i < ptrs; i++) {
      std::unique_ptr<float[]> ptr(new float[count]);
      srcs.push_back(std::move(ptr));
    }
  }

  Fixture(Fixture&& other) noexcept
    : context(other.context),
      count(other.count) {
    srcs = std::move(other.srcs);
  }

  void assignValues() {
    const auto stride = context->size * srcs.size();
    for (auto i = 0; i < srcs.size(); i++) {
      auto val = (context->rank * srcs.size()) + i;
      for (auto j = 0; j < count; j++) {
        srcs[i][j] = (j * stride) + val;
      }
    }
  }

  std::vector<float*> getFloatPointers() const {
    std::vector<float*> out;
    for (const auto& src : srcs) {
      out.push_back(src.get());
    }
    return out;
  }

  std::shared_ptr<Context> context;
  const int count;
  std::vector<std::unique_ptr<float[]> > srcs;
};

} // namespace test
} // namespace gloo
