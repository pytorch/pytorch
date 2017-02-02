#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <thread>
#include <vector>

#include "fbcollective/rendezvous/hash_store.h"
#include "fbcollective/transport/tcp/device.h"

namespace fbcollective {
namespace test {

class BaseTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    ::fbcollective::transport::tcp::attr attr = {
        .hostname = "localhost",
    };

    device_ = ::fbcollective::transport::tcp::CreateDevice(attr);
    store_ = std::unique_ptr<::fbcollective::rendezvous::Store>(
        new ::fbcollective::rendezvous::HashStore);
  }

  void spawnThreads(int size, std::function<void(int)> fn) {
    std::vector<std::thread> threads;
    for (int rank = 0; rank < size; rank++) {
      threads.push_back(std::thread(std::bind(fn, rank)));
    }

    // Wait for threads to complete
    for (auto& thread : threads) {
      thread.join();
    }
  }

  std::shared_ptr<::fbcollective::transport::Device> device_;
  std::unique_ptr<::fbcollective::rendezvous::Store> store_;
};

} // namespace test
} // namespace fbcollective
