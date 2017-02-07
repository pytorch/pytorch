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
    device_ = ::fbcollective::transport::tcp::CreateDevice("localhost");
    store_ = std::unique_ptr<::fbcollective::rendezvous::Store>(
        new ::fbcollective::rendezvous::HashStore);
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

  std::shared_ptr<::fbcollective::transport::Device> device_;
  std::unique_ptr<::fbcollective::rendezvous::Store> store_;
};

} // namespace test
} // namespace fbcollective
