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
#include <semaphore.h>
#include <signal.h>

#include <chrono>
#include <string>

#include "gloo/common/logging.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/transport/tcp/device.h"

namespace gloo {
namespace test {

const int kExitWithIoException = 10;

class MultiProcTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  void spawnAsync(int size, std::function<void(std::shared_ptr<Context>)> fn);
  void wait();
  void waitProcess(int rank);
  void signalProcess(int rank, int signal);

  int getResult(int rank) {
    return workerResults_[rank];
  }

  void spawn(int size, std::function<void(std::shared_ptr<Context>)> fn) ;

 private:
  int runWorker(
      int size,
      int rank,
      std::function<void(std::shared_ptr<Context>)> fn);

  std::string storePath_;
  std::string semaphoreName_;
  sem_t* semaphore_;
  std::vector<pid_t> workers_;
  std::vector<int> workerResults_;

};

class MultiProcWorker {
 public:
  explicit MultiProcWorker(
      const std::string& storePath,
      const std::string& semaphoreName) {
    device_ = ::gloo::transport::tcp::CreateDevice("localhost");
    store_ = std::unique_ptr<::gloo::rendezvous::Store>(
        new ::gloo::rendezvous::FileStore(storePath));
    semaphore_ = sem_open(semaphoreName.c_str(), 0);
    GLOO_ENFORCE_NE(semaphore_, (sem_t*)nullptr, strerror(errno));
  }
  ~MultiProcWorker() {
    sem_close(semaphore_);
  }

  void run(
      int size,
      int rank,
      std::function<void(std::shared_ptr<Context>)> fn) {
    auto context =
      std::make_shared<::gloo::rendezvous::Context>(rank, size);
    device_->setTimeout(std::chrono::milliseconds(300));
    context->connectFullMesh(*store_, device_);
    sem_post(semaphore_);
    fn(std::move(context));
  }

 protected:
  std::shared_ptr<::gloo::transport::Device> device_;
  std::unique_ptr<::gloo::rendezvous::Store> store_;
  sem_t* semaphore_;
};

} // namespace test
} // namespace gloo
