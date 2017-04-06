/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/test/multiproc_test.h"

#include <fcntl.h>
#include <ftw.h>

#include <string>
#include <sstream>
#include <vector>

#include "gloo/common/error.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {

static std::string createTempDirectory() {
  std::array<char, 12> dir = {"/tmp/XXXXXX"};
  return std::string(mkdtemp(dir.data()));
}

static int removeTree(const std::string& root) {
  static auto fn = [](
      const char* path, const struct stat*, int, struct FTW*) {
    return remove(path);
  };
  return nftw(root.c_str(), fn, 20, FTW_DEPTH);
}

void MultiProcTest::SetUp() {
  // Create a directory to perform the rendezvous
  storePath_ = createTempDirectory();
  ASSERT_FALSE(storePath_.empty()) << strerror(errno);
  // Create a semaphore to sync with process start
  semaphoreName_ = storePath_.substr(4);
  semaphore_ =
      sem_open(semaphoreName_.c_str(), O_CREAT | O_EXCL, S_IRUSR | S_IWUSR, 0);
  ASSERT_NE(nullptr, semaphore_) << strerror(errno);
}

void MultiProcTest::TearDown() {
  // Best effort clean up, no assert
  (void)removeTree(storePath_);
  (void)sem_close(semaphore_);
  (void)sem_unlink(semaphoreName_.c_str());
  semaphore_ = nullptr;
}

void MultiProcTest::spawnAsync(
    int size,
    std::function<void(std::shared_ptr<Context>)> fn) {
  // Start a process for each rank
  for (auto i = 0; i < size; i++) {
    const auto pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      const auto result = runWorker(size, i, fn);
      exit(result);
    } else {
      workers_.push_back(pid);
    }
  }
  // Wait for all processes to finish initializing. Set a sufficiently
  // large timeout so that we do not block forever in case one of the processes
  // fails.
  struct timespec ts;
  ASSERT_EQ(0, clock_gettime(CLOCK_REALTIME, &ts));
  ts.tv_sec += 30;
  for (auto i = 0; i < size; i++) {
    ASSERT_EQ(0, sem_timedwait(semaphore_, &ts));
  }
  workerResults_.resize(workers_.size());
}

void MultiProcTest::signalProcess(int rank, int signal) {
  ASSERT_LT(rank, workers_.size());
  const auto result = kill(workers_[rank], signal);
  ASSERT_EQ(0, result);
}

void MultiProcTest::wait()
{
  for (auto i = 0; i < workers_.size(); i++) {
    waitProcess(i);
  }
}

void MultiProcTest::waitProcess(int rank) {
  ASSERT_LT(rank, workers_.size());
  const auto& worker = workers_[rank];
  const auto& pid = waitpid(worker, &workerResults_[rank], 0);
  ASSERT_EQ(pid, worker);
}

void MultiProcTest::spawn(
    int size,
    std::function<void(std::shared_ptr<Context>)> fn) {
  spawnAsync(size, fn);
  wait();
  for (auto i = 0; i < workerResults_.size(); i++) {
    ASSERT_TRUE(WIFEXITED(workerResults_[i]));
    ASSERT_EQ(EXIT_SUCCESS, WEXITSTATUS(workerResults_[i]));
  }
}

int MultiProcTest::runWorker(
    int size,
    int rank,
    std::function<void(std::shared_ptr<Context>)> fn) {
  try {
    MultiProcWorker worker(storePath_, semaphoreName_);
    worker.run(size, rank, fn);
  } catch (const ::gloo::IoException&) {
    return kExitWithIoException;
  }
  return EXIT_SUCCESS;
}

} // namespace test
} // namespace gloo
