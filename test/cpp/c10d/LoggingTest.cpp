#include <gtest/gtest.h>

#include <future>
#include <thread>

#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/LockGuard.hpp>

TEST(LockGuard, basic) {
  std::timed_mutex mutex;

  {
    C10D_LOCK_GUARD(lock, mutex);

    // already locked
    ASSERT_FALSE(mutex.try_lock());
  }

  ASSERT_TRUE(mutex.try_lock());
  mutex.unlock();
}

TEST(LockGuard, logging) {
  // set log level to INFO
  FLAGS_caffe2_log_level = 0;

  std::timed_mutex mutex;

  mutex.lock();

  auto loggingThread = std::async(std::launch::async, [&]() {
    std::unique_lock<std::timed_mutex> name{mutex, std::defer_lock};
    ::c10d::detail::lockWithLogging(
        name, std::chrono::milliseconds(10), "my lock", __FILE__, __LINE__);
  });

  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(10);
  while (true) {
    ASSERT_LT(std::chrono::system_clock::now(), deadline);

    testing::internal::CaptureStderr();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::string output = testing::internal::GetCapturedStderr();

    if (output.find("my lock: waiting for lock for 10ms") !=
        std::string::npos) {
      break;
    }
  }

  mutex.unlock();

  loggingThread.get();
}
