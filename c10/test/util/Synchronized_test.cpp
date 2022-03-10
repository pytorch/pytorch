#include "c10/util/Synchronized.h"
#include <c10/util/Synchronized.h>
#include <gtest/gtest.h>

#include <thread>

namespace {

TEST(Synchronized, TestSingleThreadExecution) {
  c10::Synchronized<int> iv(0);
  const int kMaxValue = 100;
  for (int i = 0; i < kMaxValue; ++i) {
    auto ret = iv.withLock([](int& iv) {
      return ++iv;
    });
    EXPECT_EQ(ret, i + 1);
  }

  iv.withLock([&](int& iv) {
    EXPECT_EQ(iv, kMaxValue);
  });
}

TEST(Synchronized, TestMultiThreadedExecution) {
  c10::Synchronized<int> iv(0);
  const int kMaxValue = 10000;

  auto thread_cb = [&kMaxValue, &iv]() {
    for (int i = 0; i < kMaxValue; ++i) {
      iv.withLock([](int& iv) {
        ++iv;
      });
    }
  };

  std::thread threads[10];
  for (int i = 0; i < 10; ++i) {
    threads[i] = std::thread(thread_cb);
  }

  for (int i = 0; i < 10; ++i) {
    threads[i].join();
  }

  iv.withLock([&](int& iv) {
    EXPECT_EQ(iv, kMaxValue * 10);
  });
}

} // namespace
