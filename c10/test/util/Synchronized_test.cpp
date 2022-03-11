#include <c10/util/Synchronized.h>
#include <gtest/gtest.h>

#include <thread>

namespace {

TEST(Synchronized, TestSingleThreadExecution) {
  c10::Synchronized<int> iv(0);
  const int kMaxValue = 100;
  for (int i = 0; i < kMaxValue; ++i) {
    auto ret = iv.withLock([](int& iv) { return ++iv; });
    EXPECT_EQ(ret, i + 1);
  }

  iv.withLock([kMaxValue](int& iv) { EXPECT_EQ(iv, kMaxValue); });
}

TEST(Synchronized, TestMultiThreadedExecution) {
  c10::Synchronized<int> iv(0);
  const int kMaxValue = 10000;

  auto thread_cb = [&iv, kMaxValue]() {
    for (int i = 0; i < kMaxValue; ++i) {
      iv.withLock([](int& iv) { ++iv; });
    }
  };

  std::array<std::thread, 10> threads;
  for (int i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread(thread_cb);
  }

  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  iv.withLock([kMaxValue](int& iv) { EXPECT_EQ(iv, kMaxValue * 10); });
}

} // namespace
