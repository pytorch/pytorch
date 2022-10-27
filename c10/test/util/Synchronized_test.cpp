#include <c10/util/Synchronized.h>
#include <c10/util/irange.h>
#include <gtest/gtest.h>

#include <array>
#include <thread>

namespace {

TEST(Synchronized, TestSingleThreadExecution) {
  c10::Synchronized<int> iv(0);
  const int kMaxValue = 100;
  for (const auto i : c10::irange(kMaxValue)) {
    auto ret = iv.withLock([](int& iv) { return ++iv; });
    EXPECT_EQ(ret, i + 1);
  }

  iv.withLock([kMaxValue](int& iv) { EXPECT_EQ(iv, kMaxValue); });
}

TEST(Synchronized, TestMultiThreadedExecution) {
  c10::Synchronized<int> iv(0);
#define NUM_LOOP_INCREMENTS 10000

  auto thread_cb = [&iv]() {
    for (const auto i : c10::irange(NUM_LOOP_INCREMENTS)) {
      iv.withLock([](int& iv) { ++iv; });
    }
  };

  std::array<std::thread, 10> threads;
  for (auto& t : threads) {
    t = std::thread(thread_cb);
  }

  for (auto& t : threads) {
    t.join();
  }

  iv.withLock([](int& iv) { EXPECT_EQ(iv, NUM_LOOP_INCREMENTS * 10); });
#undef NUM_LOOP_INCREMENTS
}

} // namespace
