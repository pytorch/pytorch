#include <atomic>
#include <thread>
#include <vector>

#include <c10/util/Lazy.h>
#include <gtest/gtest.h>

namespace c10_test {

// Long enough not to fit in typical SSO.
const std::string kLongString = "I am a long enough string";

TEST(LazyTest, OptimisticLazy) {
  std::atomic<size_t> invocations = 0;
  auto factory = [&] {
    ++invocations;
    return kLongString;
  };

  c10::OptimisticLazy<std::string> s;

  constexpr size_t kNumThreads = 16;
  std::vector<std::thread> threads;
  std::atomic<std::string*> address = nullptr;

  for (size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&] {
      auto* p = &s.ensure(factory);
      auto old = address.exchange(p);
      if (old != nullptr) {
        // Even racing ensure()s should return a stable reference.
        EXPECT_EQ(old, p);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_GE(invocations.load(), 1);
  EXPECT_EQ(*address.load(), kLongString);

  invocations = 0;
  s.reset();
  s.ensure(factory);
  EXPECT_EQ(invocations.load(), 1);

  invocations = 0;

  auto sCopy = s;
  EXPECT_EQ(sCopy.ensure(factory), kLongString);
  EXPECT_EQ(invocations.load(), 0);

  auto sMove = std::move(s);
  EXPECT_EQ(sMove.ensure(factory), kLongString);
  EXPECT_EQ(invocations.load(), 0);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(s.ensure(factory), kLongString);
  EXPECT_EQ(invocations.load(), 1);

  invocations = 0;

  s = sCopy;
  EXPECT_EQ(s.ensure(factory), kLongString);
  EXPECT_EQ(invocations.load(), 0);

  s = std::move(sCopy);
  EXPECT_EQ(s.ensure(factory), kLongString);
  EXPECT_EQ(invocations.load(), 0);
}

TEST(LazyTest, PrecomputedLazyValue) {
  static const std::string kLongString = "I am a string";
  EXPECT_EQ(
      std::make_shared<c10::PrecomputedLazyValue<std::string>>(kLongString)
          ->get(),
      kLongString);
}

TEST(LazyTest, OptimisticLazyValue) {
  static const std::string kLongString = "I am a string";

  class LazyString : public c10::OptimisticLazyValue<std::string> {
    std::string compute() const override {
      return kLongString;
    }
  };

  auto ls = std::make_shared<LazyString>();
  EXPECT_EQ(ls->get(), kLongString);

  // Returned reference should be stable.
  EXPECT_EQ(&ls->get(), &ls->get());
}

} // namespace c10_test
