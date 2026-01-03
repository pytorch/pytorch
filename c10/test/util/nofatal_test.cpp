#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

namespace {
template <typename T>
inline void expectThrowsEq(T&& fn, const char* expected_msg) {
  try {
    std::forward<T>(fn)();
  } catch (const c10::Error& e) {
    EXPECT_TRUE(
        std::string(e.what_without_backtrace()).find(expected_msg) !=
        std::string::npos);
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \"" << expected_msg
                << "\" but didn't throw";
}
} // namespace

TEST(NofatalTest, TorchCheckComparisons) {
  // quick make sure that no-op works as expected
  TORCH_CHECK_EQ(1, 1) << "i am a silly message " << 1;
  expectThrowsEq(
      []() { TORCH_CHECK_EQ(1, 2) << "i am a silly message " << 1; },
      "Check failed: 1 == 2 (1 vs. 2). i am a silly message 1");
  expectThrowsEq(
      []() { TORCH_CHECK_NE(2, 2); }, "Check failed: 2 != 2 (2 vs. 2).");
  expectThrowsEq(
      []() { TORCH_CHECK_LT(2, 2); }, "Check failed: 2 < 2 (2 vs. 2).");
  expectThrowsEq(
      []() { TORCH_CHECK_LE(3, 2); }, "Check failed: 3 <= 2 (3 vs. 2).");
  expectThrowsEq(
      []() { TORCH_CHECK_GT(2, 2); }, "Check failed: 2 > 2 (2 vs. 2).");
  expectThrowsEq(
      []() { TORCH_CHECK_GE(2, 3); }, "Check failed: 2 >= 3 (2 vs. 3).");
  expectThrowsEq(
      []() {
        void* p = nullptr;
        TORCH_CHECK_NOTNULL(p);
      },
      "Check failed: 'p' must be non NULL.");

#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  // if dbg build, DCHECK should result in deth
  EXPECT_DEATH(TORCH_DCHECK_EQ(1, 2), "Check failed");
#else
  TORCH_DCHECK_EQ(1, 2); // no-op
#endif
#endif // GTEST_HAS_DEATH_TEST
}
