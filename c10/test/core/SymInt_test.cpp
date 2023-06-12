#include <gtest/gtest.h>

#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>

using namespace c10;
#ifndef C10_MOBILE
static void check(int64_t value) {
  const auto i = SymInt(value);
  EXPECT_EQ(i.maybe_as_int(), c10::make_optional(value));
}

TEST(SymIntTest, ConcreteInts) {
  check(INT64_MAX);
  check(0);
  check(-1);
  check(-4611686018427387904LL);
  check(INT64_MIN);
}

TEST(SymIntTest, CheckRange) {
  EXPECT_FALSE(SymInt::check_range(INT64_MIN));
}
#endif
