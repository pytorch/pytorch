// Copyright 2004-present Facebook. All Rights Reserved.

#include <c10/util/error.h>
#include <gtest/gtest.h>
#include <cstring>

using namespace ::testing;

TEST(StrErrorTest, cmp_test) {
  for (int err = 0; err <= EACCES; err++) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    ASSERT_EQ(c10::utils::str_error(err), std::string(strerror(err)));
  }
}
