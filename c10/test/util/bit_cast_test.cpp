#include <c10/util/bit_cast.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>

namespace c10 {
namespace {

TEST(bitCastTest, basic) {
  ASSERT_THAT(bit_cast<std::int8_t>('a'), testing::Eq(97));
}

} // namespace
} // namespace c10
