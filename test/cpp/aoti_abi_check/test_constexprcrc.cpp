#include <gtest/gtest.h>

#include <torch/headeronly/util/ConstexprCrc.h>

#include <string_view>
#include <unordered_set>

TEST(TestConstexprCrc, TestConstexprCrc) {
  using torch::headeronly::util::crc64;
  using torch::headeronly::util::crc64_t;

  constexpr crc64_t a = crc64("hello");
  constexpr crc64_t b = crc64(std::string_view("hello"));
  constexpr crc64_t c = crc64("world");
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_EQ(a.checksum(), b.checksum());

  std::unordered_set<crc64_t> s;
  s.insert(a);
  s.insert(c);
  EXPECT_EQ(s.size(), 2u);

  // c10 alias
  EXPECT_EQ(c10::util::crc64("hello").checksum(), a.checksum());
}
