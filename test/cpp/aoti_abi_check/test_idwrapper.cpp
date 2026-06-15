#include <gtest/gtest.h>

#include <torch/headeronly/util/IdWrapper.h>

#include <cstdint>
#include <unordered_set>

namespace {
struct MyId : torch::headeronly::IdWrapper<MyId, uint32_t> {
  constexpr explicit MyId(uint32_t id) : IdWrapper(id) {}
  constexpr uint32_t value() const {
    return underlyingId();
  }
};
} // namespace

C10_DEFINE_HASH_FOR_IDWRAPPER(MyId)

TEST(TestIdWrapper, TestIdWrapper) {
  MyId a(1), b(1), c(2);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_EQ(a.value(), 1u);

  std::unordered_set<MyId> s;
  s.insert(a);
  s.insert(b);
  s.insert(c);
  EXPECT_EQ(s.size(), 2u);
}
