#include <gtest/gtest.h>

#include <c10/util/Bitset.h>
#include <c10/util/irange.h>

using c10::utils::bitset;

TEST(BitsetTest, givenEmptyBitset_whenGettingBit_thenIsZero) {
  bitset b;
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

TEST(BitsetTest, givenEmptyBitset_whenUnsettingBit_thenIsZero) {
  bitset b;
  b.unset(4);
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

TEST(BitsetTest, givenEmptyBitset_whenSettingAndUnsettingBit_thenIsZero) {
  bitset b;
  b.set(4);
  b.unset(4);
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

TEST(BitsetTest, givenEmptyBitset_whenSettingBit_thenIsSet) {
  bitset b;
  b.set(6);
  EXPECT_TRUE(b.get(6));
}

TEST(BitsetTest, givenEmptyBitset_whenSettingBit_thenOthersStayUnset) {
  bitset b;
  b.set(6);
  for (const auto i : c10::irange(6)) {
    EXPECT_FALSE(b.get(i));
  }
  for (size_t i = 7; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

TEST(BitsetTest, givenNonemptyBitset_whenSettingBit_thenIsSet) {
  bitset b;
  b.set(6);
  b.set(30);
  EXPECT_TRUE(b.get(30));
}

TEST(BitsetTest, givenNonemptyBitset_whenSettingBit_thenOthersStayAtOldValue) {
  bitset b;
  b.set(6);
  b.set(30);
  for (const auto i : c10::irange(6)) {
    EXPECT_FALSE(b.get(i));
  }
  for (const auto i : c10::irange(7, 30)) {
    EXPECT_FALSE(b.get(i));
  }
  for (size_t i = 31; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

TEST(BitsetTest, givenNonemptyBitset_whenUnsettingBit_thenIsUnset) {
  bitset b;
  b.set(6);
  b.set(30);
  b.unset(6);
  EXPECT_FALSE(b.get(6));
}

TEST(
    BitsetTest,
    givenNonemptyBitset_whenUnsettingBit_thenOthersStayAtOldValue) {
  bitset b;
  b.set(6);
  b.set(30);
  b.unset(6);
  for (const auto i : c10::irange(30)) {
    EXPECT_FALSE(b.get(i));
  }
  EXPECT_TRUE(b.get(30));
  for (size_t i = 31; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

struct IndexCallbackMock final {
  std::vector<size_t> called_for_indices;

  void operator()(size_t index) {
    called_for_indices.push_back(index);
  }

  void expect_was_called_for_indices(std::vector<size_t> expected_indices) {
    EXPECT_EQ(expected_indices.size(), called_for_indices.size());
    for (const auto i : c10::irange(expected_indices.size())) {
      EXPECT_EQ(expected_indices[i], called_for_indices[i]);
    }
  }
};

TEST(BitsetTest, givenEmptyBitset_whenCallingForEachBit_thenDoesntCall) {
  IndexCallbackMock callback;
  bitset b;
  b.for_each_set_bit(callback);
  callback.expect_was_called_for_indices({});
}

TEST(
    BitsetTest,
    givenBitsetWithOneBitSet_whenCallingForEachBit_thenCallsForEachBit) {
  IndexCallbackMock callback;
  bitset b;
  b.set(5);
  b.for_each_set_bit(callback);
  callback.expect_was_called_for_indices({5});
}

TEST(
    BitsetTest,
    givenBitsetWithMultipleBitsSet_whenCallingForEachBit_thenCallsForEachBit) {
  IndexCallbackMock callback;
  bitset b;
  b.set(5);
  b.set(2);
  b.set(25);
  b.set(32);
  b.set(50);
  b.set(0);
  b.unset(25);
  b.set(10);
  b.for_each_set_bit(callback);
  callback.expect_was_called_for_indices({0, 2, 5, 10, 32, 50});
}
