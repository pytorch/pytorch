#include <gtest/gtest.h>

#include <c10/util/Bitset.h>

using c10::utils::bitset;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenEmptyBitset_whenGettingBit_thenIsZero) {
  bitset b;
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenEmptyBitset_whenUnsettingBit_thenIsZero) {
  bitset b;
  b.unset(4);
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenEmptyBitset_whenSettingAndUnsettingBit_thenIsZero) {
  bitset b;
  b.set(4);
  b.unset(4);
  for (size_t i = 0; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenEmptyBitset_whenSettingBit_thenIsSet) {
  bitset b;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(6);
  EXPECT_TRUE(b.get(6));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenEmptyBitset_whenSettingBit_thenOthersStayUnset) {
  bitset b;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(6);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FALSE(b.get(i));
  }
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  for (size_t i = 7; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenNonemptyBitset_whenSettingBit_thenIsSet) {
  bitset b;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(6);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(30);
  EXPECT_TRUE(b.get(30));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenNonemptyBitset_whenSettingBit_thenOthersStayAtOldValue) {
  bitset b;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(6);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(30);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FALSE(b.get(i));
  }
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  for (size_t i = 7; i < 30; ++i) {
    EXPECT_FALSE(b.get(i));
  }
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  for (size_t i = 31; i < bitset::NUM_BITS(); ++i) {
    EXPECT_FALSE(b.get(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenNonemptyBitset_whenUnsettingBit_thenIsUnset) {
  bitset b;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(6);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(30);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.unset(6);
  EXPECT_FALSE(b.get(6));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(
    BitsetTest,
    givenNonemptyBitset_whenUnsettingBit_thenOthersStayAtOldValue) {
  bitset b;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(6);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(30);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.unset(6);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  for (size_t i = 0; i < 30; ++i) {
    EXPECT_FALSE(b.get(i));
  }
  EXPECT_TRUE(b.get(30));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
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
    for (size_t i = 0; i < expected_indices.size(); ++i) {
      EXPECT_EQ(expected_indices[i], called_for_indices[i]);
    }
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(BitsetTest, givenEmptyBitset_whenCallingForEachBit_thenDoesntCall) {
  IndexCallbackMock callback;
  bitset b;
  b.for_each_set_bit(callback);
  callback.expect_was_called_for_indices({});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(
    BitsetTest,
    givenBitsetWithOneBitSet_whenCallingForEachBit_thenCallsForEachBit) {
  IndexCallbackMock callback;
  bitset b;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(5);
  b.for_each_set_bit(callback);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  callback.expect_was_called_for_indices({5});
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(
    BitsetTest,
    givenBitsetWithMultipleBitsSet_whenCallingForEachBit_thenCallsForEachBit) {
  IndexCallbackMock callback;
  bitset b;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(5);
  b.set(2);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(25);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(32);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(50);
  b.set(0);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.unset(25);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  b.set(10);
  b.for_each_set_bit(callback);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  callback.expect_was_called_for_indices({0, 2, 5, 10, 32, 50});
}
