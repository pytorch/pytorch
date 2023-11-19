#include <gtest/gtest.h>
#include <string>

#include <c10/util/int128.h>
#include <torch/csrc/lazy/core/hash.h>

namespace torch {
namespace lazy {

template <typename T>
void test_hash_repeatable_sensitive(const T& example_a, const T& example_b) {
  // repeatable
  EXPECT_EQ(Hash(example_a), Hash(example_a));
  EXPECT_EQ(MHash(example_a), MHash(example_a));
  EXPECT_EQ(MHash(example_a, example_a), MHash(example_a, example_a));

  // sensitive
  EXPECT_NE(Hash(example_a), Hash(example_b));
  EXPECT_NE(MHash(example_a), MHash(example_b));
  EXPECT_NE(MHash(example_a, example_a), MHash(example_a, example_b));
}

TEST(HashTest, Scalar) {
  GTEST_SKIP()
      << "Broken test. See https://github.com/pytorch/pytorch/issues/99883";
  c10::Scalar a(0);
  c10::Scalar b(0);

  // simulate some garbage in the unused bits of the
  // the tagged union that is c10::Scalar, which is bigger
  // than the size of the int64_t we're currently using it with
  *((uint8_t*)&b) = 1;
  // actual 'value' of the Scalar as a 64 bit int shouldn't have changed
  EXPECT_EQ(a.toLong(), b.toLong());
  // and hash should ignore this garbage
  EXPECT_EQ(Hash(a), Hash(b));
  EXPECT_EQ(MHash(a), MHash(b));
  EXPECT_EQ(MHash(a, a), MHash(a, b));
}

TEST(HashTest, Sanity) {
  // String
  test_hash_repeatable_sensitive(
      std::string(
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut at suscipit purus."),
      std::string(
          "Lorem Jpsum dolor sit amet, consectetur adipiscing elit. Ut at suscipit purus."));

  // Number types
  test_hash_repeatable_sensitive(true, false);
  test_hash_repeatable_sensitive((int8_t)0xfa, (int8_t)0xfb);
  test_hash_repeatable_sensitive((int16_t)0xface, (int16_t)0xfade);
  test_hash_repeatable_sensitive((int32_t)0xfaceb000, (int32_t)0xfadeb000);
  test_hash_repeatable_sensitive((int64_t)0x1faceb000, (int64_t)0x1fadeb000);
  test_hash_repeatable_sensitive((uint8_t)0xfa, (uint8_t)0xfb);
  test_hash_repeatable_sensitive((uint16_t)0xface, (uint16_t)0xfade);
  test_hash_repeatable_sensitive((uint32_t)0xfaceb000, (uint32_t)0xfadeb000);
  test_hash_repeatable_sensitive((uint64_t)0x1faceb000, (uint64_t)0x1fadeb000);

  // c10 types
  test_hash_repeatable_sensitive(c10::ScalarType::Bool, c10::ScalarType::Byte);
  test_hash_repeatable_sensitive(c10::Scalar(1.334), c10::Scalar(1.335));
  test_hash_repeatable_sensitive(c10::Scalar(true), c10::Scalar(false));
  test_hash_repeatable_sensitive(c10::Scalar(12345), c10::Scalar(12354));

  // c10::optional
  test_hash_repeatable_sensitive(
      c10::optional<std::string>("I have value!"),
      c10::optional<std::string>(c10::nullopt));

  // Containers
  auto a = std::vector<int32_t>({0, 1, 1, 2, 3, 5, 8});
  auto b = std::vector<int32_t>({1, 1, 2, 3, 5, 8, 12});
  test_hash_repeatable_sensitive(a, b);
  test_hash_repeatable_sensitive(
      c10::ArrayRef<int32_t>(a), c10::ArrayRef<int32_t>(b));

  // vector<bool> is a special case bc it is implemented as vector<bit>
  auto bool_a = std::vector<bool>({true, false, false, true});
  auto bool_b = std::vector<bool>({true, true, false, true});
  test_hash_repeatable_sensitive(bool_a, bool_b);
}

} // namespace lazy
} // namespace torch
