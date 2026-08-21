#include <gtest/gtest.h>

#include <torch/headeronly/util/Load.h>

namespace torch {
namespace aot_inductor {

TEST(TestLoad, TestLoadScalar) {
  using torch::headeronly::load;

  float f = 3.5f;
  EXPECT_EQ(load<float>(&f), 3.5f);

  int64_t i = 42;
  EXPECT_EQ(load<int64_t>(&i), 42);
}

TEST(TestLoad, TestLoadBool) {
  using torch::headeronly::load;

  // NOTE: [Loading boolean values] -- loading a byte that isn't 0/1 as a
  // bool should still yield a valid (non-UB) bool value.
  unsigned char invalid_bool_bits = 0xff;
  bool loaded = load<bool>(&invalid_bool_bits);
  EXPECT_TRUE(loaded);
}

} // namespace aot_inductor
} // namespace torch
