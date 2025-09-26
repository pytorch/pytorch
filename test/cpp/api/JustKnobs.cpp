#include <gtest/gtest.h>

#include <torch/csrc/JustKnobs.h>

TEST(JustKnobsTest, BooleanKnobReturnsDefault) {
  // These tests apply to both internal and OSS builds because
  // 1. In internal builds, a non-existent knob should throw an exception and
  // hence use the default.
  // 2. In OSS builds, we always return the default.
  static torch::jk::BooleanKnob knob("path/to:knob_that_does_not_exist");
  ASSERT_EQ(knob(true), true);
  ASSERT_EQ(
      knob(
          false,
          // Optionally, provide a hashval and/or a switchval
          "hashval_for_consistent_randomization",
          "switchval_for_overrides"),
      false);
}

TEST(JustKnobsTest, IntegerKnobReturnsDefault) {
  static torch::jk::IntegerKnob knob("path/to:knob_that_does_not_exist");
  ASSERT_EQ(knob(42), 42);
  ASSERT_EQ(knob(-100, "switchval_for_overrides"), -100);
}
