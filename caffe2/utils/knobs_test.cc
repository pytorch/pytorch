#include <gtest/gtest.h>

#include "caffe2/utils/knobs.h"
#include "caffe2/utils/knob_patcher.h"

using namespace caffe2;

TEST(KnobsTest, TestKnob) {
  auto p = KnobPatcher("example_knob", false);
  EXPECT_FALSE(CheckKnobExampleKnob());
  EXPECT_FALSE(CheckKnob("example_knob"));

  p = KnobPatcher("example_knob", true);
  EXPECT_TRUE(CheckKnobExampleKnob());
  EXPECT_TRUE(CheckKnob("example_knob"));

  // Test nested patchers
  {
    auto p2 = KnobPatcher("example_knob", false);
    EXPECT_FALSE(CheckKnobExampleKnob());
    EXPECT_FALSE(CheckKnob("example_knob"));

    auto p3 = KnobPatcher("example_knob", true);
    EXPECT_TRUE(CheckKnobExampleKnob());
    EXPECT_TRUE(CheckKnob("example_knob"));
  }
  EXPECT_TRUE(CheckKnobExampleKnob());
  EXPECT_TRUE(CheckKnob("example_knob"));
}

TEST(KnobsTest, TestUnknownKnob) {
  // Unknown knob names should throw an exception
  EXPECT_THROW(CheckKnob("this_knob_does_not_exist"), std::exception);
}
