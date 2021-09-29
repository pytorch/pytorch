#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/jit_log.h>
#include <iostream>

namespace torch {
namespace jit {

TEST(JitLoggingLevelsTest, CheckSetLoggingLevel) {
  ::torch::jit::set_jit_logging_levels("file_to_test");
  ASSERT_TRUE(::torch::jit::is_enabled(
      "file_to_test.cpp", JitLoggingLevels::GRAPH_DUMP));
}

TEST(JitLoggingLevelsTest, CheckSetMultipleLogLevels) {
  ::torch::jit::set_jit_logging_levels("f1:>f2:>>f3");
  ASSERT_TRUE(::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f2.cpp", JitLoggingLevels::GRAPH_UPDATE));
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f3.cpp", JitLoggingLevels::GRAPH_DEBUG));
}

TEST(JitLoggingLevelsTest, CheckLoggingLevelAfterUnset) {
  ::torch::jit::set_jit_logging_levels("f1");
  ASSERT_EQ("f1", ::torch::jit::get_jit_logging_levels());
  ::torch::jit::set_jit_logging_levels("invalid");
  ASSERT_FALSE(
      ::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));
}

TEST(JitLoggingLevelsTest, CheckAfterChangingLevel) {
  ::torch::jit::set_jit_logging_levels("f1");
  ::torch::jit::set_jit_logging_levels(">f1");
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_UPDATE));
}

} // namespace jit
} // namespace torch
