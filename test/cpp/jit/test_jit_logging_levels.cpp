#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/jit_log.h>
#include <sstream>

namespace torch {
namespace jit {

TEST(JitLoggingTest, CheckSetLoggingLevel) {
  ::torch::jit::set_jit_logging_levels("file_to_test");
  ASSERT_TRUE(::torch::jit::is_enabled(
      "file_to_test.cpp", JitLoggingLevels::GRAPH_DUMP));
}

TEST(JitLoggingTest, CheckSetMultipleLogLevels) {
  ::torch::jit::set_jit_logging_levels("f1:>f2:>>f3");
  ASSERT_TRUE(::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f2.cpp", JitLoggingLevels::GRAPH_UPDATE));
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f3.cpp", JitLoggingLevels::GRAPH_DEBUG));
}

TEST(JitLoggingTest, CheckLoggingLevelAfterUnset) {
  ::torch::jit::set_jit_logging_levels("f1");
  ASSERT_EQ("f1", ::torch::jit::get_jit_logging_levels());
  ::torch::jit::set_jit_logging_levels("invalid");
  ASSERT_FALSE(
      ::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));
}

TEST(JitLoggingTest, CheckAfterChangingLevel) {
  ::torch::jit::set_jit_logging_levels("f1");
  ::torch::jit::set_jit_logging_levels(">f1");
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_UPDATE));
}

TEST(JitLoggingTest, CheckOutputStreamSetting) {
  ::torch::jit::set_jit_logging_levels("test_jit_logging_levels");
  std::ostringstream test_stream;
  ::torch::jit::set_jit_logging_output_stream(test_stream);
  /* Using JIT_LOG checks if this file has logging enabled with
    is_enabled(__FILE__, level) making the test fail. since we are only testing
    the OutputStreamSetting we can forcefully output to it directly.
  */
  ::torch::jit::get_jit_logging_output_stream() << ::torch::jit::jit_log_prefix(
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP,
      __FILE__,
      __LINE__,
      ::c10::str("Message"));
  ASSERT_TRUE(test_stream.str().size() > 0);
}

} // namespace jit
} // namespace torch
