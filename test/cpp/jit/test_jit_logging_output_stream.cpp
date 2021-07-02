#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/jit_log.h>
#include <iostream>

namespace torch {
namespace jit {

TEST(JitLoggingOutputStreamTest, CheckDefaultStream) {
  ASSERT_EQ(&::torch::jit::get_jit_logging_output_stream(), &std::cerr);
  ASSERT_EQ(
      ::torch::jit::get_jit_logging_output_stream().rdbuf(),
      static_cast<std::ostringstream&>(std::cerr).rdbuf());
}

TEST(JitLoggingOutputStreamTest, CheckSetLoggingOutputStream) {
  std::ostringstream stream;
  ::torch::jit::set_jit_logging_output_stream(stream);
  ASSERT_EQ(&::torch::jit::get_jit_logging_output_stream(), &stream);
  ASSERT_EQ(
      ::torch::jit::get_jit_logging_output_stream().rdbuf(), stream.rdbuf());
}

TEST(JitLoggingOutputStreamTest, CheckSetLoggingOutputStreamStdout) {
  std::ostringstream& stdout = static_cast<std::ostringstream&>(std::cout);
  ::torch::jit::set_jit_logging_output_stream(stdout);
  ASSERT_EQ(&::torch::jit::get_jit_logging_output_stream(), &stdout);
  ASSERT_EQ(
      ::torch::jit::get_jit_logging_output_stream().rdbuf(), stdout.rdbuf());
  ASSERT_EQ(&::torch::jit::get_jit_logging_output_stream(), &std::cout);
  ASSERT_EQ(
      ::torch::jit::get_jit_logging_output_stream().rdbuf(),
      static_cast<std::ostringstream&>(std::cout).rdbuf());
}

} // namespace jit
} // namespace torch
