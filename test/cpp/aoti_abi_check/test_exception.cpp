#include <gtest/gtest.h>

#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <string>

namespace torch {
namespace aot_inductor {

TEST(TestExceptions, TestStdTorchCheck) {
  EXPECT_NO_THROW(STD_TORCH_CHECK(true, "dummy true message"));
  EXPECT_NO_THROW(STD_TORCH_CHECK(true, "dummy ", "true ", "message"));
  EXPECT_THROW(
      STD_TORCH_CHECK(false, "dummy false message"), std::runtime_error);
  EXPECT_THROW(
      STD_TORCH_CHECK(false, "dummy ", "false ", "message"),
      std::runtime_error);
}

TEST(TestExceptions, TestTorchErrorCodeCheck) {
  EXPECT_NO_THROW(TORCH_ERROR_CODE_CHECK(0));
  EXPECT_THROW(TORCH_ERROR_CODE_CHECK(1), std::runtime_error);
}

TEST(TestExceptions, TestThrowExceptionWithShimMessage) {
  auto message = [](const char* what) {
    try {
      torch::headeronly::detail::throw_exception(what, "f(x)", "a.cpp", 7);
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  EXPECT_EQ(
      message("out of memory"),
      "out of memory (originally from f(x) API call failed at a.cpp, line 7)");
  EXPECT_EQ(message(""), "f(x) API call failed at a.cpp, line 7");
  EXPECT_EQ(message(nullptr), "f(x) API call failed at a.cpp, line 7");
}

} // namespace aot_inductor
} // namespace torch
