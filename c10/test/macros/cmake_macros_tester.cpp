// gtest binary that verifies the cmake_macros are configured as
// expected.

#include <c10/macros/Macros.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string_view>

namespace c10 {
namespace {

constexpr bool build_shared_libs =
#if defined(C10_BUILD_SHARED_LIBS)
    true
#else
    false
#endif
    ;

constexpr bool use_glog =
#if defined(C10_USE_GLOG)
    true
#else
    false
#endif
    ;

constexpr bool use_gflags =
#if defined(C10_USE_GFLAGS)
    true
#else
    false
#endif
    ;

constexpr bool use_numa =
#if defined(C10_USE_NUMA)
    true
#else
    false
#endif
    ;

constexpr bool use_msvc_static_runtime =
#if defined(C10_USE_MSVC_STATIC_RUNTIME)
    true
#else
    false
#endif
    ;

// Parses a bool from "1" or "0", failing otherwise.
auto parse_bool(std::string_view value) -> std::optional<bool> {
  if (value.length() == 1) {
    switch (value[0]) {
      case '1':
        return true;
      case '0':
        return false;
    }
  }
  return std::nullopt;
}

// Matcher that verifies that a preprocessor definition is equivalent
// to an expectation set by an environment variable.
MATCHER_P(matches_arg_from_environment, name, "") {
  bool is_defined = arg;

  const char* value_string = std::getenv(name);
  if (value_string == nullptr) {
    *result_listener << "You are required to set " << name
                     << " in the environment.";
    return false;
  }

  std::optional<bool> value = parse_bool(value_string);
  if (!value.has_value()) {
    *result_listener << "want 0 or 1 for environment variable; got " << name
                     << "=" << value_string;
    return false;
  }

  return is_defined == *value;
}

TEST(CMakeMacrosTest, GoldenTest) {
  EXPECT_THAT(
      build_shared_libs, matches_arg_from_environment("BUILD_SHARED_LIBS"));
  EXPECT_THAT(use_glog, matches_arg_from_environment("USE_GLOG"));
  EXPECT_THAT(use_gflags, matches_arg_from_environment("USE_GFLAGS"));
  EXPECT_THAT(use_numa, matches_arg_from_environment("USE_NUMA"));
  EXPECT_THAT(
      use_msvc_static_runtime,
      matches_arg_from_environment("USE_MSVC_STATIC_RUNTIME"));
}

} // namespace
} // namespace c10
