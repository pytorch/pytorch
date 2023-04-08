#include <c10/test/macros/cmake_macros.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace c10 {
namespace {

TEST(BazelTest, Macros) {
  EXPECT_THAT(build_shared_libs, testing::IsTrue());
  EXPECT_THAT(use_glog, testing::IsTrue());
  EXPECT_THAT(use_gflags, testing::IsTrue());
  EXPECT_THAT(use_numa, testing::IsFalse());
  EXPECT_THAT(use_msvc_static_runtime, testing::IsTrue());
}

} // namespace
} // namespace c10
