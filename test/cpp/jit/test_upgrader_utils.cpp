#include <gtest/gtest.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>

namespace torch {
namespace jit {

TEST(UpgraderUtils, FindCorrectUpgrader) {
  std::vector<UpgraderEntry> dummy_entry = {
      {4, "foo__0_3", "foo.bar()"},
      {8, "foo__4_7", "foo.bar()"},
  };

  auto upgrader = findUpgrader(dummy_entry, 6);
  EXPECT_TRUE(upgrader.has_value());
  EXPECT_EQ(upgrader.value().upgrader_name, "foo__4_7");
}

} // namespace jit
} // namespace torch
