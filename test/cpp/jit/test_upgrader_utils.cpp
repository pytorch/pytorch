#include <gtest/gtest.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>

#include <test/cpp/jit/test_utils.h>

namespace torch {
namespace jit {

TEST(UpgraderUtils, FindCorrectUpgrader) {
  std::vector<UpgraderEntry> dummy_entry = {
      {4, "foo__0_3", "foo.bar()"},
      {8, "foo__4_7", "foo.bar()"},
  };

  auto upgrader_at_6 = findUpgrader(dummy_entry, 6);
  EXPECT_TRUE(upgrader_at_6.has_value());
  EXPECT_EQ(upgrader_at_6.value().upgrader_name, "foo__4_7");

  auto upgrader_at_1 = findUpgrader(dummy_entry, 1);
  EXPECT_TRUE(upgrader_at_1.has_value());
  EXPECT_EQ(upgrader_at_1.value().upgrader_name, "foo__0_3");

  auto upgrader_at_10 = findUpgrader(dummy_entry, 10);
  EXPECT_TRUE(upgrader_at_1.has_value());
  EXPECT_EQ(upgrader_at_1.value().upgrader_name, "foo__0_3");
}

TEST(UpgraderUtils, FindIfOpIsCurrent) {
  std::vector<UpgraderEntry> dummy_entry = {
      {4, "foo__0_3", "foo.bar()"},
      {8, "foo__4_7", "foo.bar()"},
  };

  auto isCurrent = isOpCurrentBasedOnUpgraderEntries(dummy_entry, 6);
  auto isCurrentV2 = isOpCurrentBasedOnUpgraderEntries(dummy_entry, 8);
  EXPECT_FALSE(isCurrent);
  EXPECT_TRUE(isCurrentV2);

  // symbol based look up
  test_only_add_entry("foo", dummy_entry[0]);
  test_only_add_entry("foo", dummy_entry[1]);
  EXPECT_FALSE(isOpSymbolCurrent("foo", 6));
  EXPECT_TRUE(isOpSymbolCurrent("foo", 8));
  test_only_remove_entry("foo");
}

} // namespace jit
} // namespace torch
