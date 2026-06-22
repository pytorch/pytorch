#include <gtest/gtest.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>

#include <test/cpp/jit/test_utils.h>

#include <vector>

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

TEST(UpgraderUtils, IsVersionMapSorted) {
  auto map = get_operator_version_map();
  // tests if the each list of UpgraderEntry in the map is sorted by
  // their bumped_at_version field.
  for (const auto& entry : map) {
    std::vector<int> versions;
    for (const auto& el : entry.second) {
      versions.push_back(el.bumped_at_version);
    }
    EXPECT_TRUE(std::is_sorted(versions.begin(), versions.end()));
  }
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

TEST(UpgraderUtils, CanLoadHistoricOp) {
  std::vector<UpgraderEntry> dummy_entry = {
      {4, "foo__0_3", "foo.bar()"},
      {8, "foo__4_7", "foo.foo()"},
  };

  std::vector<std::string> schemas = {"foo.bar()", "foo.foo()"};

  // symbol based look up
  test_only_add_entry("old_op_not_exist.first", dummy_entry[0]);
  test_only_add_entry("old_op_not_exist.second", dummy_entry[1]);

  auto oldSchemas = loadPossibleHistoricOps("old_op_not_exist", 2);
  EXPECT_EQ(oldSchemas.size(), 2);
  for (const auto& entry : oldSchemas) {
    EXPECT_TRUE(
        std::find(schemas.begin(), schemas.end(), entry) != schemas.end());
  }

  auto oldSchemasWithCurrentVersion =
      loadPossibleHistoricOps("old_op_not_exist", 9);
  EXPECT_EQ(oldSchemasWithCurrentVersion.size(), 0);

  test_only_remove_entry("old_op_not_exist.first");
  test_only_remove_entry("old_op_not_exist.first");

  // it is ok to have old schemas without overload
  test_only_add_entry("old_op_not_exist_no_overload", dummy_entry[0]);
  auto oldSchemasNoOverload =
      loadPossibleHistoricOps("old_op_not_exist_no_overload", 2);
  EXPECT_EQ(oldSchemasNoOverload.size(), 1);
  EXPECT_EQ(oldSchemasNoOverload[0], "foo.bar()");
  test_only_remove_entry("old_op_not_exist_no_overload");
}

} // namespace jit
} // namespace torch
