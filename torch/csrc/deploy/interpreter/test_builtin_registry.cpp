#include <Python.h>
#include <gtest/gtest.h>
#include <torch/csrc/deploy/interpreter/builtin_registry.h>

namespace torch {
namespace deploy {

bool allowLibrary(const std::string& libname) {
  return libname == "lib1" || libname == "lib2";
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
struct _frozen lib1FrozenModules[] = {
    {"mod1", nullptr, 0},
    {nullptr, nullptr, 0}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
struct _frozen lib2FrozenModules[] = {
    {"mod2", nullptr, 0},
    {"mod3", nullptr, 0},
    {nullptr, nullptr, 0}};

void builtin1() {}
void builtin2() {}
REGISTER_TORCH_DEPLOY_BUILTIN(
    lib1,
    lib1FrozenModules,
    "lib1.builtin1",
    builtin1,
    "lib1.builtin2",
    builtin2);
REGISTER_TORCH_DEPLOY_BUILTIN(lib2, lib2FrozenModules);

TEST(BuiltinRegistryTest, SimpleTest) {
  const auto& items = BuiltinRegistry::items();
  EXPECT_EQ(2, items.size());
  EXPECT_EQ(lib1FrozenModules, items[0]->frozenModules);
  EXPECT_EQ(lib2FrozenModules, items[1]->frozenModules);

  struct _frozen* allFrozenModules = BuiltinRegistry::getAllFrozenModules();
  EXPECT_EQ("mod1", allFrozenModules[0].name);
  EXPECT_EQ("mod2", allFrozenModules[1].name);
  EXPECT_EQ("mod3", allFrozenModules[2].name);
  EXPECT_EQ(nullptr, allFrozenModules[3].name);

  auto allBuiltinModules = BuiltinRegistry::getAllBuiltinModules();
  EXPECT_EQ(2, allBuiltinModules.size());
  EXPECT_EQ("lib1.builtin1", allBuiltinModules[0].first);
  EXPECT_EQ(builtin1, allBuiltinModules[0].second);
  EXPECT_EQ("lib1.builtin2", allBuiltinModules[1].first);
  EXPECT_EQ(builtin2, allBuiltinModules[1].second);

  std::string expectedBuiltinModulesCSV = "'lib1.builtin1', 'lib1.builtin2'";
  EXPECT_EQ(expectedBuiltinModulesCSV, BuiltinRegistry::getBuiltinModulesCSV());
}

} // namespace deploy
} // namespace torch
