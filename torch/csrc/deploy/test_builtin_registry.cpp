#include <Python.h>
#include <gtest/gtest.h>
#include <torch/csrc/deploy/builtin_registry.h>

using namespace torch::deploy;

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
struct _frozen lib1_frozen_modules[] = {
    {"mod1", nullptr, 0},
    {nullptr, nullptr, 0}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
struct _frozen lib2_frozen_modules[] = {
    {"mod2", nullptr, 0},
    {"mod3", nullptr, 0},
    {nullptr, nullptr, 0}};

void builtin1() {}
void builtin2() {}
REGISTER_TORCH_DEPLOY_BUILTIN(
    lib1,
    lib1_frozen_modules,
    "lib1.builtin1",
    builtin1,
    "lib1.builtin2",
    builtin2);
REGISTER_TORCH_DEPLOY_BUILTIN(lib2, lib2_frozen_modules);

TEST(BuiltinRegistryTest, SimpleTest) {
  const auto& items = builtin_registry::items();
  EXPECT_EQ(2, items.size());
  EXPECT_EQ(lib1_frozen_modules, items[0]->frozen_modules);
  EXPECT_EQ(lib2_frozen_modules, items[1]->frozen_modules);

  struct _frozen* all_frozen_modules =
      builtin_registry::get_all_frozen_modules();
  EXPECT_EQ("mod1", all_frozen_modules[0].name);
  EXPECT_EQ("mod2", all_frozen_modules[1].name);
  EXPECT_EQ("mod3", all_frozen_modules[2].name);
  EXPECT_EQ(nullptr, all_frozen_modules[3].name);

  auto all_builtin_modules = builtin_registry::get_all_builtin_modules();
  EXPECT_EQ(2, all_builtin_modules.size());
  EXPECT_EQ("lib1.builtin1", all_builtin_modules[0].first);
  EXPECT_EQ(builtin1, all_builtin_modules[0].second);
  EXPECT_EQ("lib1.builtin2", all_builtin_modules[1].first);
  EXPECT_EQ(builtin2, all_builtin_modules[1].second);

  std::string expected_builtin_modules_csv = "'lib1.builtin1', 'lib1.builtin2'";
  EXPECT_EQ(
      expected_builtin_modules_csv,
      builtin_registry::get_builtin_modules_csv());
}
