#include <gtest/gtest.h>
#include <torch/csrc/deploy/builtin_registry.h>
#include <Python.h>

using namespace torch::deploy;

struct _frozen lib1_frozen_modules[] = {
    {"mod1", nullptr, 0},
    {nullptr, nullptr, 0}};

struct _frozen lib2_frozen_modules[] = {
    {"mod2", nullptr, 0},
    {"mod3", nullptr, 0},
    {nullptr, nullptr, 0}};

REGISTER_TORCH_DEPLOY_BUILTIN(lib1, lib1_frozen_modules);
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
}
