#include <Python.h>
#include <gtest/gtest.h>
#include <torch/csrc/deploy/interpreter/builtin_registry.h>

using namespace torch::deploy;

namespace torch {
namespace deploy {
bool allowLibrary(const std::string& libname) {
  return libname == "lib1" || libname == "lib2";
}
} // namespace deploy
} // namespace torch

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
struct _frozen lib1FrozenModules[] = {
    {"mod1", nullptr, 0},
    {nullptr, nullptr, 0}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
struct _frozen lib2FrozenModules[] = {
    {"mod2", nullptr, 0},
    {"mod3", nullptr, 0},
    {nullptr, nullptr, 0}};

REGISTER_TORCH_DEPLOY_BUILTIN(lib1, lib1FrozenModules);
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
}
