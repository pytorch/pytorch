#include <gtest/gtest.h>

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ModuleDictTest : torch::test::SeedingFixture {};

TEST_F(ModuleDictTest, ConstructsFromSharedPointer) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  ModuleDict dict;
  dict->insert("A", std::make_shared<M>(1));
  dict->insert("B", std::make_shared<M>(2));
  dict->insert("C", std::make_shared<M>(3));
  ASSERT_EQ(dict->size(), 3);
}
