#include <gtest/gtest.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ParameterDictTest : torch::test::SeedingFixture {};

TEST_F(ParameterDictTest, ConstructFromTensor) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  dict->insert("B", torch::tensor({2.0}));
  dict->insert("C", torch::tensor({3.3}));
  ASSERT_EQ(dict->size(), 3);
}

TEST_F(ParameterDictTest, ConstructFromOrderedDict) {
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", torch::tensor({1.0})}, {"b", torch::tensor({2.0})}};
  auto dict = torch::nn::ParameterDict(params);
  ASSERT_EQ(dict->size(), 2);
}

TEST_F(ParameterDictTest, InsertAndContains) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  ASSERT_TRUE(dict->contains("A"));
  ASSERT_FALSE(dict->contains("C"));
}

TEST_F(ParameterDictTest, InsertAndClear) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  dict->clear();
  ASSERT_EQ(dict->size(), 0);
}

TEST_F(ParameterDictTest, InsertAndErase) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  dict->erase("A");
  ASSERT_EQ(dict->size(), 0);
}

TEST_F(ParameterDictTest, InsertAndGetTest) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  ASSERT_TRUE(torch::eq(dict["A"], torch::tensor({1.0})).item<bool>());
}

TEST_F(ParameterDictTest, Keys) {
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", torch::tensor({1.0})},
      {"b", torch::tensor({2.0})},
      {"c", torch::tensor({1.0, 2.0})}};
  auto dict = torch::nn::ParameterDict(params);
  std::vector<std::string> keys = dict->keys();
  std::vector<std::string> true_keys{"a", "b", "c"};
  ASSERT_EQ(keys, true_keys);
}

TEST_F(ParameterDictTest, Values) {
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", torch::tensor({1.0})},
      {"b", torch::tensor({2.0})},
      {"c", torch::tensor({3.0})}};
  auto dict = torch::nn::ParameterDict(params);
  std::vector<torch::Tensor> values = dict->values();
  std::vector<torch::Tensor> true_values{
      torch::tensor({1.0}), torch::tensor({2.0}), torch::tensor({3.0})};
  for (auto i = 0; i < values.size(); i += 1) {
    ASSERT_TRUE(torch::eq(values[i], true_values[i]).item<bool>());
  }
}