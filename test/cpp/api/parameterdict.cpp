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

TEST_F(ParameterDictTest, InsertAndPop) {
  ParameterDict dict;
  dict->insert("A", torch::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  ASSERT_THROWS_WITH(
      dict->pop("B"), "No Parameter with name `B` is registered");
  torch::Tensor p = dict->pop("A");
  ASSERT_EQ(dict->size(), 0);
  ASSERT_TRUE(torch::eq(p, torch::tensor({1.0})).item<bool>());
}

TEST_F(ParameterDictTest, SimpleUpdate) {
  ParameterDict dict;
  ParameterDict otherDict;
  dict->insert("A", torch::tensor({1.0}));
  dict->insert("B", torch::tensor({2.0}));
  dict->insert("C", torch::tensor({3.0}));
  otherDict->insert("A", torch::tensor({5.0}));
  otherDict->insert("D", torch::tensor({5.0}));
  dict->update(*otherDict);
  ASSERT_EQ(dict->size(), 4);
  ASSERT_TRUE(torch::eq(dict["A"], torch::tensor({5.0})).item<bool>());
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

TEST_F(ParameterDictTest, PrettyPrintParameterDict) {
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", torch::tensor({1.0})},
      {"b", torch::tensor({2.0})},
      {"c", torch::tensor({3.0})}};
  auto dict = torch::nn::ParameterDict(params);
  ASSERT_EQ(
      c10::str(dict),
      "torch::nn::ParameterDict(\n"
      "(a): Parameter containing: [CPUFloatType of size [1]]\n"
      "(b): Parameter containing: [CPUFloatType of size [1]]\n"
      "(c): Parameter containing: [CPUFloatType of size [1]]\n"
      ")");
}