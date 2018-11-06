#include <gtest/gtest.h>

#include <torch/nn/cursor.h>
#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <test/cpp/api/support.h>

#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace torch::nn;
using namespace torch::detail;

struct TestModule : public torch::nn::Module {
  TestModule(int64_t size) {
    tensor1 = register_parameter("tensor1", torch::randn({size}));
    tensor2 = register_parameter("tensor2", torch::randn({size}));
  }

  torch::Tensor tensor1;
  torch::Tensor tensor2;
};

struct Container : public torch::nn::Module {
  template <typename... Ms>
  explicit Container(Ms&&... ms) {
    add(0, ms...);
  }

  void add(size_t) {}

  template <typename Head, typename... Tail>
  void add(size_t index, Head head, Tail... tail) {
    add(std::to_string(index), std::move(head));
    add(index + 1, tail...);
  }

  template <typename M>
  void add(std::string name, M&& module) {
    m.push_back(register_module(name, std::make_shared<M>(std::move(module))));
  }

  template <typename M>
  void add(std::string name, std::shared_ptr<M>&& module) {
    m.push_back(register_module(name, std::move(module)));
  }

  Module& operator[](size_t index) {
    return *m.at(index);
  }

  std::vector<std::shared_ptr<Module>> m;
};

struct ModuleCursorFlatTest : torch::test::SeedingFixture {
  ModuleCursorFlatTest()
      : model(TestModule(1), TestModule(2), TestModule(3)),
        cursor(model.modules()) {}
  Container model;
  ModuleCursor cursor;
};

TEST_F(ModuleCursorFlatTest, IteratesInTheCorrectOrder) {
  auto iterator = cursor.begin();
  ASSERT_EQ(&iterator->value, &model[0]);
  ASSERT_EQ(&(++iterator)->value, &model[1]);
  ASSERT_EQ(&(++iterator)->value, &model[2]);
  ASSERT_EQ(++iterator, cursor.end());
}

TEST_F(ModuleCursorFlatTest, NamesAreFlat) {
  auto iterator = cursor.begin();
  ASSERT_EQ(iterator->key, "0");
  ASSERT_EQ((++iterator)->key, "1");
  ASSERT_EQ((++iterator)->key, "2");
}

TEST_F(ModuleCursorFlatTest, Apply) {
  size_t count = 0;
  cursor.apply([this, &count](Module& module) {
    ASSERT_EQ(&module, &model[count]);
    count += 1;
  });
  ASSERT_EQ(count, 3);
}

TEST_F(ModuleCursorFlatTest, ApplyItems) {
  size_t count = 0;
  cursor.apply_items([this, &count](const std::string& key, Module& module) {
    ASSERT_EQ(&module, &model[count]);
    count += 1;
  });
  ASSERT_EQ(count, 3);
}

TEST_F(ModuleCursorFlatTest, Map) {
  std::vector<Module*> vector(3);
  cursor.map(vector.begin(), [](Module& module) { return &module; });
  ASSERT_EQ(vector[0], &model[0]);
  ASSERT_EQ(vector[1], &model[1]);
  ASSERT_EQ(vector[2], &model[2]);

  std::list<Module*> list;
  cursor.map(
      std::inserter(list, list.end()), [](Module& module) { return &module; });
  ASSERT_EQ(list.size(), 3);
  auto iterator = list.begin();
  ASSERT_EQ(*iterator++, &model[0]);
  ASSERT_EQ(*iterator++, &model[1]);
  ASSERT_EQ(*iterator++, &model[2]);
  ASSERT_EQ(iterator, list.end());
}

TEST_F(ModuleCursorFlatTest, MapItems) {
  std::map<std::string, Module*> output;
  cursor.map_items(
      std::inserter(output, output.end()),
      [](const std::string& key, Module& module) {
        return std::make_pair(key, &module);
      });
  ASSERT_EQ(output.size(), 3);
  ASSERT_TRUE(output.count("0"));
  ASSERT_TRUE(output.count("1"));
  ASSERT_TRUE(output.count("2"));
  ASSERT_EQ(output["0"], &model[0]);
  ASSERT_EQ(output["1"], &model[1]);
  ASSERT_EQ(output["2"], &model[2]);
}

TEST_F(ModuleCursorFlatTest, Count) {
  ASSERT_EQ(cursor.size(), model.m.size());
}

TEST_F(ModuleCursorFlatTest, FindReturnsTheCorrectModulesWhenGivenAValidKey) {
  ASSERT_EQ(cursor.find("0"), &model[0]);
  ASSERT_EQ(cursor.find("1"), &model[1]);
  ASSERT_EQ(cursor.find("2"), &model[2]);
}

TEST_F(ModuleCursorFlatTest, FindReturnsNullptrWhenGivenAnInvalidKey) {
  ASSERT_EQ(cursor.find("foo"), nullptr);
  ASSERT_EQ(cursor.find("bar"), nullptr);
}

TEST_F(
    ModuleCursorFlatTest,
    AtWithKeyReturnsTheCorrectModulesWhenGivenAValidKey) {
  ASSERT_EQ(&cursor.at("0"), &model[0]);
  ASSERT_EQ(&cursor.at("1"), &model[1]);
  ASSERT_EQ(&cursor.at("2"), &model[2]);
}

TEST_F(ModuleCursorFlatTest, AtWithKeyThrowsWhenGivenAnInvalidKey) {
  ASSERT_THROWS_WITH(cursor.at("foo"), "No such key: 'foo'");
  ASSERT_THROWS_WITH(cursor.at("bar"), "No such key: 'bar'");
}

TEST_F(
    ModuleCursorFlatTest,
    SubscriptOperatorWithKeyReturnsCorrectModulesWhenGivenAValidKey) {
  ASSERT_EQ(&cursor["0"], &model[0]);
  ASSERT_EQ(&cursor["1"], &model[1]);
  ASSERT_EQ(&cursor["2"], &model[2]);
}

TEST_F(ModuleCursorFlatTest, SubscriptOperatorWithKeyWhenGivenAnInvalidKey) {
  ASSERT_THROWS_WITH(cursor["foo"], "No such key: 'foo'");
  ASSERT_THROWS_WITH(cursor["bar"], "No such key: 'bar'");
}

TEST_F(
    ModuleCursorFlatTest,
    AtWithIndexReturnsTheCorrectModulesWhenGivenAValidKey) {
  ASSERT_EQ(&cursor.at(0).value, &model[0]);
  ASSERT_EQ(&cursor.at(1).value, &model[1]);
  ASSERT_EQ(&cursor.at(2).value, &model[2]);
}

TEST_F(ModuleCursorFlatTest, AtWithIndexThrowsWhenGivenAnInvalidKey) {
  ASSERT_THROWS_WITH(
      cursor.at(5), "Index 5 is out of range for cursor of size 3");
  ASSERT_THROWS_WITH(
      cursor.at(123), "Index 123 is out of range for cursor of size 3");
}

TEST_F(
    ModuleCursorFlatTest,
    SubscriptOperatorWithIndexReturnsCorrectModulesWhenGivenAValidKey) {
  ASSERT_EQ(&cursor[0].value, &model[0]);
  ASSERT_EQ(&cursor[1].value, &model[1]);
  ASSERT_EQ(&cursor[2].value, &model[2]);
}

TEST_F(ModuleCursorFlatTest, SubscriptOperatorWithIndexWhenGivenAnInvalidKey) {
  ASSERT_THROWS_WITH(cursor[5], "Index 5 is out of range for cursor of size 3");
  ASSERT_THROWS_WITH(
      cursor[123], "Index 123 is out of range for cursor of size 3");
}

TEST_F(ModuleCursorFlatTest, ContainReturnsTrueWhenKeyIsPresent) {
  ASSERT_TRUE(cursor.contains("0"));
  ASSERT_TRUE(cursor.contains("1"));
  ASSERT_TRUE(cursor.contains("2"));
}

struct ModuleCursorDeepTest : torch::test::SeedingFixture {
  ModuleCursorDeepTest()
      : model(
            Container(TestModule(1), TestModule(2)),
            TestModule(3),
            Container(TestModule(4), Container(TestModule(5), TestModule(6)))) {
  }
  Container model;
};

TEST_F(ModuleCursorDeepTest, IteratesInTheCorrectOrder) {
  auto cursor = model.modules();
  auto iterator = cursor.begin();

  ASSERT_EQ(&iterator->value, &model[0]);

  auto* seq = dynamic_cast<Container*>(&model[0]);
  ASSERT_NE(seq, nullptr);
  ASSERT_EQ(&(++iterator)->value, &(*seq)[0]);
  ASSERT_EQ(&(++iterator)->value, &(*seq)[1]);

  ASSERT_EQ(&(++iterator)->value, &model[1]);
  ASSERT_EQ(&(++iterator)->value, &model[2]);

  seq = dynamic_cast<Container*>(&model[2]);
  ASSERT_NE(seq, nullptr);
  ASSERT_EQ(&(++iterator)->value, &(*seq)[0]);
  ASSERT_EQ(&(++iterator)->value, &(*seq)[1]);

  seq = dynamic_cast<Container*>(&(*seq)[1]);
  ASSERT_NE(seq, nullptr);
  ASSERT_EQ(&(++iterator)->value, &(*seq)[0]);
  ASSERT_EQ(&(++iterator)->value, &(*seq)[1]);
}

TEST_F(ModuleCursorDeepTest, ChildrenReturnsOnlyTheFirstLevelOfSubmodules) {
  auto children = model.children();
  ASSERT_EQ(children.size(), 3);
  ASSERT_EQ(&children.at("0"), &model[0]);
  ASSERT_EQ(&children.at("1"), &model[1]);
  ASSERT_EQ(&children.at("2"), &model[2]);
  ASSERT_FALSE(children.contains("0.0"));
  size_t count = 0;
  for (auto& child : children) {
    ASSERT_EQ(child.key, std::to_string(count));
    ASSERT_EQ(&child.value, &model[count]);
    count += 1;
  }
}

struct ParameterCursorFlatTest : torch::test::SeedingFixture {
  ParameterCursorFlatTest()
      : first(std::make_shared<TestModule>(1)),
        second(std::make_shared<TestModule>(2)),
        model(first, second),
        cursor(model.parameters()) {}
  std::shared_ptr<TestModule> first, second;
  Container model;
  ParameterCursor cursor;
};

TEST(ParameterCursorTest, IteratesInTheCorrectOrderOverSimpleModels) {
  torch::manual_seed(0);
  TestModule model(1);
  auto cursor = model.parameters();
  auto iterator = cursor.begin();
  ASSERT_TRUE(iterator->value.equal(model.tensor1));
  ASSERT_TRUE((++iterator)->value.equal(model.tensor2));
}

TEST_F(ParameterCursorFlatTest, IteratesInTheCorrectOrder) {
  auto iterator = cursor.begin();
  ASSERT_TRUE(iterator->value.equal(first->tensor1));
  ASSERT_TRUE((++iterator)->value.equal(first->tensor2));
  ASSERT_TRUE((++iterator)->value.equal(second->tensor1));
  ASSERT_TRUE((++iterator)->value.equal(second->tensor2));
}

TEST_F(ParameterCursorFlatTest, ApplyItemsWorks) {
  size_t count = 0;
  cursor.apply_items(
      [this, &count](const std::string& key, torch::Tensor& tensor) {
        switch (count) {
          case 0: {
            ASSERT_TRUE(tensor.equal(first->tensor1));
            break;
          }
          case 1: {
            ASSERT_TRUE(tensor.equal(first->tensor2));
            break;
          }
          case 2: {
            ASSERT_TRUE(tensor.equal(second->tensor1));
            break;
          }
          case 3: {
            ASSERT_TRUE(tensor.equal(second->tensor2));
            break;
          }
        }
        count += 1;
      });
  ASSERT_EQ(count, 4);
}

struct ParameterCursorDeepTest : torch::test::SeedingFixture {
  std::vector<std::shared_ptr<TestModule>> make_modules() {
    std::vector<std::shared_ptr<TestModule>> modules;
    for (size_t i = 1; i <= 6; ++i) {
      modules.push_back(std::make_shared<TestModule>(i));
    }
    return modules;
  }

  ParameterCursorDeepTest()
      : modules(make_modules()),
        model(
            Container(modules[0], modules[1]),
            modules[2],
            Container(modules[3], Container(modules[4], modules[5]))) {}

  std::vector<std::shared_ptr<TestModule>> modules;
  Container model;
};

TEST_F(ParameterCursorDeepTest, IteratesInTheCorrectOrderOverDeepModels) {
  auto cursor = model.parameters();
  auto iterator = cursor.begin();
  ASSERT_TRUE(iterator->value.equal(modules[0]->tensor1));
  ASSERT_TRUE((++iterator)->value.equal(modules[0]->tensor2));
  for (size_t index = 1; index < 6; ++index) {
    ASSERT_TRUE((++iterator)->value.equal(modules[index]->tensor1));
    ASSERT_TRUE((++iterator)->value.equal(modules[index]->tensor2));
  }
}

TEST_F(ParameterCursorDeepTest, NamesAreHierarchical) {
  auto cursor = model.parameters();
  auto iterator = cursor.begin();
  ASSERT_EQ(iterator->key, "0.0.tensor1");
  ASSERT_EQ((++iterator)->key, "0.0.tensor2");
  ASSERT_EQ((++iterator)->key, "0.1.tensor1");
  ASSERT_EQ((++iterator)->key, "0.1.tensor2");
  ASSERT_EQ((++iterator)->key, "1.tensor1");
  ASSERT_EQ((++iterator)->key, "1.tensor2");
  ASSERT_EQ((++iterator)->key, "2.0.tensor1");
  ASSERT_EQ((++iterator)->key, "2.0.tensor2");
  ASSERT_EQ((++iterator)->key, "2.1.0.tensor1");
  ASSERT_EQ((++iterator)->key, "2.1.0.tensor2");
  ASSERT_EQ((++iterator)->key, "2.1.1.tensor1");
  ASSERT_EQ((++iterator)->key, "2.1.1.tensor2");
  ASSERT_EQ(++iterator, cursor.end());
}

struct CursorTest : torch::test::SeedingFixture {};

TEST_F(CursorTest, NonConstToConstConversion) {
  auto first = std::make_shared<TestModule>(1);
  auto second = std::make_shared<TestModule>(2);
  Container model(first, second);

  {
    ConstModuleCursor const_cursor(model.modules());
    {
      ModuleCursor cursor = model.modules();
      ConstModuleCursor const_cursor = cursor;
    }
  }
  {
    ConstParameterCursor const_cursor(model.parameters());
    {
      ParameterCursor cursor = model.parameters();
      ConstParameterCursor const_cursor = cursor;
    }
  }
  {
    ConstBufferCursor const_cursor(model.buffers());
    {
      BufferCursor cursor = model.buffers();
      ConstBufferCursor const_cursor = cursor;
    }
  }
}

TEST_F(CursorTest, CanInvokeConstMethodOnConstCursor) {
  TestModule model(1);

  /// This will only compile if `Cursor` has the appropriate const methods.
  const auto cursor = model.parameters();
  ASSERT_TRUE(cursor.contains("tensor1"));
}
