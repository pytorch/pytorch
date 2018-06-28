#include <catch.hpp>

#include <torch/nn/cursor.h>
#include <torch/nn/module.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace torch::nn;
using namespace torch::detail;

using Catch::StartsWith;

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

TEST_CASE("cursor/module") {
  torch::manual_seed(0);
  SECTION("Works for flat models (depth = 1)") {
    Container model(TestModule(1), TestModule(2), TestModule(3));
    auto cursor = model.modules();

    SECTION("Iterates in the correct order") {
      auto iterator = cursor.begin();
      REQUIRE(&iterator->value == &model[0]);
      REQUIRE(&(++iterator)->value == &model[1]);
      REQUIRE(&(++iterator)->value == &model[2]);
      REQUIRE(++iterator == cursor.end());
    }

    SECTION("names are flat") {
      auto iterator = cursor.begin();
      REQUIRE(iterator->key == "0");
      REQUIRE((++iterator)->key == "1");
      REQUIRE((++iterator)->key == "2");
    }

    SECTION("Apply works") {
      size_t count = 0;
      cursor.apply([&count, &model](Module& module) {
        REQUIRE(&module == &model[count]);
        count += 1;
      });
      REQUIRE(count == 3);
    }

    SECTION("Apply_items works") {
      size_t count = 0;
      cursor.apply_items(
          [&count, &model](const std::string& key, Module& module) {
            REQUIRE(&module == &model[count]);
            count += 1;
          });
      REQUIRE(count == 3);
    }

    SECTION("Map works") {
      std::vector<Module*> vector(3);
      cursor.map(vector.begin(), [](Module& module) { return &module; });

      std::list<Module*> list;
      cursor.map(
          std::back_inserter(list), [](Module& module) { return &module; });
    }

    SECTION("Map_items works") {
      std::map<const char*, Module*> output;
      cursor.map_items(
          std::inserter(output, output.end()),
          [](const std::string& key, Module& module) {
            return std::make_pair(key.c_str(), &module);
          });
    }

    SECTION("Count works for flat models") {
      REQUIRE(cursor.size() == model.m.size());
    }

    SECTION("find() finds the correct modules when given a valid key") {
      REQUIRE(cursor.find("0") == &model[0]);
      REQUIRE(cursor.find("1") == &model[1]);
      REQUIRE(cursor.find("2") == &model[2]);
    }

    SECTION("find() returns nullptr when given an invalid key") {
      REQUIRE(cursor.find("foo") == nullptr);
      REQUIRE(cursor.find("bar") == nullptr);
    }

    SECTION("at(key) returns the correct modules when given a valid key") {
      REQUIRE(&cursor.at("0") == &model[0]);
      REQUIRE(&cursor.at("1") == &model[1]);
      REQUIRE(&cursor.at("2") == &model[2]);
    }

    SECTION("at(key) throws when given an invalid key") {
      REQUIRE_THROWS_WITH(cursor.at("foo"), StartsWith("No such key: 'foo'"));
      REQUIRE_THROWS_WITH(cursor.at("bar"), StartsWith("No such key: 'bar'"));
    }

    SECTION(
        "operator[key] returns the correct modules when given a valid key") {
      REQUIRE(&cursor["0"] == &model[0]);
      REQUIRE(&cursor["1"] == &model[1]);
      REQUIRE(&cursor["2"] == &model[2]);
    }

    SECTION("operator[key] throws when given an invalid key") {
      REQUIRE_THROWS_WITH(cursor["foo"], StartsWith("No such key: 'foo'"));
      REQUIRE_THROWS_WITH(cursor["bar"], StartsWith("No such key: 'bar'"));
    }

    SECTION("at(index) returns the correct modules when given a valid index") {
      REQUIRE(&cursor.at(0).value == &model[0]);
      REQUIRE(&cursor.at(1).value == &model[1]);
      REQUIRE(&cursor.at(2).value == &model[2]);
    }

    SECTION("at(index) throws when given an invalid index") {
      REQUIRE_THROWS_WITH(
          cursor.at(5),
          StartsWith("Index 5 is out of range for cursor of size 3"));
      REQUIRE_THROWS_WITH(
          cursor.at(123),
          StartsWith("Index 123 is out of range for cursor of size 3"));
    }

    SECTION(
        "operator[index] returns the correct modules when given a valid index") {
      REQUIRE(&cursor[0].value == &model[0]);
      REQUIRE(&cursor[1].value == &model[1]);
      REQUIRE(&cursor[2].value == &model[2]);
    }

    SECTION("operator[index] throws when given an invalid key") {
      REQUIRE_THROWS_WITH(
          cursor[5],
          StartsWith("Index 5 is out of range for cursor of size 3"));
      REQUIRE_THROWS_WITH(
          cursor[123],
          StartsWith("Index 123 is out of range for cursor of size 3"));
    }

    SECTION("contains() is correct") {
      REQUIRE(cursor.contains("0"));
      REQUIRE(cursor.contains("1"));
      REQUIRE(cursor.contains("2"));
    }
  }

  SECTION("Works for deeper hierarchies (depth > 1)") {
    // clang-format off
    Container model(
        Container(
          TestModule(1),
          TestModule(2)),
        TestModule(3),
        Container(
          TestModule(4),
          Container(
            TestModule(5),
            TestModule(6))
        ));
    // clang-format on

    auto cursor = model.modules();
    // This is sufficient for the hierarchical case
    // (other tests build on top)
    SECTION("Iterates in the correct order") {
      auto iterator = cursor.begin();

      REQUIRE(&iterator->value == &model[0]);

      auto* seq = dynamic_cast<Container*>(&model[0]);
      REQUIRE(seq != nullptr);
      REQUIRE(&(++iterator)->value == &(*seq)[0]);
      REQUIRE(&(++iterator)->value == &(*seq)[1]);

      REQUIRE(&(++iterator)->value == &model[1]);
      REQUIRE(&(++iterator)->value == &model[2]);

      seq = dynamic_cast<Container*>(&model[2]);
      REQUIRE(seq != nullptr);
      REQUIRE(&(++iterator)->value == &(*seq)[0]);
      REQUIRE(&(++iterator)->value == &(*seq)[1]);

      seq = dynamic_cast<Container*>(&(*seq)[1]);
      REQUIRE(seq != nullptr);
      REQUIRE(&(++iterator)->value == &(*seq)[0]);
      REQUIRE(&(++iterator)->value == &(*seq)[1]);
    }

    SECTION("children() returns only the first level of submodules") {
      auto children = model.children();
      REQUIRE(children.size() == 3);
      REQUIRE(&children.at("0") == &model[0]);
      REQUIRE(&children.at("1") == &model[1]);
      REQUIRE(&children.at("2") == &model[2]);
      REQUIRE(!children.contains("0.0"));
      size_t count = 0;
      for (auto& child : children) {
        REQUIRE(child.key == std::to_string(count));
        REQUIRE(&child.value == &model[count]);
        count += 1;
      }
    }
  }
}

TEST_CASE("cursor/parameter") {
  torch::manual_seed(0);
  SECTION("Works for single models") {
    TestModule model(1);
    auto cursor = model.parameters();

    SECTION("Iterates in the correct order") {
      auto iterator = cursor.begin();
      REQUIRE(iterator->value.equal(model.tensor1));
      REQUIRE((++iterator)->value.equal(model.tensor2));
    }
  }

  SECTION("Works for flat models (depth = 1)") {
    auto first = std::make_shared<TestModule>(1);
    auto second = std::make_shared<TestModule>(2);
    Container model(first, second);
    auto cursor = model.parameters();

    SECTION("Iterates in the correct order") {
      auto iterator = cursor.begin();
      REQUIRE(iterator->value.equal(first->tensor1));
      REQUIRE((++iterator)->value.equal(first->tensor2));
      REQUIRE((++iterator)->value.equal(second->tensor1));
      REQUIRE((++iterator)->value.equal(second->tensor2));
    }

    SECTION("Apply_items works") {
      size_t count = 0;
      cursor.apply_items(
          [&count, &model, &first, &second](
              const std::string& key, torch::Tensor& tensor) {
            switch (count) {
              case 0: {
                REQUIRE(tensor.equal(first->tensor1));
                break;
              }
              case 1: {
                REQUIRE(tensor.equal(first->tensor2));
                break;
              }
              case 2: {
                REQUIRE(tensor.equal(second->tensor1));
                break;
              }
              case 3: {
                REQUIRE(tensor.equal(second->tensor2));
                break;
              }
            }
            count += 1;
          });
      REQUIRE(count == 4);
    }

    // Other tests are correct based on correct iteration behavior and apply
    // working.
  }

  SECTION("Works for deeper hierarchies (depth > 1)") {
    std::vector<std::shared_ptr<TestModule>> modules;
    for (size_t i = 1; i <= 6; ++i) {
      modules.push_back(std::make_shared<TestModule>(i));
    }
    // clang-format off
    Container model(
        Container(
          modules[0],
          modules[1]),
        modules[2],
        Container(
          modules[3],
          Container(
            modules[4],
            modules[5])
        ));
    // clang-format on
    auto cursor = model.parameters();

    SECTION("Iterates in the correct order") {
      auto iterator = cursor.begin();
      REQUIRE(iterator->value.equal(modules[0]->tensor1));
      REQUIRE((++iterator)->value.equal(modules[0]->tensor2));
      for (size_t index = 1; index < 6; ++index) {
        REQUIRE((++iterator)->value.equal(modules[index]->tensor1));
        REQUIRE((++iterator)->value.equal(modules[index]->tensor2));
      }
    }

    SECTION("names are hierarchical") {
      auto iterator = cursor.begin();
      REQUIRE(iterator->key == "0.0.tensor1");
      REQUIRE((++iterator)->key == "0.0.tensor2");
      REQUIRE((++iterator)->key == "0.1.tensor1");
      REQUIRE((++iterator)->key == "0.1.tensor2");
      REQUIRE((++iterator)->key == "1.tensor1");
      REQUIRE((++iterator)->key == "1.tensor2");
      REQUIRE((++iterator)->key == "2.0.tensor1");
      REQUIRE((++iterator)->key == "2.0.tensor2");
      REQUIRE((++iterator)->key == "2.1.0.tensor1");
      REQUIRE((++iterator)->key == "2.1.0.tensor2");
      REQUIRE((++iterator)->key == "2.1.1.tensor1");
      REQUIRE((++iterator)->key == "2.1.1.tensor2");
      REQUIRE(++iterator == cursor.end());
    }
  }
}

TEST_CASE("cursor/non-const-to-const-conversion") {
  torch::manual_seed(0);
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

TEST_CASE("cursor/can-invoke-const-method-on-const-cursor") {
  torch::manual_seed(0);
  TestModule model(1);

  /// This will only compile if `Cursor` has the appropriate const methods.
  const auto cursor = model.parameters();
  REQUIRE(cursor.contains("tensor1"));
}
