#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <torch/torch.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/engine.h>

#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace torch;
using namespace torch::nn;

struct TestModule : public torch::nn::Module {
  TestModule(long size)
      : torch::nn::Module("TestModule"),
        tensor1(at::randn(torch::CPU(at::kFloat), {size})),
        tensor2(at::randn(torch::CPU(at::kFloat), {size})) {
    // Temporary, will figure out an API for this
    as_variable_ref(tensor1).set_requires_grad(true);
    as_variable_ref(tensor2).set_requires_grad(true);
    register_parameters({{"tensor1", tensor1}, {"tensor2", tensor2}});
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    return {tensor1 + tensor2};
  }

  Tensor tensor1;
  Tensor tensor2;
};

TEST_CASE("ModuleCursor", "[cursor]") {
  SECTION("Works for flat models (depth = 1)") {
    Sequential model(TestModule(1), TestModule(2), TestModule(3));
    auto cursor = model.modules();

    SECTION("Iterates in the correct order") {
      auto iterator = cursor.begin();
      REQUIRE(&iterator->value == &model[0]);
      REQUIRE(&(++iterator)->value == &model[1]);
      REQUIRE(&(++iterator)->value == &model[2]);
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
      REQUIRE(cursor.size() == model.size());
    }

    SECTION("find() finds the correct modules") {
      REQUIRE(cursor.find("0") == &model[0]);
      REQUIRE(cursor.find("1") == &model[1]);
      REQUIRE(cursor.find("2") == &model[2]);
      REQUIRE(cursor.find("foo") == nullptr);
      REQUIRE(cursor.find("bar") == nullptr);
    }

    SECTION("contains() is correct") {
      REQUIRE(cursor.contains("0"));
      REQUIRE(cursor.contains("1"));
      REQUIRE(cursor.contains("2"));
    }
  }

  SECTION("Works for deeper hierarchies (depth > 1)") {
    // clang-format off
    Sequential model(
        Sequential(
          TestModule(1),
          TestModule(2)),
        TestModule(3),
        Sequential(
          TestModule(4),
          Sequential(
            TestModule(5),
            TestModule(6))
        ));
    // clang-format on

    auto cursor = model.modules();
    // This is sufficient for the hierarchical case (other tests build on top)
    SECTION("Iterates in the correct order") {
      auto iterator = cursor.begin();

      REQUIRE(&iterator->value == &model[0]);

      auto* seq = dynamic_cast<Sequential*>(&model[0]);
      REQUIRE(seq != nullptr);
      REQUIRE(&(++iterator)->value == &(*seq)[0]);
      REQUIRE(&(++iterator)->value == &(*seq)[1]);

      REQUIRE(&(++iterator)->value == &model[1]);
      REQUIRE(&(++iterator)->value == &model[2]);

      seq = dynamic_cast<Sequential*>(&model[2]);
      REQUIRE(seq != nullptr);
      REQUIRE(&(++iterator)->value == &(*seq)[0]);
      REQUIRE(&(++iterator)->value == &(*seq)[1]);

      seq = dynamic_cast<Sequential*>(&(*seq)[1]);
      REQUIRE(seq != nullptr);
      REQUIRE(&(++iterator)->value == &(*seq)[0]);
      REQUIRE(&(++iterator)->value == &(*seq)[1]);
    }
  }
}

TEST_CASE("ParameterCursor", "[cursor]") {
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
    Sequential model(first, second);
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
      cursor.apply_items([&count, &model, &first, &second](
                             const std::string& key, Tensor& tensor) {
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
    Sequential model(
        Sequential(
          modules[0],
          modules[1]),
        modules[2],
        Sequential(
          modules[3],
          Sequential(
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
  }
}

void backward(Tensor loss) {
  torch::autograd::edge_list edgelst;
  torch::autograd::variable_list varlst;
  edgelst.emplace_back(loss.grad_fn(), loss.output_nr());
  varlst.emplace_back(
      torch::autograd::make_variable(ones_like(loss.data()), false));
  torch::autograd::Engine engine;
  engine.execute(edgelst, varlst, false, false);
}

TEST_CASE("Modules", "[modules]") {
  TestModule module(1);
  SECTION("Recursive transformations") {
    SECTION("zero_grad zeroes out gradients") {
      auto output = module({});
      auto loss = output.front().sum();
      backward(loss);
      module.zero_grad();
      REQUIRE(as_variable_ref(module.tensor1).grad().sum().toCFloat() == 0);
    }
  }
}
