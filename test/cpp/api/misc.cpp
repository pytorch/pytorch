#include "catch_utils.hpp"

#include <torch/detail/ordered_dict.h>
#include <torch/expanding_array.h>
#include <torch/nn/init.h>
#include <torch/nn/modules/linear.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <torch/csrc/utils/memory.h>

#include <ATen/core/optional.h>

using namespace torch::nn;

template <typename T>
using OrderedDict = torch::detail::OrderedDict<std::string, T>;

using Catch::StartsWith;

CATCH_TEST_CASE("NoGrad") {
  torch::manual_seed(0);
  torch::NoGradGuard guard;
  Linear model(5, 2);
  auto x = torch::randn({10, 5}, torch::requires_grad());
  auto y = model->forward(x);
  torch::Tensor s = y.sum();

  s.backward();
  CATCH_REQUIRE(!model->parameters()["weight"].grad().defined());
}

CATCH_TEST_CASE("autograd") {
  torch::manual_seed(0);
  auto x = torch::randn({3, 3}, torch::requires_grad());
  auto y = torch::randn({3, 3});
  auto z = x * y;
  CATCH_SECTION("derivatives of zero-dim tensors") {
    z.sum().backward();
    CATCH_REQUIRE(x.grad().allclose(y));
  }
  CATCH_SECTION("derivatives of tensors") {
    z.backward();
    CATCH_REQUIRE(x.grad().allclose(y));
  }
  CATCH_SECTION("custom gradient inputs") {
    z.sum().backward(torch::ones({}) * 2);
    CATCH_REQUIRE(x.grad().allclose(y * 2));
  }
  // Assume everything else is safe from PyTorch tests.
}

CATCH_TEST_CASE("nn::init") {
  auto tensor = torch::empty({3, 4}, torch::requires_grad());
  CATCH_REQUIRE_THROWS_WITH(
      tensor.fill_(1),
      StartsWith("a leaf Variable that requires grad "
                 "has been used in an in-place operation"));
  CATCH_REQUIRE(torch::nn::init::ones_(tensor).sum().toCInt() == 12);
}

CATCH_TEST_CASE("expanding-array") {
  torch::manual_seed(0);
  CATCH_SECTION("successful construction") {
    CATCH_SECTION("initializer_list") {
      torch::ExpandingArray<5> e({1, 2, 3, 4, 5});
      CATCH_REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        CATCH_REQUIRE((*e)[i] == i + 1);
      }
    }

    CATCH_SECTION("vector") {
      torch::ExpandingArray<5> e(std::vector<int64_t>{1, 2, 3, 4, 5});
      CATCH_REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        CATCH_REQUIRE((*e)[i] == i + 1);
      }
    }

    CATCH_SECTION("array") {
      torch::ExpandingArray<5> e(std::array<int64_t, 5>({1, 2, 3, 4, 5}));
      CATCH_REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        CATCH_REQUIRE((*e)[i] == i + 1);
      }
    }

    CATCH_SECTION("single value") {
      torch::ExpandingArray<5> e(5);
      CATCH_REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        CATCH_REQUIRE((*e)[i] == 5);
      }
    }
  }
  CATCH_SECTION("throws for incorrect size on construction") {
    CATCH_SECTION("initializer_list") {
      CATCH_REQUIRE_THROWS_WITH(
          torch::ExpandingArray<5>({1, 2, 3, 4, 5, 6, 7}),
          StartsWith("Expected 5 values, but instead got 7"));
    }
    CATCH_SECTION("vector") {
      CATCH_REQUIRE_THROWS_WITH(
          torch::ExpandingArray<5>(std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7})),
          StartsWith("Expected 5 values, but instead got 7"));
    }
  }
}

CATCH_TEST_CASE("make_unique") {
  struct Test {
    explicit Test(const int& x) : lvalue_(x) {}
    explicit Test(int&& x) : rvalue_(x) {}

    at::optional<int> lvalue_;
    at::optional<int> rvalue_;
  };

  CATCH_SECTION("forwards rvalues correctly") {
    auto ptr = torch::make_unique<Test>(123);
    CATCH_REQUIRE(!ptr->lvalue_.has_value());
    CATCH_REQUIRE(ptr->rvalue_.has_value());
    CATCH_REQUIRE(*ptr->rvalue_ == 123);
  }

  CATCH_SECTION("forwards lvalues correctly") {
    int x = 5;
    auto ptr = torch::make_unique<Test>(x);
    CATCH_REQUIRE(ptr->lvalue_.has_value());
    CATCH_REQUIRE(*ptr->lvalue_ == 5);
    CATCH_REQUIRE(!ptr->rvalue_.has_value());
  }

  CATCH_SECTION("Can construct unique_ptr of array") {
    auto ptr = torch::make_unique<int[]>(3);
    // Value initialization is required by the standard.
    CATCH_REQUIRE(ptr[0] == 0);
    CATCH_REQUIRE(ptr[1] == 0);
    CATCH_REQUIRE(ptr[2] == 0);
  }
}

CATCH_TEST_CASE("ordered-dict") {
  CATCH_SECTION("is empty after default construction") {
    OrderedDict<int> dict;
    CATCH_REQUIRE(dict.subject() == "Key");
    CATCH_REQUIRE(dict.is_empty());
    CATCH_REQUIRE(dict.size() == 0);
  }

  CATCH_SECTION("insert inserts elements when they are not yet present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    CATCH_REQUIRE(dict.size() == 2);
  }

  CATCH_SECTION("get returns values when present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    CATCH_REQUIRE(dict.get("a") == 1);
    CATCH_REQUIRE(dict.get("b") == 2);
  }

  CATCH_SECTION("get throws when passed keys that are not present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    CATCH_REQUIRE_THROWS_WITH(
        dict.get("foo"), StartsWith("Key 'foo' is not defined"));
    CATCH_REQUIRE_THROWS_WITH(dict.get(""), StartsWith("Key '' is not defined"));
  }

  CATCH_SECTION("can initialize from list") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE(dict.size() == 2);
    CATCH_REQUIRE(dict.get("a") == 1);
    CATCH_REQUIRE(dict.get("b") == 2);
  }

  CATCH_SECTION("insert throws when passed elements that are present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE_THROWS_WITH(
        dict.insert("a", 1), StartsWith("Key 'a' already defined"));
    CATCH_REQUIRE_THROWS_WITH(
        dict.insert("b", 1), StartsWith("Key 'b' already defined"));
  }

  CATCH_SECTION("front() returns the first item") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE(dict.front().key == "a");
    CATCH_REQUIRE(dict.front().value == 1);
  }

  CATCH_SECTION("back() returns the last item") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE(dict.back().key == "b");
    CATCH_REQUIRE(dict.back().value == 2);
  }

  CATCH_SECTION("find returns pointers to values when present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE(dict.find("a") != nullptr);
    CATCH_REQUIRE(*dict.find("a") == 1);
    CATCH_REQUIRE(dict.find("b") != nullptr);
    CATCH_REQUIRE(*dict.find("b") == 2);
  }

  CATCH_SECTION("find returns null pointers when passed keys that are not present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE(dict.find("bar") == nullptr);
    CATCH_REQUIRE(dict.find("") == nullptr);
  }

  CATCH_SECTION("operator[] returns values when passed keys that are present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE(dict["a"] == 1);
    CATCH_REQUIRE(dict["b"] == 2);
  }

  CATCH_SECTION("operator[] returns items positionally when passed integers") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE(dict[0].key == "a");
    CATCH_REQUIRE(dict[0].value == 1);
    CATCH_REQUIRE(dict[1].key == "b");
    CATCH_REQUIRE(dict[1].value == 2);
  }

  CATCH_SECTION("operator[] throws when passed keys that are not present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE_THROWS_WITH(
        dict.get("foo"), StartsWith("Key 'foo' is not defined"));
    CATCH_REQUIRE_THROWS_WITH(dict.get(""), StartsWith("Key '' is not defined"));
  }

  CATCH_SECTION("update inserts all items from another OrderedDict") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> dict2 = {{"c", 3}};
    dict2.update(dict);
    CATCH_REQUIRE(dict2.size() == 3);
    CATCH_REQUIRE(dict2.find("a") != nullptr);
    CATCH_REQUIRE(dict2.find("b") != nullptr);
    CATCH_REQUIRE(dict2.find("c") != nullptr);
  }

  CATCH_SECTION("update also checks for duplicates") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> dict2 = {{"a", 1}};
    CATCH_REQUIRE_THROWS_WITH(
        dict2.update(dict), StartsWith("Key 'a' already defined"));
  }

  CATCH_SECTION("Can iterate items") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    auto iterator = dict.begin();
    CATCH_REQUIRE(iterator != dict.end());
    CATCH_REQUIRE(iterator->key == "a");
    CATCH_REQUIRE(iterator->value == 1);
    ++iterator;
    CATCH_REQUIRE(iterator != dict.end());
    CATCH_REQUIRE(iterator->key == "b");
    CATCH_REQUIRE(iterator->value == 2);
    ++iterator;
    CATCH_REQUIRE(iterator == dict.end());
  }

  CATCH_SECTION("clear makes the dict empty") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    CATCH_REQUIRE(!dict.is_empty());
    dict.clear();
    CATCH_REQUIRE(dict.is_empty());
  }

  CATCH_SECTION("can copy construct") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = dict;
    CATCH_REQUIRE(copy.size() == 2);
    CATCH_REQUIRE(*copy[0] == 1);
    CATCH_REQUIRE(*copy[1] == 2);
  }

  CATCH_SECTION("can copy assign") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = {{"c", 1}};
    CATCH_REQUIRE(copy.find("c") != nullptr);
    copy = dict;
    CATCH_REQUIRE(copy.size() == 2);
    CATCH_REQUIRE(*copy[0] == 1);
    CATCH_REQUIRE(*copy[1] == 2);
    CATCH_REQUIRE(copy.find("c") == nullptr);
  }

  CATCH_SECTION("can move construct") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = std::move(dict);
    CATCH_REQUIRE(copy.size() == 2);
    CATCH_REQUIRE(*copy[0] == 1);
    CATCH_REQUIRE(*copy[1] == 2);
  }

  CATCH_SECTION("can move assign") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = {{"c", 1}};
    CATCH_REQUIRE(copy.find("c") != nullptr);
    copy = std::move(dict);
    CATCH_REQUIRE(copy.size() == 2);
    CATCH_REQUIRE(*copy[0] == 1);
    CATCH_REQUIRE(*copy[1] == 2);
    CATCH_REQUIRE(copy.find("c") == nullptr);
  }

  CATCH_SECTION("can insert with braces") {
    OrderedDict<std::pair<int, int>> dict;
    dict.insert("a", {1, 2});
    CATCH_REQUIRE(!dict.is_empty());
    CATCH_REQUIRE(dict["a"].first == 1);
    CATCH_REQUIRE(dict["a"].second == 2);
  }

  CATCH_SECTION("Error messages include the what") {
    OrderedDict<int> dict("Penguin");
    CATCH_REQUIRE(dict.subject() == "Penguin");
    dict.insert("a", 1);
    CATCH_REQUIRE(!dict.is_empty());
    CATCH_REQUIRE_THROWS_WITH(
        dict.get("b"), StartsWith("Penguin 'b' is not defined"));
    CATCH_REQUIRE_THROWS_WITH(
        dict.insert("a", 1), StartsWith("Penguin 'a' already defined"));
  }
}
