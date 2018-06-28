#include <catch.hpp>

#include <torch/detail/ordered_dict.h>
#include <torch/expanding_array.h>
#include <torch/nn/modules/linear.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <torch/csrc/utils/memory.h>

#include <ATen/optional.h>

using namespace torch::nn;

template <typename T>
using OrderedDict = torch::detail::OrderedDict<std::string, T>;

using Catch::StartsWith;

TEST_CASE("NoGrad") {
  torch::manual_seed(0);
  torch::NoGradGuard guard;
  Linear model(5, 2);
  auto x = torch::randn({10, 5}, torch::requires_grad());
  auto y = model->forward(x);
  torch::Tensor s = y.sum();

  s.backward();
  REQUIRE(!model->parameters()["weight"].grad().defined());
}

TEST_CASE("autograd") {
  torch::manual_seed(0);
  auto x = torch::randn({3, 3}, torch::requires_grad());
  auto y = torch::randn({3, 3});
  auto z = x * y;
  SECTION("derivatives of zero-dim tensors") {
    z.sum().backward();
    REQUIRE(x.grad().allclose(y));
  }
  SECTION("derivatives of tensors") {
    z.backward();
    REQUIRE(x.grad().allclose(y));
  }
  SECTION("custom gradient inputs") {
    z.sum().backward(torch::ones({}) * 2);
    REQUIRE(x.grad().allclose(y * 2));
  }
  // Assume everything else is safe from PyTorch tests.
}

TEST_CASE("expanding-array") {
  torch::manual_seed(0);
  SECTION("successful construction") {
    SECTION("initializer_list") {
      torch::ExpandingArray<5> e({1, 2, 3, 4, 5});
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("vector") {
      torch::ExpandingArray<5> e(std::vector<int64_t>{1, 2, 3, 4, 5});
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("array") {
      torch::ExpandingArray<5> e(std::array<int64_t, 5>({1, 2, 3, 4, 5}));
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == i + 1);
      }
    }

    SECTION("single value") {
      torch::ExpandingArray<5> e(5);
      REQUIRE(e.size() == 5);
      for (size_t i = 0; i < e.size(); ++i) {
        REQUIRE((*e)[i] == 5);
      }
    }
  }
  SECTION("throws for incorrect size on construction") {
    SECTION("initializer_list") {
      REQUIRE_THROWS_WITH(
          torch::ExpandingArray<5>({1, 2, 3, 4, 5, 6, 7}),
          StartsWith("Expected 5 values, but instead got 7"));
    }
    SECTION("vector") {
      REQUIRE_THROWS_WITH(
          torch::ExpandingArray<5>(std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7})),
          StartsWith("Expected 5 values, but instead got 7"));
    }
  }
}

TEST_CASE("make_unique") {
  struct Test {
    explicit Test(const int& x) : lvalue_(x) {}
    explicit Test(int&& x) : rvalue_(x) {}

    at::optional<int> lvalue_;
    at::optional<int> rvalue_;
  };

  SECTION("forwards rvalues correctly") {
    auto ptr = torch::make_unique<Test>(123);
    REQUIRE(!ptr->lvalue_.has_value());
    REQUIRE(ptr->rvalue_.has_value());
    REQUIRE(*ptr->rvalue_ == 123);
  }

  SECTION("forwards lvalues correctly") {
    int x = 5;
    auto ptr = torch::make_unique<Test>(x);
    REQUIRE(ptr->lvalue_.has_value());
    REQUIRE(*ptr->lvalue_ == 5);
    REQUIRE(!ptr->rvalue_.has_value());
  }

  SECTION("Can construct unique_ptr of array") {
    auto ptr = torch::make_unique<int[]>(3);
    // Value initialization is required by the standard.
    REQUIRE(ptr[0] == 0);
    REQUIRE(ptr[1] == 0);
    REQUIRE(ptr[2] == 0);
  }
}

TEST_CASE("ordered-dict") {
  SECTION("is empty after default construction") {
    OrderedDict<int> dict;
    REQUIRE(dict.subject() == "Key");
    REQUIRE(dict.is_empty());
    REQUIRE(dict.size() == 0);
  }

  SECTION("insert inserts elements when they are not yet present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    REQUIRE(dict.size() == 2);
  }

  SECTION("get returns values when present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    REQUIRE(dict.get("a") == 1);
    REQUIRE(dict.get("b") == 2);
  }

  SECTION("get throws when passed keys that are not present") {
    OrderedDict<int> dict;
    dict.insert("a", 1);
    dict.insert("b", 2);
    REQUIRE_THROWS_WITH(
        dict.get("foo"), StartsWith("Key 'foo' is not defined"));
    REQUIRE_THROWS_WITH(dict.get(""), StartsWith("Key '' is not defined"));
  }

  SECTION("can initialize from list") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.size() == 2);
    REQUIRE(dict.get("a") == 1);
    REQUIRE(dict.get("b") == 2);
  }

  SECTION("insert throws when passed elements that are present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE_THROWS_WITH(
        dict.insert("a", 1), StartsWith("Key 'a' already defined"));
    REQUIRE_THROWS_WITH(
        dict.insert("b", 1), StartsWith("Key 'b' already defined"));
  }

  SECTION("front() returns the first item") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.front().key == "a");
    REQUIRE(dict.front().value == 1);
  }

  SECTION("back() returns the last item") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.back().key == "b");
    REQUIRE(dict.back().value == 2);
  }

  SECTION("find returns pointers to values when present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.find("a") != nullptr);
    REQUIRE(*dict.find("a") == 1);
    REQUIRE(dict.find("b") != nullptr);
    REQUIRE(*dict.find("b") == 2);
  }

  SECTION("find returns null pointers when passed keys that are not present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict.find("bar") == nullptr);
    REQUIRE(dict.find("") == nullptr);
  }

  SECTION("operator[] returns values when passed keys that are present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict["a"] == 1);
    REQUIRE(dict["b"] == 2);
  }

  SECTION("operator[] returns items positionally when passed integers") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(dict[0].key == "a");
    REQUIRE(dict[0].value == 1);
    REQUIRE(dict[1].key == "b");
    REQUIRE(dict[1].value == 2);
  }

  SECTION("operator[] throws when passed keys that are not present") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE_THROWS_WITH(
        dict.get("foo"), StartsWith("Key 'foo' is not defined"));
    REQUIRE_THROWS_WITH(dict.get(""), StartsWith("Key '' is not defined"));
  }

  SECTION("update inserts all items from another OrderedDict") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> dict2 = {{"c", 3}};
    dict2.update(dict);
    REQUIRE(dict2.size() == 3);
    REQUIRE(dict2.find("a") != nullptr);
    REQUIRE(dict2.find("b") != nullptr);
    REQUIRE(dict2.find("c") != nullptr);
  }

  SECTION("update also checks for duplicates") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> dict2 = {{"a", 1}};
    REQUIRE_THROWS_WITH(
        dict2.update(dict), StartsWith("Key 'a' already defined"));
  }

  SECTION("Can iterate items") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    auto iterator = dict.begin();
    REQUIRE(iterator != dict.end());
    REQUIRE(iterator->key == "a");
    REQUIRE(iterator->value == 1);
    ++iterator;
    REQUIRE(iterator != dict.end());
    REQUIRE(iterator->key == "b");
    REQUIRE(iterator->value == 2);
    ++iterator;
    REQUIRE(iterator == dict.end());
  }

  SECTION("clear makes the dict empty") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    REQUIRE(!dict.is_empty());
    dict.clear();
    REQUIRE(dict.is_empty());
  }

  SECTION("can copy construct") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = dict;
    REQUIRE(copy.size() == 2);
    REQUIRE(*copy[0] == 1);
    REQUIRE(*copy[1] == 2);
  }

  SECTION("can copy assign") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = {{"c", 1}};
    REQUIRE(copy.find("c") != nullptr);
    copy = dict;
    REQUIRE(copy.size() == 2);
    REQUIRE(*copy[0] == 1);
    REQUIRE(*copy[1] == 2);
    REQUIRE(copy.find("c") == nullptr);
  }

  SECTION("can move construct") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = std::move(dict);
    REQUIRE(copy.size() == 2);
    REQUIRE(*copy[0] == 1);
    REQUIRE(*copy[1] == 2);
  }

  SECTION("can move assign") {
    OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
    OrderedDict<int> copy = {{"c", 1}};
    REQUIRE(copy.find("c") != nullptr);
    copy = std::move(dict);
    REQUIRE(copy.size() == 2);
    REQUIRE(*copy[0] == 1);
    REQUIRE(*copy[1] == 2);
    REQUIRE(copy.find("c") == nullptr);
  }

  SECTION("can insert with braces") {
    OrderedDict<std::pair<int, int>> dict;
    dict.insert("a", {1, 2});
    REQUIRE(!dict.is_empty());
    REQUIRE(dict["a"].first == 1);
    REQUIRE(dict["a"].second == 2);
  }

  SECTION("Error messages include the what") {
    OrderedDict<int> dict("Penguin");
    REQUIRE(dict.subject() == "Penguin");
    dict.insert("a", 1);
    REQUIRE(!dict.is_empty());
    REQUIRE_THROWS_WITH(
        dict.get("b"), StartsWith("Penguin 'b' is not defined"));
    REQUIRE_THROWS_WITH(
        dict.insert("a", 1), StartsWith("Penguin 'a' already defined"));
  }
}
