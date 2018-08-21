#include <catch.hpp>

#include <torch/nn/modules/any.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <algorithm>
#include <string>

using namespace torch::nn;
using namespace torch::detail;

using Catch::Contains;
using Catch::StartsWith;

TEST_CASE("any-module") {
  torch::manual_seed(0);
  SECTION("int()") {
    struct M : torch::nn::Module {
      int forward() {
        return 123;
      }
    };
    AnyModule any(M{});
    REQUIRE(any.forward().get<int>() == 123);
  }
  SECTION("int(int)") {
    struct M : torch::nn::Module {
      int forward(int x) {
        return x;
      }
    };
    AnyModule any(M{});
    REQUIRE(any.forward(5).get<int>() == 5);
  }
  SECTION("const char*(const char*)") {
    struct M : torch::nn::Module {
      const char* forward(const char* x) {
        return x;
      }
    };
    AnyModule any(M{});
    REQUIRE(any.forward("hello").get<const char*>() == std::string("hello"));
  }

  SECTION("string(int, const double)") {
    struct M : torch::nn::Module {
      std::string forward(int x, const double f) {
        return std::to_string(static_cast<int>(x + f));
      }
    };
    AnyModule any(M{});
    int x = 4;
    REQUIRE(any.forward(x, 3.14).get<std::string>() == std::string("7"));
  }

  SECTION("Tensor(string, const string&, string&&)") {
    struct M : torch::nn::Module {
      torch::Tensor forward(
          std::string a,
          const std::string& b,
          std::string&& c) {
        const auto s = a + b + c;
        return torch::ones({static_cast<int64_t>(s.size())});
      }
    };
    AnyModule any(M{});
    REQUIRE(
        any.forward(std::string("a"), std::string("ab"), std::string("abc"))
            .get<torch::Tensor>()
            .sum()
            .toCInt() == 6);
  }
  SECTION("wrong argument type") {
    struct M : torch::nn::Module {
      int forward(float x) {
        return x;
      }
    };
    AnyModule any(M{});
    REQUIRE_THROWS_WITH(
        any.forward(5.0),
        StartsWith("Expected argument #0 to be of type float, "
                   "but received value of type double"));
  }
  SECTION("wrong number of arguments") {
    struct M : torch::nn::Module {
      int forward(int a, int b) {
        return a + b;
      }
    };
    AnyModule any(M{});
    REQUIRE_THROWS_WITH(
        any.forward(),
        Contains("M's forward() method expects 2 arguments, but received 0"));
    REQUIRE_THROWS_WITH(
        any.forward(5),
        Contains("M's forward() method expects 2 arguments, but received 1"));
    REQUIRE_THROWS_WITH(
        any.forward(1, 2, 3),
        Contains("M's forward() method expects 2 arguments, but received 3"));
  }
  SECTION("get()") {
    struct M : torch::nn::Module {
      explicit M(int value_) : torch::nn::Module("M"), value(value_) {}
      int value;
      int forward(float x) {
        return x;
      }
    };
    AnyModule any(M{5});

    SECTION("good cast") {
      REQUIRE(any.get<M>().value == 5);
    }

    SECTION("bad cast") {
      struct N : torch::nn::Module {};
      REQUIRE_THROWS_WITH(any.get<N>(), StartsWith("Attempted to cast module"));
    }
  }
  SECTION("ptr()") {
    struct M : torch::nn::Module {
      explicit M(int value_) : torch::nn::Module("M"), value(value_) {}
      int value;
      int forward(float x) {
        return x;
      }
    };
    AnyModule any(M{5});

    SECTION("base class cast") {
      auto ptr = any.ptr();
      REQUIRE(ptr != nullptr);
      REQUIRE(ptr->name() == "M");
    }

    SECTION("good downcast") {
      auto ptr = any.ptr<M>();
      REQUIRE(ptr != nullptr);
      REQUIRE(ptr->value == 5);
    }

    SECTION("bad downcast") {
      struct N : torch::nn::Module {};
      REQUIRE_THROWS_WITH(any.ptr<N>(), StartsWith("Attempted to cast module"));
    }
  }
  SECTION("default state is empty") {
    struct M : torch::nn::Module {
      explicit M(int value_) : value(value_) {}
      int value;
      int forward(float x) {
        return x;
      }
    };
    AnyModule any;
    REQUIRE(any.is_empty());
    any = std::make_shared<M>(5);
    REQUIRE(!any.is_empty());
    REQUIRE(any.get<M>().value == 5);
  }
  SECTION("all methods throw for empty AnyModule") {
    struct M : torch::nn::Module {
      int forward(int x) {
        return x;
      }
    };
    AnyModule any;
    REQUIRE(any.is_empty());
    REQUIRE_THROWS_WITH(
        any.get<M>(), StartsWith("Cannot call get() on an empty AnyModule"));
    REQUIRE_THROWS_WITH(
        any.ptr<M>(), StartsWith("Cannot call ptr() on an empty AnyModule"));
    REQUIRE_THROWS_WITH(
        any.ptr(), StartsWith("Cannot call ptr() on an empty AnyModule"));
    REQUIRE_THROWS_WITH(
        any.type_info(),
        StartsWith("Cannot call type_info() on an empty AnyModule"));
    REQUIRE_THROWS_WITH(
        any.forward<int>(5),
        StartsWith("Cannot call forward() on an empty AnyModule"));
  }
  SECTION("can move assign differentm modules") {
    struct M : torch::nn::Module {
      std::string forward(int x) {
        return std::to_string(x);
      }
    };
    struct N : torch::nn::Module {
      int forward(float x) {
        return 3 + x;
      }
    };
    AnyModule any;
    REQUIRE(any.is_empty());
    any = std::make_shared<M>();
    REQUIRE(!any.is_empty());
    REQUIRE(any.forward(5).get<std::string>() == "5");
    any = std::make_shared<N>();
    REQUIRE(!any.is_empty());
    REQUIRE(any.forward(5.0f).get<int>() == 8);
  }
  SECTION("constructs from ModuleHolder") {
    struct MImpl : torch::nn::Module {
      explicit MImpl(int value_) : torch::nn::Module("M"), value(value_) {}
      int value;
      int forward(float x) {
        return x;
      }
    };

    struct M : torch::nn::ModuleHolder<MImpl> {
      using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
      using torch::nn::ModuleHolder<MImpl>::get;
    };

    AnyModule any(M{5});
    REQUIRE(any.get<MImpl>().value == 5);
    REQUIRE(any.get<M>()->value == 5);
  }
  SECTION("converts autograd::Variable to torch::Tensor correctly") {
    struct M : torch::nn::Module {
      torch::Tensor forward(torch::Tensor input) {
        return input;
      }
    };
    {
      // When you have an autograd::Variable, it should be converted to a
      // torch::Tensor before being passed to the function (to avoid a type
      // mismatch).
      AnyModule any(M{});
      REQUIRE(
          any.forward(torch::autograd::Variable(torch::ones(5)))
              .get<torch::Tensor>()
              .sum()
              .toCFloat() == 5);
      // at::Tensors that are not variables work too.
      REQUIRE(
          any.forward(at::ones(5)).get<torch::Tensor>().sum().toCFloat() == 5);
    }
  }
}

namespace torch {
namespace nn {
struct TestValue {
  template <typename T>
  explicit TestValue(T&& value) : value_(std::forward<T>(value)) {}
  AnyModule::Value operator()() {
    return std::move(value_);
  }
  AnyModule::Value value_;
};
template <typename T>
AnyModule::Value make_value(T&& value) {
  return TestValue(std::forward<T>(value))();
}
} // namespace nn
} // namespace torch

TEST_CASE("any-value") {
  torch::manual_seed(0);
  SECTION("gets the correct value for the right type") {
    SECTION("int") {
      auto value = make_value(5);
      // const and non-const types have the same typeid()
      REQUIRE(value.try_get<int>() != nullptr);
      REQUIRE(value.try_get<const int>() != nullptr);
      REQUIRE(value.get<int>() == 5);
    }
    SECTION("const int") {
      auto value = make_value(5);
      REQUIRE(value.try_get<const int>() != nullptr);
      REQUIRE(value.try_get<int>() != nullptr);
      REQUIRE(value.get<const int>() == 5);
    }
    SECTION("const char*") {
      auto value = make_value("hello");
      REQUIRE(value.try_get<const char*>() != nullptr);
      REQUIRE(value.get<const char*>() == std::string("hello"));
    }
    SECTION("std::string") {
      auto value = make_value(std::string("hello"));
      REQUIRE(value.try_get<std::string>() != nullptr);
      REQUIRE(value.get<std::string>() == "hello");
    }
    SECTION("pointers") {
      std::string s("hello");
      std::string* p = &s;
      auto value = make_value(p);
      REQUIRE(value.try_get<std::string*>() != nullptr);
      REQUIRE(*value.get<std::string*>() == "hello");
    }
    SECTION("references") {
      std::string s("hello");
      const std::string& t = s;
      auto value = make_value(t);
      REQUIRE(value.try_get<std::string>() != nullptr);
      REQUIRE(value.get<std::string>() == "hello");
    }
  }
  SECTION("try_get returns nullptr for the wrong type") {
    auto value = make_value(5);
    REQUIRE(value.try_get<int>() != nullptr);
    REQUIRE(value.try_get<float>() == nullptr);
    REQUIRE(value.try_get<long>() == nullptr);
    REQUIRE(value.try_get<std::string>() == nullptr);
  }
  SECTION("get throws for the wrong type") {
    auto value = make_value(5);
    REQUIRE(value.try_get<int>() != nullptr);
    REQUIRE_THROWS_WITH(
        value.get<float>(),
        StartsWith("Attempted to cast Value to float, "
                   "but its actual type is int"));
    REQUIRE_THROWS_WITH(
        value.get<long>(),
        StartsWith("Attempted to cast Value to long, "
                   "but its actual type is int"));
  }
  SECTION("move is allowed") {
    auto value = make_value(5);
    SECTION("construction") {
      auto copy = make_value(std::move(value));
      REQUIRE(copy.try_get<int>() != nullptr);
      REQUIRE(copy.get<int>() == 5);
    }
    SECTION("assignment") {
      auto copy = make_value(10);
      copy = std::move(value);
      REQUIRE(copy.try_get<int>() != nullptr);
      REQUIRE(copy.get<int>() == 5);
    }
  }
  SECTION("type_info is correct") {
    SECTION("int") {
      auto value = make_value(5);
      REQUIRE(value.type_info().hash_code() == typeid(int).hash_code());
    }
    SECTION("const char") {
      auto value = make_value("hello");
      REQUIRE(value.type_info().hash_code() == typeid(const char*).hash_code());
    }
    SECTION("std::string") {
      auto value = make_value(std::string("hello"));
      REQUIRE(value.type_info().hash_code() == typeid(std::string).hash_code());
    }
  }
}
