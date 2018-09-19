#include "catch_utils.hpp"

#include <torch/nn/modules/any.h>
#include <torch/torch.h>
#include <torch/utils.h>

#include <algorithm>
#include <string>

using namespace torch::nn;
using namespace torch::detail;

using Catch::Contains;
using Catch::StartsWith;

CATCH_TEST_CASE("any-module") {
  torch::manual_seed(0);
  CATCH_SECTION("int()") {
    struct M : torch::nn::Module {
      int forward() {
        return 123;
      }
    };
    AnyModule any(M{});
    CATCH_REQUIRE(any.forward<int>() == 123);
  }

  CATCH_SECTION("int(int)") {
    struct M : torch::nn::Module {
      int forward(int x) {
        return x;
      }
    };
    AnyModule any(M{});
    CATCH_REQUIRE(any.forward<int>(5) == 5);
  }

  CATCH_SECTION("const char*(const char*)") {
    struct M : torch::nn::Module {
      const char* forward(const char* x) {
        return x;
      }
    };
    AnyModule any(M{});
    CATCH_REQUIRE(any.forward<const char*>("hello") == std::string("hello"));
  }

  CATCH_SECTION("string(int, const double)") {
    struct M : torch::nn::Module {
      std::string forward(int x, const double f) {
        return std::to_string(static_cast<int>(x + f));
      }
    };
    AnyModule any(M{});
    int x = 4;
    CATCH_REQUIRE(any.forward<std::string>(x, 3.14) == std::string("7"));
  }

  CATCH_SECTION("Tensor(string, const string&, string&&)") {
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
    CATCH_REQUIRE(
        any.forward(
               std::string("a"), std::string("ab"), std::string("abc"))
            .sum()
            .toCInt() == 6);
  }
  CATCH_SECTION("wrong argument type") {
    struct M : torch::nn::Module {
      int forward(float x) {
        return x;
      }
    };
    AnyModule any(M{});
    CATCH_REQUIRE_THROWS_WITH(
        any.forward(5.0),
        StartsWith("Expected argument #0 to be of type float, "
                   "but received value of type double"));
  }
  CATCH_SECTION("wrong number of arguments") {
    struct M : torch::nn::Module {
      int forward(int a, int b) {
        return a + b;
      }
    };
    AnyModule any(M{});
    CATCH_REQUIRE_THROWS_WITH(
        any.forward(),
        Contains("M's forward() method expects 2 arguments, but received 0"));
    CATCH_REQUIRE_THROWS_WITH(
        any.forward(5),
        Contains("M's forward() method expects 2 arguments, but received 1"));
    CATCH_REQUIRE_THROWS_WITH(
        any.forward(1, 2, 3),
        Contains("M's forward() method expects 2 arguments, but received 3"));
  }
  CATCH_SECTION("get()") {
    struct M : torch::nn::Module {
      explicit M(int value_) : torch::nn::Module("M"), value(value_) {}
      int value;
      int forward(float x) {
        return x;
      }
    };
    AnyModule any(M{5});

    CATCH_SECTION("good cast") {
      CATCH_REQUIRE(any.get<M>().value == 5);
    }

    CATCH_SECTION("bad cast") {
      struct N : torch::nn::Module {};
      CATCH_REQUIRE_THROWS_WITH(any.get<N>(), StartsWith("Attempted to cast module"));
    }
  }
  CATCH_SECTION("ptr()") {
    struct M : torch::nn::Module {
      explicit M(int value_) : torch::nn::Module("M"), value(value_) {}
      int value;
      int forward(float x) {
        return x;
      }
    };
    AnyModule any(M{5});

    CATCH_SECTION("base class cast") {
      auto ptr = any.ptr();
      CATCH_REQUIRE(ptr != nullptr);
      CATCH_REQUIRE(ptr->name() == "M");
    }

    CATCH_SECTION("good downcast") {
      auto ptr = any.ptr<M>();
      CATCH_REQUIRE(ptr != nullptr);
      CATCH_REQUIRE(ptr->value == 5);
    }

    CATCH_SECTION("bad downcast") {
      struct N : torch::nn::Module {};
      CATCH_REQUIRE_THROWS_WITH(any.ptr<N>(), StartsWith("Attempted to cast module"));
    }
  }
  CATCH_SECTION("default state is empty") {
    struct M : torch::nn::Module {
      explicit M(int value_) : value(value_) {}
      int value;
      int forward(float x) {
        return x;
      }
    };
    AnyModule any;
    CATCH_REQUIRE(any.is_empty());
    any = std::make_shared<M>(5);
    CATCH_REQUIRE(!any.is_empty());
    CATCH_REQUIRE(any.get<M>().value == 5);
  }
  CATCH_SECTION("all methods throw for empty AnyModule") {
    struct M : torch::nn::Module {
      int forward(int x) {
        return x;
      }
    };
    AnyModule any;
    CATCH_REQUIRE(any.is_empty());
    CATCH_REQUIRE_THROWS_WITH(
        any.get<M>(), StartsWith("Cannot call get() on an empty AnyModule"));
    CATCH_REQUIRE_THROWS_WITH(
        any.ptr<M>(), StartsWith("Cannot call ptr() on an empty AnyModule"));
    CATCH_REQUIRE_THROWS_WITH(
        any.ptr(), StartsWith("Cannot call ptr() on an empty AnyModule"));
    CATCH_REQUIRE_THROWS_WITH(
        any.type_info(),
        StartsWith("Cannot call type_info() on an empty AnyModule"));
    CATCH_REQUIRE_THROWS_WITH(
        any.forward<int>(5),
        StartsWith("Cannot call forward() on an empty AnyModule"));
  }
  CATCH_SECTION("can move assign different modules") {
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
    CATCH_REQUIRE(any.is_empty());
    any = std::make_shared<M>();
    CATCH_REQUIRE(!any.is_empty());
    CATCH_REQUIRE(any.forward<std::string>(5) == "5");
    any = std::make_shared<N>();
    CATCH_REQUIRE(!any.is_empty());
    CATCH_REQUIRE(any.forward<int>(5.0f) == 8);
  }
  CATCH_SECTION("constructs from ModuleHolder") {
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
    CATCH_REQUIRE(any.get<MImpl>().value == 5);
    CATCH_REQUIRE(any.get<M>()->value == 5);

    AnyModule module(Linear(3, 4));
    std::shared_ptr<Module> ptr = module.ptr();
    Linear linear(module.get<Linear>());
  }
  CATCH_SECTION("converts autograd::Variable to torch::Tensor correctly") {
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
      CATCH_REQUIRE(
          any.forward(torch::autograd::Variable(torch::ones(5)))
              .sum()
              .toCFloat() == 5);
      // at::Tensors that are not variables work too.
      CATCH_REQUIRE(any.forward(at::ones(5)).sum().toCFloat() == 5);
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

CATCH_TEST_CASE("any-value") {
  torch::manual_seed(0);
  CATCH_SECTION("gets the correct value for the right type") {
    CATCH_SECTION("int") {
      auto value = make_value(5);
      // const and non-const types have the same typeid()
      CATCH_REQUIRE(value.try_get<int>() != nullptr);
      CATCH_REQUIRE(value.try_get<const int>() != nullptr);
      CATCH_REQUIRE(value.get<int>() == 5);
    }
    CATCH_SECTION("const int") {
      auto value = make_value(5);
      CATCH_REQUIRE(value.try_get<const int>() != nullptr);
      CATCH_REQUIRE(value.try_get<int>() != nullptr);
      CATCH_REQUIRE(value.get<const int>() == 5);
    }
    CATCH_SECTION("const char*") {
      auto value = make_value("hello");
      CATCH_REQUIRE(value.try_get<const char*>() != nullptr);
      CATCH_REQUIRE(value.get<const char*>() == std::string("hello"));
    }
    CATCH_SECTION("std::string") {
      auto value = make_value(std::string("hello"));
      CATCH_REQUIRE(value.try_get<std::string>() != nullptr);
      CATCH_REQUIRE(value.get<std::string>() == "hello");
    }
    CATCH_SECTION("pointers") {
      std::string s("hello");
      std::string* p = &s;
      auto value = make_value(p);
      CATCH_REQUIRE(value.try_get<std::string*>() != nullptr);
      CATCH_REQUIRE(*value.get<std::string*>() == "hello");
    }
    CATCH_SECTION("references") {
      std::string s("hello");
      const std::string& t = s;
      auto value = make_value(t);
      CATCH_REQUIRE(value.try_get<std::string>() != nullptr);
      CATCH_REQUIRE(value.get<std::string>() == "hello");
    }
  }
  CATCH_SECTION("try_get returns nullptr for the wrong type") {
    auto value = make_value(5);
    CATCH_REQUIRE(value.try_get<int>() != nullptr);
    CATCH_REQUIRE(value.try_get<float>() == nullptr);
    CATCH_REQUIRE(value.try_get<long>() == nullptr);
    CATCH_REQUIRE(value.try_get<std::string>() == nullptr);
  }
  CATCH_SECTION("get throws for the wrong type") {
    auto value = make_value(5);
    CATCH_REQUIRE(value.try_get<int>() != nullptr);
    CATCH_REQUIRE_THROWS_WITH(
        value.get<float>(),
        StartsWith("Attempted to cast Value to float, "
                   "but its actual type is int"));
    CATCH_REQUIRE_THROWS_WITH(
        value.get<long>(),
        StartsWith("Attempted to cast Value to long, "
                   "but its actual type is int"));
  }
  CATCH_SECTION("move is allowed") {
    auto value = make_value(5);
    CATCH_SECTION("construction") {
      auto copy = make_value(std::move(value));
      CATCH_REQUIRE(copy.try_get<int>() != nullptr);
      CATCH_REQUIRE(copy.get<int>() == 5);
    }
    CATCH_SECTION("assignment") {
      auto copy = make_value(10);
      copy = std::move(value);
      CATCH_REQUIRE(copy.try_get<int>() != nullptr);
      CATCH_REQUIRE(copy.get<int>() == 5);
    }
  }
  CATCH_SECTION("type_info is correct") {
    CATCH_SECTION("int") {
      auto value = make_value(5);
      CATCH_REQUIRE(value.type_info().hash_code() == typeid(int).hash_code());
    }
    CATCH_SECTION("const char") {
      auto value = make_value("hello");
      CATCH_REQUIRE(value.type_info().hash_code() == typeid(const char*).hash_code());
    }
    CATCH_SECTION("std::string") {
      auto value = make_value(std::string("hello"));
      CATCH_REQUIRE(value.type_info().hash_code() == typeid(std::string).hash_code());
    }
  }
}
