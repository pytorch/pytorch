#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <algorithm>
#include <string>

using namespace torch::nn;
using namespace torch::detail;

struct AnyModuleTest : torch::test::SeedingFixture {};

TEST_F(AnyModuleTest, SimpleReturnType) {
  struct M : torch::nn::Module {
    int forward() {
      return 123;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<int>(), 123);
}

TEST_F(AnyModuleTest, SimpleReturnTypeAndSingleArgument) {
  struct M : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<int>(5), 5);
}

TEST_F(AnyModuleTest, StringLiteralReturnTypeAndArgument) {
  struct M : torch::nn::Module {
    const char* forward(const char* x) {
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<const char*>("hello"), std::string("hello"));
}

TEST_F(AnyModuleTest, StringReturnTypeWithConstArgument) {
  struct M : torch::nn::Module {
    std::string forward(int x, const double f) {
      return std::to_string(static_cast<int>(x + f));
    }
  };
  AnyModule any(M{});
  int x = 4;
  ASSERT_EQ(any.forward<std::string>(x, 3.14), std::string("7"));
}

TEST_F(
    AnyModuleTest,
    TensorReturnTypeAndStringArgumentsWithFunkyQualifications) {
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
  ASSERT_TRUE(
      any.forward(std::string("a"), std::string("ab"), std::string("abc"))
          .sum()
          .item<int32_t>() == 6);
}

TEST_F(AnyModuleTest, WrongArgumentType) {
  struct M : torch::nn::Module {
    int forward(float x) {
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_THROWS_WITH(
      any.forward(5.0),
      "Expected argument #0 to be of type float, "
      "but received value of type double");
}

TEST_F(AnyModuleTest, WrongNumberOfArguments) {
  struct M : torch::nn::Module {
    int forward(int a, int b) {
      return a + b;
    }
  };
  AnyModule any(M{});
  ASSERT_THROWS_WITH(
      any.forward(),
      "M's forward() method expects 2 arguments, but received 0");
  ASSERT_THROWS_WITH(
      any.forward(5),
      "M's forward() method expects 2 arguments, but received 1");
  ASSERT_THROWS_WITH(
      any.forward(1, 2, 3),
      "M's forward() method expects 2 arguments, but received 3");
}

struct M : torch::nn::Module {
  explicit M(int value_) : torch::nn::Module("M"), value(value_) {}
  int value;
  int forward(float x) {
    return x;
  }
};

TEST_F(AnyModuleTest, GetWithCorrectTypeSucceeds) {
  AnyModule any(M{5});
  ASSERT_EQ(any.get<M>().value, 5);
}

TEST_F(AnyModuleTest, GetWithIncorrectTypeThrows) {
  struct N : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return input;
    }
  };
  AnyModule any(M{5});
  ASSERT_THROWS_WITH(any.get<N>(), "Attempted to cast module");
}

TEST_F(AnyModuleTest, PtrWithBaseClassSucceeds) {
  AnyModule any(M{5});
  auto ptr = any.ptr();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->name(), "M");
}

TEST_F(AnyModuleTest, PtrWithGoodDowncastSuccceeds) {
  AnyModule any(M{5});
  auto ptr = any.ptr<M>();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->value, 5);
}

TEST_F(AnyModuleTest, PtrWithBadDowncastThrows) {
  struct N : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return input;
    }
  };
  AnyModule any(M{5});
  ASSERT_THROWS_WITH(any.ptr<N>(), "Attempted to cast module");
}

TEST_F(AnyModuleTest, DefaultStateIsEmpty) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
    int forward(float x) {
      return x;
    }
  };
  AnyModule any;
  ASSERT_TRUE(any.is_empty());
  any = std::make_shared<M>(5);
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.get<M>().value, 5);
}

TEST_F(AnyModuleTest, AllMethodsThrowForEmptyAnyModule) {
  struct M : torch::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  AnyModule any;
  ASSERT_TRUE(any.is_empty());
  ASSERT_THROWS_WITH(any.get<M>(), "Cannot call get() on an empty AnyModule");
  ASSERT_THROWS_WITH(any.ptr<M>(), "Cannot call ptr() on an empty AnyModule");
  ASSERT_THROWS_WITH(any.ptr(), "Cannot call ptr() on an empty AnyModule");
  ASSERT_THROWS_WITH(
      any.type_info(), "Cannot call type_info() on an empty AnyModule");
  ASSERT_THROWS_WITH(
      any.forward<int>(5), "Cannot call forward() on an empty AnyModule");
}

TEST_F(AnyModuleTest, CanMoveAssignDifferentModules) {
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
  ASSERT_TRUE(any.is_empty());
  any = std::make_shared<M>();
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.forward<std::string>(5), "5");
  any = std::make_shared<N>();
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.forward<int>(5.0f), 8);
}

TEST_F(AnyModuleTest, ConstructsFromModuleHolder) {
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
  ASSERT_EQ(any.get<MImpl>().value, 5);
  ASSERT_EQ(any.get<M>()->value, 5);

  AnyModule module(Linear(3, 4));
  std::shared_ptr<Module> ptr = module.ptr();
  Linear linear(module.get<Linear>());
}

TEST_F(AnyModuleTest, ConvertsVariableToTensorCorrectly) {
  struct M : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return input;
    }
  };

  // When you have an autograd::Variable, it should be converted to a
  // torch::Tensor before being passed to the function (to avoid a type
  // mismatch).
  AnyModule any(M{});
  ASSERT_TRUE(
      any.forward(torch::autograd::Variable(torch::ones(5)))
          .sum()
          .item<float>() == 5);
  // at::Tensors that are not variables work too.
  ASSERT_EQ(any.forward(at::ones(5)).sum().item<float>(), 5);
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

struct AnyValueTest : torch::test::SeedingFixture {};

TEST_F(AnyValueTest, CorrectlyAccessesIntWhenCorrectType) {
  auto value = make_value(5);
  // const and non-const types have the same typeid()
  ASSERT_NE(value.try_get<int>(), nullptr);
  ASSERT_NE(value.try_get<const int>(), nullptr);
  ASSERT_EQ(value.get<int>(), 5);
}
TEST_F(AnyValueTest, CorrectlyAccessesConstIntWhenCorrectType) {
  auto value = make_value(5);
  ASSERT_NE(value.try_get<const int>(), nullptr);
  ASSERT_NE(value.try_get<int>(), nullptr);
  ASSERT_EQ(value.get<const int>(), 5);
}
TEST_F(AnyValueTest, CorrectlyAccessesStringLiteralWhenCorrectType) {
  auto value = make_value("hello");
  ASSERT_NE(value.try_get<const char*>(), nullptr);
  ASSERT_EQ(value.get<const char*>(), std::string("hello"));
}
TEST_F(AnyValueTest, CorrectlyAccessesStringWhenCorrectType) {
  auto value = make_value(std::string("hello"));
  ASSERT_NE(value.try_get<std::string>(), nullptr);
  ASSERT_EQ(value.get<std::string>(), "hello");
}
TEST_F(AnyValueTest, CorrectlyAccessesPointersWhenCorrectType) {
  std::string s("hello");
  std::string* p = &s;
  auto value = make_value(p);
  ASSERT_NE(value.try_get<std::string*>(), nullptr);
  ASSERT_EQ(*value.get<std::string*>(), "hello");
}
TEST_F(AnyValueTest, CorrectlyAccessesReferencesWhenCorrectType) {
  std::string s("hello");
  const std::string& t = s;
  auto value = make_value(t);
  ASSERT_NE(value.try_get<std::string>(), nullptr);
  ASSERT_EQ(value.get<std::string>(), "hello");
}

TEST_F(AnyValueTest, TryGetReturnsNullptrForTheWrongType) {
  auto value = make_value(5);
  ASSERT_NE(value.try_get<int>(), nullptr);
  ASSERT_EQ(value.try_get<float>(), nullptr);
  ASSERT_EQ(value.try_get<long>(), nullptr);
  ASSERT_EQ(value.try_get<std::string>(), nullptr);
}

TEST_F(AnyValueTest, GetThrowsForTheWrongType) {
  auto value = make_value(5);
  ASSERT_NE(value.try_get<int>(), nullptr);
  ASSERT_THROWS_WITH(
      value.get<float>(),
      "Attempted to cast Value to float, "
      "but its actual type is int");
  ASSERT_THROWS_WITH(
      value.get<long>(),
      "Attempted to cast Value to long, "
      "but its actual type is int");
}

TEST_F(AnyValueTest, MoveConstructionIsAllowed) {
  auto value = make_value(5);
  auto copy = make_value(std::move(value));
  ASSERT_NE(copy.try_get<int>(), nullptr);
  ASSERT_EQ(copy.get<int>(), 5);
}

TEST_F(AnyValueTest, MoveAssignmentIsAllowed) {
  auto value = make_value(5);
  auto copy = make_value(10);
  copy = std::move(value);
  ASSERT_NE(copy.try_get<int>(), nullptr);
  ASSERT_EQ(copy.get<int>(), 5);
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForInt) {
  auto value = make_value(5);
  ASSERT_EQ(value.type_info().hash_code(), typeid(int).hash_code());
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForStringLiteral) {
  auto value = make_value("hello");
  ASSERT_EQ(value.type_info().hash_code(), typeid(const char*).hash_code());
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForString) {
  auto value = make_value(std::string("hello"));
  ASSERT_EQ(value.type_info().hash_code(), typeid(std::string).hash_code());
}
