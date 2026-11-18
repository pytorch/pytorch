#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/csrc/utils/variadic.h>
#include <torch/detail/static.h>
#include <torch/torch.h>

#include <string>
#include <type_traits>
#include <vector>

template <
    typename T,
    typename = std::enable_if_t<!torch::detail::is_module<T>::value>>
bool f(T&& m) {
  return false;
}

template <typename T>
torch::detail::enable_if_module_t<T, bool> f(T&& m) {
  return true;
}

TEST(TestStatic, EnableIfModule) {
  ASSERT_TRUE(f(torch::nn::LinearImpl(1, 2)));
  ASSERT_FALSE(f(5));
  ASSERT_TRUE(torch::detail::check_not_lvalue_references<int>());
  ASSERT_TRUE((torch::detail::check_not_lvalue_references<float, int, char>()));
  ASSERT_FALSE(
      (torch::detail::check_not_lvalue_references<float, int&, char>()));
  ASSERT_TRUE(torch::detail::check_not_lvalue_references<std::string>());
  ASSERT_FALSE(torch::detail::check_not_lvalue_references<std::string&>());
}

namespace {

struct A : torch::nn::Module {
  int forward() {
    return 5;
  }
};

struct B : torch::nn::Module {
  std::string forward(torch::Tensor tensor) {
    return "";
  }
};

struct C : torch::nn::Module {
  float forward(torch::Tensor& tensor) {
    return 5.0;
  }
};

struct D : torch::nn::Module {
  char forward(torch::Tensor&& tensor) {
    return 'x';
  }
};

struct E : torch::nn::Module {};

} // anonymous namespace

// Put in a function because macros don't handle the comma between arguments to
// is_same well ...
template <typename Module, typename ExpectedType, typename... Args>
void assert_has_expected_type() {
  using ReturnType =
      typename torch::detail::return_type_of_forward<Module, Args...>::type;
  constexpr bool is_expected_type = std::is_same_v<ReturnType, ExpectedType>;
  ASSERT_TRUE(is_expected_type) << Module().name();
}

TEST(TestStatic, ReturnTypeOfForward) {
  assert_has_expected_type<A, int>();
  assert_has_expected_type<B, std::string, torch::Tensor>();
  assert_has_expected_type<C, float, torch::Tensor&>();
  assert_has_expected_type<D, char, torch::Tensor&&>();
  assert_has_expected_type<E, void>();
}

TEST(TestStatic, Apply) {
  std::vector<int> v;
  torch::apply([&v](int x) { v.push_back(x); }, 1, 2, 3, 4, 5);
  ASSERT_EQ(v.size(), 5);
  for (const auto i : c10::irange(v.size())) {
    ASSERT_EQ(v.at(i), i + 1);
  }
}
