#include <gtest/gtest.h>

#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/any.h>
#include <torch/nn/modules/linear.h>

#include <torch/csrc/utils/variadic.h>

#include <string>
#include <vector>

template <
    typename T,
    typename = torch::enable_if_t<!torch::detail::is_module<T>::value>>
bool f(T&& m) {
  return false;
}

template <typename T>
torch::detail::enable_if_module_t<T, bool> f(T&& m) {
  return true;
}

TEST(TestStatic, AllOf) {
  ASSERT_TRUE(torch::all_of<>::value);
  ASSERT_TRUE(torch::all_of<true>::value);
  ASSERT_TRUE((torch::all_of<true, true, true>::value));
  ASSERT_FALSE(torch::all_of<false>::value);
  ASSERT_FALSE((torch::all_of<false, false, false>::value));
  ASSERT_FALSE((torch::all_of<true, true, false>::value));
}

TEST(TestStatic, AnyOf) {
  ASSERT_FALSE(torch::any_of<>::value);
  ASSERT_TRUE(bool((torch::any_of<true>::value)));
  ASSERT_TRUE(bool((torch::any_of<true, true, true>::value)));
  ASSERT_FALSE(bool((torch::any_of<false>::value)));
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

TEST(TestStatic, Apply) {
  std::vector<int> v;
  torch::apply([&v](int x) { v.push_back(x); }, 1, 2, 3, 4, 5);
  ASSERT_EQ(v.size(), 5);
  for (size_t i = 0; i < v.size(); ++i) {
    ASSERT_EQ(v.at(i), i + 1);
  }
}
