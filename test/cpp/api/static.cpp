
#include "gtest/gtest.h"

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

TEST(TestStatic, All_Of){
  EXPECT_TRUE(torch::all_of<>::value);
  EXPECT_TRUE(torch::all_of<true>::value);
  EXPECT_TRUE((torch::all_of<true, true, true>::value));
  EXPECT_FALSE(torch::all_of<false>::value);
  EXPECT_FALSE((torch::all_of<false, false, false>::value));
  EXPECT_FALSE((torch::all_of<true, true, false>::value));
}
TEST(TestStatic, Any_Of){
  EXPECT_FALSE(torch::any_of<>::value);
  EXPECT_TRUE(bool((torch::any_of<true>::value)));
  EXPECT_TRUE(bool((torch::any_of<true, true, true>::value)));
  EXPECT_FALSE(bool((torch::any_of<false>::value)));
}
TEST(TestStatic, Enable_If_Module){
  EXPECT_TRUE(f(torch::nn::LinearImpl(1, 2)));
  EXPECT_FALSE(f(5));
  EXPECT_TRUE(torch::detail::check_not_lvalue_references<int>());
  EXPECT_TRUE((torch::detail::check_not_lvalue_references<float, int, char>()));
  EXPECT_FALSE(
      (torch::detail::check_not_lvalue_references<float, int&, char>()));
  EXPECT_TRUE(torch::detail::check_not_lvalue_references<std::string>());
  EXPECT_FALSE(torch::detail::check_not_lvalue_references<std::string&>());
}
TEST(TestStatic, Apply){
  std::vector<int> v;
  torch::apply([&v](int x) { v.push_back(x); }, 1, 2, 3, 4, 5);
  EXPECT_EQ(v.size(), 5);
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v.at(i), i + 1);
  }
}
