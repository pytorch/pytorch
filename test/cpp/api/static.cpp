
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

// TEST_CASE("static") {
TEST(static, static){
  EXPECT_EQ(torch::all_of<>::value, true);
  EXPECT_EQ(torch::all_of<true>::value, true);
  // EXPECT_EQ(torch::all_of<true, true, true>::value, true);
  EXPECT_EQ(torch::all_of<false>::value, false);
  // EXPECT_EQ(torch::all_of<false, false, false>::value, false);
  // EXPECT_EQ(torch::all_of<true, true, false>::value, false);
  EXPECT_EQ(torch::any_of<>::value, false);
  // EXPECT_EQ(torch::any_of<true>::value, true);
  // EXPECT_EQ(torch::any_of<true, true, true>::value, true);
  // EXPECT_EQ(torch::any_of<false>::value, false);
  EXPECT_EQ(f(torch::nn::LinearImpl(1, 2)), true);
  EXPECT_EQ(f(5), false);
  EXPECT_EQ(torch::detail::check_not_lvalue_references<int>(), true);
  // EXPECT_EQ(torch::detail::check_not_lvalue_references<float, int, char>(),
  // true); EXPECT_EQ(torch::detail::check_not_lvalue_references<float, int&,
  // char>(), false);
  EXPECT_EQ(torch::detail::check_not_lvalue_references<std::string>(), true);
  EXPECT_EQ(torch::detail::check_not_lvalue_references<std::string&>(), false);

  std::vector<int> v;
  torch::apply([&v](int x) { v.push_back(x); }, 1, 2, 3, 4, 5);
  EXPECT_EQ(v.size(), 5);
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v.at(i), i + 1);
  }
}
