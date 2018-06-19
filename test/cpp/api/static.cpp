#include <catch.hpp>

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

TEST_CASE("static") {
  SECTION("all_of") {
    REQUIRE(torch::all_of<>::value == true);
    REQUIRE(torch::all_of<true>::value == true);
    REQUIRE(torch::all_of<true, true, true>::value == true);
    REQUIRE(torch::all_of<false>::value == false);
    REQUIRE(torch::all_of<false, false, false>::value == false);
    REQUIRE(torch::all_of<true, true, false>::value == false);
  }
  SECTION("any_of") {
    REQUIRE(torch::any_of<>::value == false);
    REQUIRE(torch::any_of<true>::value == true);
    REQUIRE(torch::any_of<true, true, true>::value == true);
    REQUIRE(torch::any_of<false>::value == false);
    REQUIRE(torch::any_of<true, true, false>::value == true);
  }
  SECTION("enable_if_module_t") {
    REQUIRE(f(torch::nn::LinearImpl({1, 2})) == true);
    REQUIRE(f(5) == false);
  }
  SECTION("check_not_lvalue_references") {
    REQUIRE(torch::detail::check_not_lvalue_references<int>() == true);
    REQUIRE(
        torch::detail::check_not_lvalue_references<float, int, char>() == true);
    REQUIRE(
        torch::detail::check_not_lvalue_references<float, int&, char>() ==
        false);
    REQUIRE(torch::detail::check_not_lvalue_references<std::string>() == true);
    REQUIRE(
        torch::detail::check_not_lvalue_references<std::string&>() == false);
  }
  SECTION("apply") {
    std::vector<int> v;
    torch::apply([&v](int x) { v.push_back(x); }, 1, 2, 3, 4, 5);
    REQUIRE(v.size() == 5);
    for (size_t i = 0; i < v.size(); ++i) {
      REQUIRE(v.at(i) == 1 + i);
    }
  }
}
