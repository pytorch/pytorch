#include "catch_utils.hpp"

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

CATCH_TEST_CASE("static") {
  CATCH_SECTION("all_of") {
    CATCH_REQUIRE(torch::all_of<>::value == true);
    CATCH_REQUIRE(torch::all_of<true>::value == true);
    CATCH_REQUIRE(torch::all_of<true, true, true>::value == true);
    CATCH_REQUIRE(torch::all_of<false>::value == false);
    CATCH_REQUIRE(torch::all_of<false, false, false>::value == false);
    CATCH_REQUIRE(torch::all_of<true, true, false>::value == false);
  }
  CATCH_SECTION("any_of") {
    CATCH_REQUIRE(torch::any_of<>::value == false);
    CATCH_REQUIRE(torch::any_of<true>::value == true);
    CATCH_REQUIRE(torch::any_of<true, true, true>::value == true);
    CATCH_REQUIRE(torch::any_of<false>::value == false);
    CATCH_REQUIRE(torch::any_of<true, true, false>::value == true);
  }
  CATCH_SECTION("enable_if_module_t") {
    CATCH_REQUIRE(f(torch::nn::LinearImpl(1, 2)) == true);
    CATCH_REQUIRE(f(5) == false);
  }
  CATCH_SECTION("check_not_lvalue_references") {
    CATCH_REQUIRE(torch::detail::check_not_lvalue_references<int>() == true);
    CATCH_REQUIRE(
        torch::detail::check_not_lvalue_references<float, int, char>() == true);
    CATCH_REQUIRE(
        torch::detail::check_not_lvalue_references<float, int&, char>() ==
        false);
    CATCH_REQUIRE(torch::detail::check_not_lvalue_references<std::string>() == true);
    CATCH_REQUIRE(
        torch::detail::check_not_lvalue_references<std::string&>() == false);
  }
  CATCH_SECTION("apply") {
    std::vector<int> v;
    torch::apply([&v](int x) { v.push_back(x); }, 1, 2, 3, 4, 5);
    CATCH_REQUIRE(v.size() == 5);
    for (size_t i = 0; i < v.size(); ++i) {
      CATCH_REQUIRE(v.at(i) == 1 + i);
    }
  }
}
