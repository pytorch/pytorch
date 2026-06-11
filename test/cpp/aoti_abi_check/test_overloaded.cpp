#include <gtest/gtest.h>

#include <torch/headeronly/util/overloaded.h>

#include <string>
#include <variant>

TEST(TestOverloaded, TestOverloaded) {
  auto visitor = torch::headeronly::overloaded(
      [](int) { return std::string("int"); },
      [](double) { return std::string("double"); });
  std::variant<int, double> v = 3;
  EXPECT_EQ(std::visit(visitor, v), "int");
  v = 2.0;
  EXPECT_EQ(std::visit(visitor, v), "double");

  // c10 alias
  auto single = c10::overloaded([](int x) { return x; });
  EXPECT_EQ(single(5), 5);
}
