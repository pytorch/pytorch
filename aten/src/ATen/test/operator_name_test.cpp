#include <gtest/gtest.h>

#include <ATen/core/operator_name.h>

TEST(OperatorNameTest, SetNamespaceIfNotSetWithoutExistingNamespace) {
  c10::OperatorName testName("operator", "operator.overload");

  const auto result = testName.setNamespaceIfNotSet("ns");
  EXPECT_TRUE(result);
  EXPECT_EQ(testName.name, "ns::operator");
  EXPECT_EQ(testName.overload_name, "operator.overload");
  EXPECT_EQ(testName.getNamespace(), std::optional<std::string_view>("ns"));
}

TEST(OperatorNameTest, SetNamespaceIfNotSetWithExistingNamespace) {
  c10::OperatorName namespacedName("already_namespaced::operator", "operator.overload");
  const auto result = namespacedName.setNamespaceIfNotSet("namespace");
  EXPECT_FALSE(result);
  EXPECT_EQ(namespacedName.name, "already_namespaced::operator");
  EXPECT_EQ(namespacedName.overload_name, "operator.overload");
  EXPECT_EQ(namespacedName.getNamespace(), std::optional<std::string_view>("already_namespaced"));
}
