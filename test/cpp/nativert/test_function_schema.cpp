#include <gtest/gtest.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/nativert/executor/memory/FunctionSchema.h>

using namespace ::testing;

int64_t increment_kernel(const at::Tensor& tensor, int64_t input) {
  return input + 1;
}

at::Tensor slice_kernel(const at::Tensor& tensor, int64_t dim) {
  return tensor.slice(dim);
}

TEST(TestFunctionSchema, testNoAlias) {
  auto registrar = c10::RegisterOperators().op(
      "_test::my_op(Tensor dummy, int input) -> int", &increment_kernel);
  auto handle = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});

  EXPECT_TRUE(handle.has_value());
  EXPECT_TRUE(handle->hasSchema());

  auto nativert_schema = torch::nativert::FunctionSchema(handle->schema());

  EXPECT_FALSE(nativert_schema.alias(0, 0));
  EXPECT_FALSE(nativert_schema.alias(1, 0));

  // bounds check
  EXPECT_THROW(nativert_schema.alias(2, 0), c10::Error);
  EXPECT_THROW(nativert_schema.alias(1, 1), c10::Error);
}

TEST(TestFunctionSchema, testAliasOverride) {
  auto registrar = c10::RegisterOperators().op(
      "_test::my_op(Tensor dummy, int input) -> int", &increment_kernel);
  auto handle = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});

  EXPECT_TRUE(handle.has_value());
  EXPECT_TRUE(handle->hasSchema());

  auto nativert_schema =
      torch::nativert::FunctionSchema(handle->schema(), {{0, 0}});

  EXPECT_TRUE(nativert_schema.alias(0, 0));
  EXPECT_FALSE(nativert_schema.alias(1, 0));

  // bounds check
  EXPECT_THROW(nativert_schema.alias(2, 0), c10::Error);
  EXPECT_THROW(nativert_schema.alias(1, 1), c10::Error);
}

TEST(TestFunctionSchema, testAlias) {
  auto registrar = c10::RegisterOperators().op(
      "_test::my_op(Tensor(a) dummy, int input) -> Tensor(a)", &slice_kernel);
  auto handle = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});

  EXPECT_TRUE(handle.has_value());
  EXPECT_TRUE(handle->hasSchema());

  auto nativert_schema = torch::nativert::FunctionSchema(handle->schema());

  EXPECT_TRUE(nativert_schema.alias(0, 0));
  EXPECT_FALSE(nativert_schema.alias(1, 0));

  // bounds check
  EXPECT_THROW(nativert_schema.alias(2, 0), c10::Error);
  EXPECT_THROW(nativert_schema.alias(1, 1), c10::Error);
}
