/**
 * This file contains some general registration test cases.
 * More detailed test cases containing different APIs for registering kernels
 * are found in other files in this directory.
 */

#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>

using c10::RegisterOperators;
using c10::OperatorKernel;
using c10::FunctionSchema;
using c10::Argument;
using c10::kernel;
using c10::dispatchKey;
using c10::Dispatcher;
using at::Tensor;

namespace {

C10_DECLARE_TENSOR_TYPE(TensorType1);
C10_DEFINE_TENSOR_TYPE(TensorType1);
C10_DECLARE_TENSOR_TYPE(TensorType2);
C10_DEFINE_TENSOR_TYPE(TensorType2);

struct DummyKernel final : OperatorKernel {
  void operator()(Tensor) {}
};

struct MockKernel final : OperatorKernel {
  MockKernel(bool* called): called_(called) {}

  void operator()(Tensor) {
    *called_ = true;
  }
private:
  bool* called_;
};

FunctionSchema dummySchema(
    "_test::dummy",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest, givenOpWithoutFallbackKernel_whenCallingOpWithWrongDispatchKey_thenFails) {
  auto registrar = c10::RegisterOperators().op(dummySchema, kernel<DummyKernel>(), dispatchKey(TensorType1()));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType2()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenOpWithFallbackKernelOutOfScope_whenCallingOpWithWrongDispatchKey_thenFails) {
  auto registrar = c10::RegisterOperators().op(dummySchema, kernel<DummyKernel>(), dispatchKey(TensorType1()));
  {
    auto inner_registrar = c10::RegisterOperators().op(dummySchema, kernel<DummyKernel>());
    // this registered a fallback kernel, but now that registration goes out of scope and deregisters it
  }

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType2()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenOpWithOnlyFallbackKernel_whenCallingOp_thenCallsFallbackKernel) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(dummySchema, kernel<MockKernel>(&called)); // note: no dispatch key means this is the fallback kernel

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(TensorType2()));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, givenOpWithOnlyFallbackKernelAndOtherKernelOutOfScope_whenCallingOp_thenCallsFallbackKernel) {
  bool called = false;
  bool other_called = false;
  auto registrar = c10::RegisterOperators().op(dummySchema, kernel<MockKernel>(&called)); // note: no dispatch key means this is the fallback kernel
  {
    auto inner_registrar = c10::RegisterOperators().op(dummySchema, kernel<MockKernel>(&other_called), dispatchKey(TensorType2()));
  }

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(TensorType2()));
  EXPECT_TRUE(called);
  EXPECT_FALSE(other_called);
}

TEST(OperatorRegistrationTest, givenOpWithFirstFallbackAndThenOtherKernel_whenCallingWithCorrectDispatchKey_thenCallsCorrectKernel) {
  bool called_kernel = false;
  bool called_fallback = false;
  auto registrar = c10::RegisterOperators()
    .op(dummySchema, kernel<MockKernel>(&called_fallback)) // note: no dispatch key means this is the fallback kernel
    .op(dummySchema, kernel<MockKernel>(&called_kernel), dispatchKey(TensorType1()));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called_kernel);
  EXPECT_FALSE(called_fallback);
  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(called_kernel);
  EXPECT_FALSE(called_fallback);
}

TEST(OperatorRegistrationTest, givenOpWithFirstFallbackAndThenOtherKernel_whenCallingWithWrongDispatchKey_thenCallsFallbackKernel) {
  bool called_kernel = false;
  bool called_fallback = false;
  auto registrar = c10::RegisterOperators()
    .op(dummySchema, kernel<MockKernel>(&called_fallback)) // note: no dispatch key means this is the fallback kernel
    .op(dummySchema, kernel<MockKernel>(&called_kernel), dispatchKey(TensorType1()));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called_kernel);
  EXPECT_FALSE(called_fallback);
  callOp(*op, dummyTensor(TensorType2()));
  EXPECT_FALSE(called_kernel);
  EXPECT_TRUE(called_fallback);
}


TEST(OperatorRegistrationTest, givenOpWithFirstOtherAndThenFallbackKernel_whenCallingWithCorrectDispatchKey_thenCallsCorrectKernel) {
  bool called_kernel = false;
  bool called_fallback = false;
  auto registrar = c10::RegisterOperators()
    .op(dummySchema, kernel<MockKernel>(&called_kernel), dispatchKey(TensorType1()))
    .op(dummySchema, kernel<MockKernel>(&called_fallback)); // note: no dispatch key means this is the fallback kernel

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called_kernel);
  EXPECT_FALSE(called_fallback);
  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(called_kernel);
  EXPECT_FALSE(called_fallback);
}

TEST(OperatorRegistrationTest, givenOpWithFirstOtherAndThenFallbackKernel_whenCallingWithWrongDispatchKey_thenCallsFallbackKernel) {
  bool called_kernel = false;
  bool called_fallback = false;
  auto registrar = c10::RegisterOperators()
    .op(dummySchema, kernel<MockKernel>(&called_kernel), dispatchKey(TensorType1()))
    .op(dummySchema, kernel<MockKernel>(&called_fallback)); // note: no dispatch key means this is the fallback kernel

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called_kernel);
  EXPECT_FALSE(called_fallback);
  callOp(*op, dummyTensor(TensorType2()));
  EXPECT_FALSE(called_kernel);
  EXPECT_TRUE(called_fallback);
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegistering_thenOnlyRegistersSchema) {
  auto registrar = c10::RegisterOperators().op(dummySchema);

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType1()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRunningOutOfScope_thenSchemaIsGone) {
  {
    auto registrar = c10::RegisterOperators().op(dummySchema);
  }

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  EXPECT_FALSE(op.has_value());
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringKernelAfterwards_thenCanBeCalled) {
  auto registrar1 = c10::RegisterOperators().op(dummySchema);

  bool called_kernel = false;
  auto registrar2 = c10::RegisterOperators().op(dummySchema, kernel<MockKernel>(&called_kernel), dispatchKey(TensorType1()));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(called_kernel);
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringKernelAfterwardsAndRunsOutOfScope_thenSchemaIsStillThereButCannotBeCalledAnymore) {
  auto registrar1 = c10::RegisterOperators().op(dummySchema);

  {
    auto registrar2 = c10::RegisterOperators().op(dummySchema, kernel<DummyKernel>(), dispatchKey(TensorType1()));
  }

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType1()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

}
