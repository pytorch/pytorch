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
using at::Tensor;

namespace {

C10_DECLARE_TENSOR_TYPE(TensorType1);
C10_DEFINE_TENSOR_TYPE(TensorType1);

struct DummyKernel final : OperatorKernel {
  void operator()(Tensor) {}
};

FunctionSchema dummySchema(
    "_test::dummy",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest, whenTryingToRegisterWithoutKernel_thenFails) {
  // make sure it crashes when kernel is absent
  EXPECT_THROW(
    c10::RegisterOperators().op(dummySchema, dispatchKey(TensorType1())),
    c10::Error
  );

  // but make sure it doesn't crash when kernel is present
  c10::RegisterOperators().op(dummySchema, kernel<DummyKernel>(), dispatchKey(TensorType1()));
}

TEST(OperatorRegistrationTest, whenTryingToRegisterWithoutDispatchKey_thenFails) {
  // make sure it crashes when dispatch key is absent
  EXPECT_THROW(
    c10::RegisterOperators().op(dummySchema, kernel<DummyKernel>()),
    c10::Error
  );

  // but make sure it doesn't crash when dispatch key is present
  c10::RegisterOperators().op(dummySchema, kernel<DummyKernel>(), dispatchKey(TensorType1()));
}

}
