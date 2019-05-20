/**
 * This file contains some general registration test cases.
 * More detailed test cases containing different APIs for registering kernels
 * are found in other files in this directory.
 */

#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <functional>

using c10::RegisterOperators;
using c10::OperatorKernel;
using c10::Dispatcher;
using c10::IValue;
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
TEST(OperatorRegistrationTest, givenOpWithoutFallbackKernel_whenCallingOpWithWrongDispatchKey_thenFails) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>().dispatchKey(TensorType1()));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType2()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenOpWithFallbackKernelOutOfScope_whenCallingOpWithWrongDispatchKey_thenFails) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>().dispatchKey(TensorType1()));
  {
    auto inner_registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>());
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
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called)); // note: no dispatch key means this is the fallback kernel

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(TensorType2()));
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, givenOpWithOnlyFallbackKernelAndOtherKernelOutOfScope_whenCallingOp_thenCallsFallbackKernel) {
  bool called = false;
  bool other_called = false;
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called)); // note: no dispatch key means this is the fallback kernel
  {
    auto inner_registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&other_called).dispatchKey(TensorType2()));
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
    .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_fallback)) // note: no dispatch key means this is the fallback kernel
    .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel).dispatchKey(TensorType1()));

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
    .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_fallback)) // note: no dispatch key means this is the fallback kernel
    .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel).dispatchKey(TensorType1()));

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
    .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel).dispatchKey(TensorType1()))
    .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_fallback)); // note: no dispatch key means this is the fallback kernel

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
    .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel).dispatchKey(TensorType1()))
    .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_fallback)); // note: no dispatch key means this is the fallback kernel

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called_kernel);
  EXPECT_FALSE(called_fallback);
  callOp(*op, dummyTensor(TensorType2()));
  EXPECT_FALSE(called_kernel);
  EXPECT_TRUE(called_fallback);
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegistering_thenOnlyRegistersSchema) {
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType1()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRunningOutOfScope_thenSchemaIsGone) {
  {
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
  }

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  EXPECT_FALSE(op.has_value());
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringKernelAfterwards_thenCanBeCalled) {
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");

  bool called_kernel = false;
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel).dispatchKey(TensorType1()));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(called_kernel);
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringKernelAfterwardsAndRunsOutOfScope_thenSchemaIsStillThereButCannotBeCalledAnymore) {
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");

  {
    auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>().dispatchKey(TensorType1()));
  }

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType1()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernelsWithoutTensorInputs_whenRegistering_thenRegisters) {
  // as long as we don't register non-fallback kernels, ops without tensor arguments are fine

  auto registrar = c10::RegisterOperators().op("_test::dummy() -> ()");

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered
}

TEST(OperatorRegistrationTest, givenKernelsWithSameDispatchKey_whenRegistering_thenShowsWarning) {
  auto registrar = c10::RegisterOperators()
      .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>().dispatchKey(TensorType1()));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  testing::internal::CaptureStderr();
  c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>().dispatchKey(TensorType1()));
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, testing::HasSubstr("Registered a kernel that overwrote a previously registered kernel with same dispatch key"));
}

TEST(OperatorRegistrationTest, givenKernelsWithSameDispatchKey_whenCalled_thenCallsNewerKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1).dispatchKey(TensorType1()));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2).dispatchKey(TensorType1()));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_FALSE(called_kernel1);
  EXPECT_TRUE(called_kernel2);
}

TEST(OperatorRegistrationTest, givenKernelsWithSameFallbackDispatchKey_whenCalled_thenCallsNewerKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2));

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_FALSE(called_kernel1);
  EXPECT_TRUE(called_kernel2);
}

TEST(OperatorRegistrationTest, givenKernelsWithSameDispatchKey_whenNewerKernelDeletedAndOpCalled_thenCallsOlderKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1).dispatchKey(TensorType1()));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2).dispatchKey(TensorType1()));

  registrar2 = c10::RegisterOperators(); // destruct the registrar

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(called_kernel1);
  EXPECT_FALSE(called_kernel2);
}

TEST(OperatorRegistrationTest, givenKernelsWithSameFallbackDispatchKey_whenNewerKernelDeletedAndOpCalled_thenCallsOlderKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2));

  registrar2 = c10::RegisterOperators(); // destruct the registrar

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(called_kernel1);
  EXPECT_FALSE(called_kernel2);
}

TEST(OperatorRegistrationTest, givenKernelsWithSameDispatchKey_whenOlderKernelDeletedAndOpCalled_thenCallsNewerKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1).dispatchKey(TensorType1()));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2).dispatchKey(TensorType1()));

  registrar1 = c10::RegisterOperators(); // destruct the registrar

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_FALSE(called_kernel1);
  EXPECT_TRUE(called_kernel2);
}

TEST(OperatorRegistrationTest, givenKernelsWithSameFallbackDispatchKey_whenOlderKernelDeletedAndOpCalled_thenCallsNewerKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2));

  registrar1 = c10::RegisterOperators(); // destruct the registrar

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  callOp(*op, dummyTensor(TensorType1()));
  EXPECT_FALSE(called_kernel1);
  EXPECT_TRUE(called_kernel2);
}

TEST(OperatorRegistrationTest, givenKernelsWithSameDispatchKey_whenOlderAndThenNewerKernelDeletedAndOpCalled_thenFails) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar0 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1).dispatchKey(TensorType1()));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2).dispatchKey(TensorType1()));

  registrar1 = c10::RegisterOperators(); // destruct the registrar
  registrar2 = c10::RegisterOperators(); // destruct the registrar

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType1()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenKernelsWithSameFallbackDispatchKey_whenOlderAndThenNewerKernelDeletedAndOpCalled_thenFails) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar0 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2));

  registrar1 = c10::RegisterOperators(); // destruct the registrar
  registrar2 = c10::RegisterOperators(); // destruct the registrar

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType1()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenKernelsWithSameDispatchKey_whenNewerAndThenOlderKernelDeletedAndOpCalled_thenFails) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar0 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1).dispatchKey(TensorType1()));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2).dispatchKey(TensorType1()));

  registrar2 = c10::RegisterOperators(); // destruct the registrar
  registrar1 = c10::RegisterOperators(); // destruct the registrar

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType1()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}

TEST(OperatorRegistrationTest, givenKernelsWithSameFallbackDispatchKey_whenNewerAndThenOlderKernelDeletedAndOpCalled_thenFails) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  auto registrar0 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel1));
  auto registrar2 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(&called_kernel2));

  registrar2 = c10::RegisterOperators(); // destruct the registrar
  registrar1 = c10::RegisterOperators(); // destruct the registrar

  auto op = Dispatcher::singleton().findSchema("_test::dummy", "");
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(TensorType1()));
  }, "Didn't find kernel to dispatch to for operator '_test::dummy'");
}



/**
 * This is used to check that a given type works correctly when passed as input
 * to or as output from a kernel.
 *
 * Call ArgTypeTestKernel<Input, Output>::test(input, inputExpectation, output, outputExpectation, schema)
 * to test that a kernel with `Input` as input type and `Output` as output types,
 * when called with `input` fulfills `inputExpectation` inside the kernel, then
 * returns `output` and the returned value fulfills `outputExpectation`.
 *
 * `inputExpectation` and `outputExpectation` should be lambdas that run
 * googletest expect macros (or use other ways to assert the expectation is met).
 *
 * Optionally, you can specify the argument list part of a function schema
 * (e.g. "(Tensor a) -> Tensor") as an additional argument to use when
 * registering the kernel. In this case, the operator registration logic will
 * check that the kernel function signature matches the one you specified.
 */
template<class InputType, class OutputType = InputType>
struct ArgTypeTestKernel final : OperatorKernel {
  explicit ArgTypeTestKernel(InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output)
  : input_(std::move(input)), inputExpectation_(std::move(inputExpectation)), output_(std::move(output)) {}

  OutputType operator()(InputType input) const {
    inputExpectation_(std::move(input));
    return output_;
  }

  static void test(InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
    auto registry = c10::RegisterOperators().op("_test::my_op" + schema, c10::RegisterOperators::options().kernel<ArgTypeTestKernel>(input, std::move(inputExpectation), std::move(output)));
    auto op = Dispatcher::singleton().findSchema("_test::my_op", "");
    ASSERT_TRUE(op.has_value()); // assert schema is registered
    auto actualOutput = callOp(*op, std::move(input));
    outputExpectation(actualOutput);
  }

private:

  InputType input_;
  std::function<void(const InputType&)> inputExpectation_;
  OutputType output_;
  std::string schema_;
};

template<class InputType, class OutputType = InputType>
struct testArgTypes final {
  static void test(InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const IValue&)> outputExpectation, const std::string& schema) {
    // Test with explicitly specified schema
    ArgTypeTestKernel<InputType, OutputType>::test(
      input, inputExpectation, output, [&] (const c10::Stack& output) {
        EXPECT_EQ(1, output.size());
        outputExpectation(output[0]);
      }, schema
    );

    // Test with inferred schema
    ArgTypeTestKernel<InputType, OutputType>::test(
      input, inputExpectation, output, [&] (const c10::Stack& output) {
        EXPECT_EQ(1, output.size());
        outputExpectation(output[0]);
      }, ""
    );

    // Test taking argument and returning nothing
    ArgTypeTestKernel<InputType, std::tuple<>>::test(
      input, inputExpectation, {}, [] (const c10::Stack&) {}, ""
    );

    // Test taking argument and returning multiple outputs
    ArgTypeTestKernel<InputType, std::tuple<int64_t, OutputType>>::test(
      input, inputExpectation, std::tuple<int64_t, OutputType>{3, output}, [&] (const c10::Stack& output) {
        EXPECT_EQ(2, output.size());
        EXPECT_EQ(3, output[0].toInt());
        outputExpectation(output[1]);
      }, ""
    );
  }
};

TEST(OperatorRegistrationTest, testAvailableArgTypes) {
  // TODO Test Scalar

  // primitive types
  testArgTypes<double>::test(
    1.5, [] (const double& v) {EXPECT_EQ(1.5, v);},
    2.5, [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
    "(float a) -> float");
  testArgTypes<int64_t>::test(
    1, [] (const int64_t& v) {EXPECT_EQ(1, v);},
    2, [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
    "(int a) -> int");
  testArgTypes<bool>::test(
    true, [] (const bool& v) {EXPECT_EQ(true, v);},
    false, [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
    "(bool a) -> bool");
  testArgTypes<bool>::test(
    false, [] (const bool& v) {EXPECT_EQ(false, v);},
    true, [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
    "(bool a) -> bool");
  testArgTypes<std::string>::test(
    "string1", [] (const std::string& v) {EXPECT_EQ("string1", v);},
    "string2", [] (const IValue& v) {EXPECT_EQ("string2", v.toString()->string());},
    "(str a) -> str");
  testArgTypes<Tensor>::test(
    dummyTensor(TensorType1()), [] (const Tensor& v) {EXPECT_EQ(TensorType1(), v.type_id());},
    dummyTensor(TensorType2()), [] (const IValue& v) {EXPECT_EQ(TensorType2(), v.toTensor().type_id());},
    "(Tensor a) -> Tensor");


  // optional types (with has_value() == true)
  testArgTypes<c10::optional<double>>::test(
    c10::optional<double>(1.5), [] (const c10::optional<double>& v) {EXPECT_EQ(1.5, v.value());},
    c10::optional<double>(2.5), [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
    "(float? a) -> float?");
  testArgTypes<c10::optional<int64_t>>::test(
    c10::optional<int64_t>(1), [] (const c10::optional<int64_t>& v) {EXPECT_EQ(1, v.value());},
    c10::optional<int64_t>(2), [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
    "(int? a) -> int?");
  testArgTypes<c10::optional<bool>>::test(
    c10::optional<bool>(true), [] (const c10::optional<bool>& v) {EXPECT_EQ(true, v.value());},
    c10::optional<bool>(false), [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
    "(bool? a) -> bool?");
  testArgTypes<c10::optional<bool>>::test(
    c10::optional<bool>(false), [] (const c10::optional<bool>& v) {EXPECT_EQ(false, v.value());},
    c10::optional<bool>(true), [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
    "(bool? a) -> bool?");
  testArgTypes<c10::optional<std::string>>::test(
    c10::optional<std::string>("string1"), [] (const c10::optional<std::string>& v) {EXPECT_EQ("string1", v.value());},
    c10::optional<std::string>("string2"), [] (const IValue& v) {EXPECT_EQ("string2", v.toString()->string());},
    "(str? a) -> str?");
  testArgTypes<c10::optional<Tensor>>::test(
    c10::optional<Tensor>(dummyTensor(TensorType1())), [] (const c10::optional<Tensor>& v) {EXPECT_EQ(TensorType1(), v.value().type_id());},
    c10::optional<Tensor>(dummyTensor(TensorType2())), [] (const IValue& v) {EXPECT_EQ(TensorType2(), v.toTensor().type_id());},
    "(Tensor? a) -> Tensor?");


  // optional types (with has_value() == false)
  testArgTypes<c10::optional<double>>::test(
    c10::optional<double>(), [] (const c10::optional<double>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<double>(), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(float? a) -> float?");
  testArgTypes<c10::optional<int64_t>>::test(
    c10::optional<int64_t>(), [] (const c10::optional<int64_t>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<int64_t>(), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(int? a) -> int?");
  testArgTypes<c10::optional<bool>>::test(
    c10::optional<bool>(), [] (const c10::optional<bool>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<bool>(), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(bool? a) -> bool?");
  testArgTypes<c10::optional<bool>>::test(
    c10::optional<bool>(), [] (const c10::optional<bool>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<bool>(), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(bool? a) -> bool?");
  testArgTypes<c10::optional<std::string>>::test(
    c10::optional<std::string>(), [] (const c10::optional<std::string>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<std::string>(), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(str? a) -> str?");
  testArgTypes<c10::optional<Tensor>>::test(
    c10::optional<Tensor>(), [] (const c10::optional<Tensor>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<Tensor>(), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(Tensor? a) -> Tensor?");


  // list types (with empty list)
  testArgTypes<std::vector<double>>::test(
    std::vector<double>(), [] (const std::vector<double>& v) {EXPECT_EQ(0, v.size());},
    std::vector<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.toDoubleListRef().size());},
    "(float[] a) -> float[]");
  testArgTypes<std::vector<int64_t>, std::vector<int64_t>>::test(
    std::vector<int64_t>(), [] (const std::vector<int64_t>& v) {EXPECT_EQ(0, v.size());},
    std::vector<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.toIntListRef().size());},
    "(int[] a) -> int[]");
  testArgTypes<std::vector<bool>>::test(
    std::vector<bool>(), [] (const std::vector<bool>& v) {EXPECT_EQ(0, v.size());},
    std::vector<bool>(), [] (const IValue& v) {EXPECT_EQ(0, v.toBoolListRef().size());},
    "(bool[] a) -> bool[]");
  testArgTypes<std::vector<std::string>>::test(
    std::vector<std::string>(), [] (const std::vector<std::string>& v) {EXPECT_EQ(0, v.size());},
    std::vector<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toGenericListRef().size());},
    "(str[] a) -> str[]");


  // list types (with non-empty list)
  testArgTypes<std::vector<double>>::test(
    std::vector<double>({1.5, 2.5}), [] (const std::vector<double>& v) {EXPECT_EQ(std::vector<double>({1.5, 2.5}), v);},
    std::vector<double>({3.5, 4.5}), [] (const IValue& v) {EXPECT_EQ(std::vector<double>({3.5, 4.5}), v.toDoubleListRef());},
    "(float[] a) -> float[]");
  testArgTypes<std::vector<int64_t>>::test(
    std::vector<int64_t>({1, 2}), [] (const std::vector<int64_t>& v) {EXPECT_EQ(std::vector<int64_t>({1, 2}), v);},
    std::vector<int64_t>({3, 4}), [] (const IValue& v) {EXPECT_EQ(std::vector<int64_t>({3, 4}), v.toIntListRef());},
    "(int[] a) -> int[]");
  testArgTypes<std::vector<bool>>::test(
    std::vector<bool>({true, false}), [] (const std::vector<bool>& v) {EXPECT_EQ(std::vector<bool>({true, false}), v);},
    std::vector<bool>({true, false}), [] (const IValue& v) {EXPECT_EQ(std::vector<bool>({true, false}), v.toBoolListRef());},
    "(bool[] a) -> bool[]");
  testArgTypes<std::vector<std::string>>::test(
    std::vector<std::string>({"first", "second"}), [] (const std::vector<std::string>& v) {EXPECT_EQ(std::vector<std::string>({"first", "second"}), v);},
    std::vector<std::string>({"first", "second"}), [] (const IValue& v) {
      EXPECT_EQ(2, v.toGenericListRef().size());
      EXPECT_EQ("first", v.toGenericListRef()[0].toStringRef());
      EXPECT_EQ("second", v.toGenericListRef()[1].toStringRef());
    },
    "(str[] a) -> str[]");
  testArgTypes<std::vector<Tensor>>::test(
    std::vector<Tensor>({dummyTensor(TensorType1()), dummyTensor(TensorType2())}), [] (const std::vector<Tensor>& v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(TensorType1(), v[0].type_id());
      EXPECT_EQ(TensorType2(), v[1].type_id());
    },
    std::vector<Tensor>({dummyTensor(TensorType2()), dummyTensor(TensorType1())}), [] (const IValue& v) {
      EXPECT_EQ(2, v.toTensorListRef().size());
      EXPECT_EQ(TensorType2(), v.toTensorListRef()[0].type_id());
      EXPECT_EQ(TensorType1(), v.toTensorListRef()[1].type_id());
    },
    "(Tensor[] a) -> Tensor[]");

  // Test optional of list (with nullopt)
  testArgTypes<c10::optional<std::vector<int64_t>>>::test(
    c10::optional<std::vector<int64_t>>(c10::nullopt), [] (const c10::optional<std::vector<int64_t>>& v) {EXPECT_FALSE(v.has_value());},
    c10::optional<std::vector<int64_t>>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    "(int[]? a) -> int[]?");

  // Test optional of list (with empty list)
  testArgTypes<c10::optional<std::vector<int64_t>>>::test(
    c10::optional<std::vector<int64_t>>(std::vector<int64_t>{}), [] (const c10::optional<std::vector<int64_t>>& v) {EXPECT_EQ(0, v.value().size());},
    c10::optional<std::vector<int64_t>>(std::vector<int64_t>{}), [] (const IValue& v) {EXPECT_EQ(0, v.toIntListRef().size());},
    "(int[]? a) -> int[]?");

  // Test optional of list (with values)
  testArgTypes<c10::optional<std::vector<int64_t>>>::test(
    c10::optional<std::vector<int64_t>>({1, 2}), [] (const c10::optional<std::vector<int64_t>>& v) {EXPECT_EQ(std::vector<int64_t>({1, 2}), v.value());},
    c10::optional<std::vector<int64_t>>({3, 4}), [] (const IValue& v) {EXPECT_EQ(std::vector<int64_t>({3, 4}), v.toIntListRef());},
    "(int[]? a) -> int[]?");

  // TODO Do we want to support list of optional ?

  // dict types
  c10::Dict<std::string, std::string> str_dict;
  str_dict.insert("key1", "value1");
  str_dict.insert("key2", "value2");
  testArgTypes<c10::Dict<std::string, std::string>>::test(
    str_dict, [] (c10::Dict<std::string, std::string> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ("value1", v.at("key1"));
      EXPECT_EQ("value2", v.at("key2"));
    },
    str_dict, [] (const IValue& v) {
      c10::Dict<std::string, std::string> dict = c10::impl::toTypedDict<std::string, std::string>(std::move(v.toGenericDict()->elements()));
      EXPECT_EQ(2, dict.size());
      EXPECT_EQ("value1", dict.at("key1"));
      EXPECT_EQ("value2", dict.at("key2"));
    },
    "(Dict(str, str) a) -> Dict(str, str)");
  c10::Dict<int64_t, Tensor> tensor_dict;
  tensor_dict.insert(1, dummyTensor(TensorType1()));
  tensor_dict.insert(2, dummyTensor(TensorType2()));
  testArgTypes<c10::Dict<int64_t, Tensor>>::test(
    tensor_dict, [] (c10::Dict<int64_t, Tensor> v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(TensorType1(), v.at(1).type_id());
      EXPECT_EQ(TensorType2(), v.at(2).type_id());
    },
    tensor_dict, [] (const IValue& v) {
      c10::Dict<int64_t, Tensor> dict = c10::impl::toTypedDict<int64_t, Tensor>(std::move(v.toGenericDict()->elements()));
      EXPECT_EQ(2, dict.size());
      EXPECT_EQ(TensorType1(), dict.at(1).type_id());
      EXPECT_EQ(TensorType2(), dict.at(2).type_id());
    },
    "(Dict(int, Tensor) a) -> Dict(int, Tensor)");
}

}
