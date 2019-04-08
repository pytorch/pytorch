#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>

/**
 * This file tests the legacy lambda-based API for registering kernels:
 *
 * > auto registry = c10::RegisterOperators()
 * >    .op("myfunc(Tensor a) -> Tensor", [] (Tensor a) -> Tensor {...});
 */

// This intentionally tests a deprecated API
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

using c10::RegisterOperators;
using c10::FunctionSchema;
using c10::Argument;
using c10::IntType;
using c10::FloatType;
using c10::ListType;
using c10::kernel;
using c10::dispatchKey;
using c10::TensorTypeId;
using c10::KernelCache;
using c10::Stack;
using c10::guts::make_unique;
using c10::ivalue::TensorList;
using c10::ivalue::IntList;
using c10::intrusive_ptr;
using c10::ArrayRef;
using std::unique_ptr;
using at::Tensor;

namespace {

C10_DECLARE_TENSOR_TYPE(TensorType1);
C10_DEFINE_TENSOR_TYPE(TensorType1);
C10_DECLARE_TENSOR_TYPE(TensorType2);
C10_DEFINE_TENSOR_TYPE(TensorType2);

FunctionSchema errorOpSchema(
    "_test::error",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", IntType::get())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

FunctionSchema opSchema(
    "_test::my_op",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", IntType::get())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

void expectCallsIncrement(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema("_test::my_op", "");
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(opSchema, [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernel_whenRegisteredInConstructor_thenCanBeCalled) {
  auto registrar = RegisterOperators(opSchema, [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op(opSchema, [] (const Tensor& tensor, int64_t input) -> int64_t {
          return input + 1;
        })
      .op(errorOpSchema, [] (const Tensor& tensor, int64_t input) -> int64_t {
          EXPECT_TRUE(false); // this kernel should never be called
          return 0;
        });
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op(opSchema, [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  auto registrar2 = RegisterOperators().op(errorOpSchema, [] (const Tensor& tensor, int64_t input) -> int64_t {
      EXPECT_TRUE(false); // this kernel should never be called
      return 0;
    });
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar = RegisterOperators().op(opSchema, [] (const Tensor& tensor, int64_t input) -> int64_t {
        return input + 1;
      });

    expectCallsIncrement(TensorType1());
  }

  // now the registrar is destructed. Assert that the schema is gone.
  expectDoesntFindOperator("_test::my_op");
}

bool was_called = false;

FunctionSchema opWithoutOutputSchema(
    "_test::no_return",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(opWithoutOutputSchema, [] (const Tensor&) -> void {
    was_called = true;
  });

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_return", "");
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

FunctionSchema opWithZeroOutputsSchema(
    "_test::zero_outputs",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(opWithZeroOutputsSchema, [] (const Tensor&) -> std::tuple<> {
    was_called = true;
    return std::make_tuple();
  });

  auto op = c10::Dispatcher::singleton().findSchema("_test::zero_outputs", "");
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

FunctionSchema opWithIntOutputSchema(
    "_test::int_output",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("a", IntType::get()),
                           Argument("b", IntType::get())}),
    (std::vector<Argument>{Argument("sum", IntType::get())}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntOutputSchema, [] (Tensor, int64_t a, int64_t b) -> int64_t {
        return a + b;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

FunctionSchema opWithTensorOutput(
    "_test::returning_tensor",
    "",
    (std::vector<Argument>{Argument("input")}),
    (std::vector<Argument>{Argument("output")}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorOutput, [] (const Tensor& input) -> Tensor {
        return input;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::returning_tensor", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType1(), result[0].toTensor().type_id());

  result = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
}

FunctionSchema opWithTensorListOutputSchema(
    "_test::list_output",
    "",
    (std::vector<Argument>{Argument("input1"),
                           Argument("input2"),
                           Argument("input3")}),
    (std::vector<Argument>{Argument("output", ListType::ofTensors())}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorListOutputSchema, [] (const Tensor& input1, const Tensor& input2, const Tensor& input3) -> std::vector<Tensor> {
        return {input1, input2, input3};
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::list_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), dummyTensor(TensorType2()), dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toTensorListRef().size());
  EXPECT_EQ(TensorType1(), result[0].toTensorListRef()[0].type_id());
  EXPECT_EQ(TensorType2(), result[0].toTensorListRef()[1].type_id());
  EXPECT_EQ(TensorType1(), result[0].toTensorListRef()[2].type_id());
}

FunctionSchema opWithIntListOutputSchema(
    "_test::list_output",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input1", IntType::get()),
                           Argument("input2", IntType::get()),
                           Argument("input3", IntType::get())}),
    (std::vector<Argument>{Argument("output", ListType::ofInts())}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntListOutputSchema, [](const Tensor&, int64_t input1, int64_t input2, int64_t input3) -> std::vector<int64_t> {
        return {input1, input2, input3};
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::list_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), 2, 4, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toIntListRef().size());
  EXPECT_EQ(2, result[0].toIntListRef()[0]);
  EXPECT_EQ(4, result[0].toIntListRef()[1]);
  EXPECT_EQ(6, result[0].toIntListRef()[2]);
}

FunctionSchema opWithMultipleOutputsSchema(
    "_test::multiple_outputs",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{Argument("output1"),
                           Argument("output2", IntType::get()),
                           Argument("output3", ListType::ofTensors())}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
     .op(opWithMultipleOutputsSchema, [] (Tensor) -> std::tuple<Tensor, int64_t, std::vector<Tensor>> {
       return std::tuple<Tensor, int64_t, std::vector<Tensor>>(
         dummyTensor(TensorType2()), 5, {dummyTensor(TensorType1()), dummyTensor(TensorType2())}
       );
     });

  auto op = c10::Dispatcher::singleton().findSchema("_test::multiple_outputs", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(3, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
  EXPECT_EQ(5, result[1].toInt());
  EXPECT_EQ(2, result[2].toTensorListRef().size());
  EXPECT_EQ(TensorType1(), result[2].toTensorListRef()[0].type_id());
  EXPECT_EQ(TensorType2(), result[2].toTensorListRef()[1].type_id());
}

FunctionSchema opWithTensorInputWithOutput(
    "_test::tensor_input",
    "",
    (std::vector<Argument>{Argument("input")}),
    (std::vector<Argument>{Argument("output")}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorInputWithOutput, [] (const Tensor& input1) -> Tensor {
        return input1;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType1(), result[0].toTensor().type_id());

  result = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorInputWithOutput, [](Tensor input1) -> Tensor {
        return input1;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType1(), result[0].toTensor().type_id());

  result = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
}

Tensor captured_input;

FunctionSchema opWithTensorInputWithoutOutput(
    "_test::tensor_input",
    "",
    (std::vector<Argument>{Argument("input")}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorInputWithoutOutput, [] (const Tensor& input1) -> void {
        captured_input = input1;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType1(), captured_input.type_id());

  outputs = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType2(), captured_input.type_id());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorInputWithoutOutput, [] (Tensor input1) -> void {
        captured_input = input1;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType1(), captured_input.type_id());

  outputs = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType2(), captured_input.type_id());
}

int64_t captured_int_input = 0;

FunctionSchema opWithIntInputWithoutOutput(
    "_test::int_input",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", IntType::get())}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntInputWithoutOutput, [](Tensor, int64_t input1) -> void {
        captured_int_input = input1;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  captured_int_input = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_int_input);
}

FunctionSchema opWithIntInputWithOutput(
    "_test::int_input",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", IntType::get())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntInputWithOutput, [] (Tensor, int64_t input1) -> int64_t {
        return input1 + 1;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

FunctionSchema opWithIntListInputWithoutOutput(
    "_test::int_list_input",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", ListType::ofInts())}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntListInputWithoutOutput, [] (Tensor, ArrayRef<int64_t> input1) -> void {
        captured_input_list_size = input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

FunctionSchema opWithIntListInputWithOutput(
    "_test::int_list_input",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", ListType::ofInts())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntListInputWithOutput, [](Tensor, ArrayRef<int64_t> input1) -> int64_t {
        return input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(3, outputs[0].toInt());
}

FunctionSchema opWithTensorListInputWithoutOutput(
    "_test::tensor_list_input",
    "",
    (std::vector<Argument>{Argument("input", ListType::ofTensors())}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorListInputWithoutOutput, [] (ArrayRef<Tensor> input1) -> void {
        captured_input_list_size = input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

FunctionSchema opWithTensorListInputWithOutput(
    "_test::tensor_list_input",
    "",
    (std::vector<Argument>{Argument("input", ListType::ofTensors())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorListInputWithOutput, [] (ArrayRef<Tensor> input1) -> int64_t {
        return input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg")}),
          (std::vector<Argument>{Argument("ret", IntType::get())})
      ), [] (Tensor) -> int64_t {return 0;});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg"), Argument("arg2")}),
            (std::vector<Argument>{Argument("ret", IntType::get())})
        ), [] (Tensor) -> int64_t {return 0;});
    }, "The number of arguments is different. Specified 2 but inferred 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg"), Argument("arg2")}),
          (std::vector<Argument>{})
      ), [] (Tensor, Tensor) -> void {});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{}),
            (std::vector<Argument>{})
        ), [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. Specified 0 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{})
        ), [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. Specified 1 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg"), Argument("arg2"), Argument("arg3")}),
            (std::vector<Argument>{})
        ), [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. Specified 3 but inferred 2"
  );
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg1"), Argument("arg2", IntType::get())}),
          (std::vector<Argument>{Argument("ret", IntType::get())})
      ), [] (Tensor, int64_t) -> int64_t {return 0;});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg1"), Argument("arg2", FloatType::get())}),
            (std::vector<Argument>{Argument("ret", IntType::get())})
        ), [] (Tensor, int64_t) -> int64_t {return 0;});
    }, "Type mismatch in argument 2: specified float but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg1", IntType::get()), Argument("arg2", IntType::get())}),
            (std::vector<Argument>{Argument("ret", IntType::get())})
        ), [] (Tensor, int64_t) -> int64_t {return 0;});
    }, "Type mismatch in argument 1: specified int but inferred Tensor"
  );
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg")}),
          (std::vector<Argument>{Argument("ret", IntType::get())})
      ), [] (Tensor) -> int64_t {return 0;});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{})
        ), [] (Tensor) -> int64_t {return 0;});
    }, "The number of returns is different. Specified 0 but inferred 1"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret1", IntType::get()),
                                   Argument("ret2", IntType::get())})
        ), [] (Tensor) -> int64_t {return 0;});
    }, "The number of returns is different. Specified 2 but inferred 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg")}),
          (std::vector<Argument>{})
      ), [] (Tensor) -> void {});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret")})
        ), [] (Tensor) -> void {});
    }, "The number of returns is different. Specified 1 but inferred 0"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret"), Argument("ret2")})
        ), [] (Tensor) -> void {});
    }, "The number of returns is different. Specified 2 but inferred 0"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg")}),
          (std::vector<Argument>{Argument("ret1"), Argument("ret2")})
      ), [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{})
        ), [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. Specified 0 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret1")})
        ), [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. Specified 1 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret1"), Argument("ret2"), Argument("ret3")})
        ), [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. Specified 3 but inferred 2"
  );
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg")}),
          (std::vector<Argument>{Argument("ret", IntType::get())})
      ), [] (Tensor) -> int64_t {return 0;});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret")})
        ), [] (Tensor) -> int64_t {return 0;});
    }, "Type mismatch in return 1: specified Tensor but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret", FloatType::get())})
        ), [] (Tensor) -> int64_t {return 0;});
    }, "Type mismatch in return 1: specified float but inferred int"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg")}),
          (std::vector<Argument>{Argument("ret")})
      ), [] (Tensor) -> Tensor {return {};});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret", FloatType::get())})
        ), [] (Tensor) -> Tensor {return {};});
    }, "Type mismatch in return 1: specified float but inferred Tensor"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op(FunctionSchema(
          "_test::mismatch",
          "",
          (std::vector<Argument>{Argument("arg")}),
          (std::vector<Argument>{Argument("ret1"), Argument("ret2", IntType::get())})
      ), [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret1"), Argument("ret2", FloatType::get())})
        ), [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});
    }, "Type mismatch in return 2: specified float but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op(FunctionSchema(
            "_test::mismatch",
            "",
            (std::vector<Argument>{Argument("arg")}),
            (std::vector<Argument>{Argument("ret1", IntType::get()), Argument("ret2", IntType::get())})
        ), [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});
    }, "Type mismatch in return 1: specified int but inferred Tensor"
  );
}

}

#pragma GCC diagnostic pop
