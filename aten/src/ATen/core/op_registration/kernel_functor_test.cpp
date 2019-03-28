#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>

using c10::RegisterOperators;
using c10::FunctionSchema;
using c10::OperatorKernel;
using c10::Argument;
using c10::IntType;
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

struct ErrorKernel final : public OperatorKernel {
  void operator()(const Tensor&) {
    EXPECT_TRUE(false); // this kernel should never be called
  }
};

FunctionSchema errorOpSchema(
    "_test::error",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{}));

struct IncrementKernel final : OperatorKernel {
  int operator()(const Tensor& tensor, int input) {
    return input + 1;
  }
};

struct DecrementKernel final : OperatorKernel {
  int operator()(const Tensor& tensor, int input) {
    return input - 1;
  }
};

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

void expectCallsDecrement(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema("_test::my_op", "");
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(4, result[0].toInt());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(opSchema, kernel<IncrementKernel>(), dispatchKey(TensorType1()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op(opSchema, kernel<IncrementKernel>(), dispatchKey(TensorType1()))
      .op(opSchema, kernel<ErrorKernel>(), dispatchKey(TensorType2()))
      .op(errorOpSchema, kernel<ErrorKernel>(), dispatchKey(TensorType1()))
      .op(errorOpSchema, kernel<ErrorKernel>(), dispatchKey(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op(opSchema, kernel<IncrementKernel>(), dispatchKey(TensorType1()));
  auto registrar2 = RegisterOperators().op(opSchema, kernel<ErrorKernel>(), dispatchKey(TensorType2()));
  auto registrar3 = RegisterOperators().op(errorOpSchema, kernel<ErrorKernel>(), dispatchKey(TensorType1()));
  auto registrar4 = RegisterOperators().op(errorOpSchema, kernel<ErrorKernel>(), dispatchKey(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar1 = RegisterOperators().op(opSchema, kernel<IncrementKernel>(), dispatchKey(TensorType1()));
    {
      auto registrar2 = RegisterOperators().op(opSchema, kernel<DecrementKernel>(), dispatchKey(TensorType2()));

      // assert that schema and cpu kernel are present
      expectCallsIncrement(TensorType1());
      expectCallsDecrement(TensorType2());
    }

    // now registrar2 is destructed. Assert that schema is still present but cpu kernel is not
    expectCallsIncrement(TensorType1());
    expectDoesntFindKernel("_test::my_op", TensorType2());
  }

  // now both registrars are destructed. Assert that the whole schema is gone
  expectDoesntFindOperator("_test::my_op");
}

bool was_called = false;

struct KernelWithoutOutput final : OperatorKernel {
  void operator()(const Tensor&) {
    was_called = true;
  }
};

FunctionSchema opWithoutOutputSchema(
    "_test::no_return",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(opWithoutOutputSchema, kernel<KernelWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_return", "");
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

struct KernelWithZeroOutputs final : OperatorKernel {
  std::tuple<> operator()(const Tensor&) {
    was_called = true;
    return std::make_tuple();
  }
};

FunctionSchema opWithZeroOutputsSchema(
    "_test::zero_outputs",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(opWithZeroOutputsSchema, kernel<KernelWithZeroOutputs>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::zero_outputs", "");
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

struct KernelWithIntOutput final : OperatorKernel {
  int operator()(Tensor, int a, int b) {
    return a + b;
  }
};


FunctionSchema opWithIntOutputSchema(
    "_test::int_output",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("a", IntType::get()),
                           Argument("b", IntType::get())}),
    (std::vector<Argument>{Argument("sum", IntType::get())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntOutputSchema, kernel<KernelWithIntOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

struct KernelWithTensorOutput final : OperatorKernel {
  Tensor operator()(const Tensor& input) {
    return input;
  }
};

FunctionSchema opWithTensorOutput(
    "_test::returning_tensor",
    "",
    (std::vector<Argument>{Argument("input")}),
    (std::vector<Argument>{Argument("output")}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorOutput, kernel<KernelWithTensorOutput>(), dispatchKey(TensorType1()))
      .op(opWithTensorOutput, kernel<KernelWithTensorOutput>(), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::returning_tensor", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType1(), result[0].toTensor().type_id());

  result = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
}

struct KernelWithTensorListOutput final : OperatorKernel {
  std::vector<Tensor> operator()(const Tensor& input1, const Tensor& input2, const Tensor& input3) {
    return {input1, input2, input3};
  }
};

FunctionSchema opWithTensorListOutputSchema(
    "_test::list_output",
    "",
    (std::vector<Argument>{Argument("input1"),
                           Argument("input2"),
                           Argument("input3")}),
    (std::vector<Argument>{Argument("output", ListType::ofTensors())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorListOutputSchema, kernel<KernelWithTensorListOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::list_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), dummyTensor(TensorType2()), dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toTensorListRef().size());
  EXPECT_EQ(TensorType1(), result[0].toTensorListRef()[0].type_id());
  EXPECT_EQ(TensorType2(), result[0].toTensorListRef()[1].type_id());
  EXPECT_EQ(TensorType1(), result[0].toTensorListRef()[2].type_id());
}

struct KernelWithIntListOutput final : OperatorKernel {
  std::vector<int64_t> operator()(const Tensor&, int input1, int input2, int input3) {
    return {input1, input2, input3};
  }
};

FunctionSchema opWithIntListOutputSchema(
    "_test::list_output",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input1", IntType::get()),
                           Argument("input2", IntType::get()),
                           Argument("input3", IntType::get())}),
    (std::vector<Argument>{Argument("output", ListType::ofInts())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntListOutputSchema, kernel<KernelWithIntListOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::list_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), 2, 4, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toIntListRef().size());
  EXPECT_EQ(2, result[0].toIntListRef()[0]);
  EXPECT_EQ(4, result[0].toIntListRef()[1]);
  EXPECT_EQ(6, result[0].toIntListRef()[2]);
}

struct KernelWithMultipleOutputs final : OperatorKernel {
  std::tuple<Tensor, int64_t, std::vector<Tensor>> operator()(Tensor) {
    return std::tuple<Tensor, int64_t, std::vector<Tensor>>(
      dummyTensor(TensorType2()), 5, {dummyTensor(TensorType1()), dummyTensor(TensorType2())}
    );
  }
};

FunctionSchema opWithMultipleOutputsSchema(
    "_test::multiple_outputs",
    "",
    (std::vector<Argument>{Argument("dummy")}),
    (std::vector<Argument>{Argument("output1"),
                           Argument("output2", IntType::get()),
                           Argument("output3", ListType::ofTensors())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
     .op(opWithMultipleOutputsSchema, kernel<KernelWithMultipleOutputs>(), dispatchKey(TensorType1()));

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

struct KernelWithTensorInputByReferenceWithOutput final : OperatorKernel {
  Tensor operator()(const Tensor& input1) {
    return input1;
  }
};

struct KernelWithTensorInputByValueWithOutput final : OperatorKernel {
  Tensor operator()(Tensor input1) {
    return input1;
  }
};

FunctionSchema opWithTensorInputWithOutput(
    "_test::tensor_input",
    "",
    (std::vector<Argument>{Argument("input")}),
    (std::vector<Argument>{Argument("output")}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorInputWithOutput, kernel<KernelWithTensorInputByReferenceWithOutput>(), dispatchKey(TensorType1()))
      .op(opWithTensorInputWithOutput, kernel<KernelWithTensorInputByReferenceWithOutput>(), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType1(), result[0].toTensor().type_id());

  result = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorInputWithOutput, kernel<KernelWithTensorInputByValueWithOutput>(), dispatchKey(TensorType1()))
      .op(opWithTensorInputWithOutput, kernel<KernelWithTensorInputByValueWithOutput>(), dispatchKey(TensorType2()));

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

struct KernelWithTensorInputByReferenceWithoutOutput final : OperatorKernel {
  void operator()(const Tensor& input1) {
    captured_input = input1;
  }
};

struct KernelWithTensorInputByValueWithoutOutput final : OperatorKernel {
  void operator()(Tensor input1) {
    captured_input = input1;
  }
};

FunctionSchema opWithTensorInputWithoutOutput(
    "_test::tensor_input",
    "",
    (std::vector<Argument>{Argument("input")}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorInputWithoutOutput, kernel<KernelWithTensorInputByReferenceWithoutOutput>(), dispatchKey(TensorType1()))
      .op(opWithTensorInputWithoutOutput, kernel<KernelWithTensorInputByReferenceWithoutOutput>(), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType1(), captured_input.type_id());

  outputs = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType2(), captured_input.type_id());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorInputWithoutOutput, kernel<KernelWithTensorInputByValueWithoutOutput>(), dispatchKey(TensorType1()))
      .op(opWithTensorInputWithoutOutput, kernel<KernelWithTensorInputByValueWithoutOutput>(), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType1(), captured_input.type_id());

  outputs = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType2(), captured_input.type_id());
}

int captured_int_input = 0;

struct KernelWithIntInputWithoutOutput final : OperatorKernel {
  void operator()(Tensor, int input1) {
    captured_int_input = input1;
  }
};

FunctionSchema opWithIntInputWithoutOutput(
    "_test::int_input",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", IntType::get())}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntInputWithoutOutput, kernel<KernelWithIntInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  captured_int_input = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_int_input);
}

struct KernelWithIntInputWithOutput final : OperatorKernel {
  int operator()(Tensor, int input1) {
    return input1 + 1;
  }
};

FunctionSchema opWithIntInputWithOutput(
    "_test::int_input",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", IntType::get())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntInputWithOutput, kernel<KernelWithIntInputWithOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int captured_input_list_size = 0;

struct KernelWithIntListInputWithoutOutput final : OperatorKernel {
  void operator()(Tensor, ArrayRef<int64_t> input1) {
    captured_input_list_size = input1.size();
  }
};

FunctionSchema opWithIntListInputWithoutOutput(
    "_test::int_list_input",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", ListType::ofInts())}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntListInputWithoutOutput, kernel<KernelWithIntListInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

struct KernelWithIntListInputWithOutput final : OperatorKernel {
  int operator()(Tensor, ArrayRef<int64_t> input1) {
    return input1.size();
  }
};

FunctionSchema opWithIntListInputWithOutput(
    "_test::int_list_input",
    "",
    (std::vector<Argument>{Argument("dummy"),
                           Argument("input", ListType::ofInts())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithIntListInputWithOutput, kernel<KernelWithIntListInputWithOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(3, outputs[0].toInt());
}

struct KernelWithTensorListInputWithoutOutput final : OperatorKernel {
  void operator()(ArrayRef<Tensor> input1) {
    captured_input_list_size = input1.size();
  }
};

FunctionSchema opWithTensorListInputWithoutOutput(
    "_test::tensor_list_input",
    "",
    (std::vector<Argument>{Argument("input", ListType::ofTensors())}),
    (std::vector<Argument>{}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorListInputWithoutOutput, kernel<KernelWithTensorListInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

struct KernelWithTensorListInputWithOutput final : OperatorKernel {
  int operator()(ArrayRef<Tensor> input1) {
    return input1.size();
  }
};

FunctionSchema opWithTensorListInputWithOutput(
    "_test::tensor_list_input",
    "",
    (std::vector<Argument>{Argument("input", ListType::ofTensors())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithTensorListInputWithOutput, kernel<KernelWithTensorListInputWithOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

class KernelWithCache final : public OperatorKernel {
public:
  KernelWithCache(): counter(3) {}

  int64_t operator()(Tensor) {
    return ++counter;
  }
private:
  int64_t counter;
};

FunctionSchema opWithCacheSchema(
    "_test::cache_op",
    "",
    (std::vector<Argument>{Argument("input")}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithCache_thenCacheIsKeptCorrectly) {
  auto registrar = RegisterOperators()
      .op(opWithCacheSchema, kernel<KernelWithCache>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::cache_op", "");
  ASSERT_TRUE(op.has_value());

  // expect first time calling returns a 4 (4 is the initial value in the cache)
  auto stack = makeStack(dummyTensor(TensorType1()));
  auto kernel = c10::Dispatcher::singleton().lookup(*op, &stack);
  kernel.call(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(4, stack[0].toInt());

  // expect second time calling returns a 5
  stack = makeStack(dummyTensor(TensorType1()));
  kernel.call(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(5, stack[0].toInt());

  // expect third time calling returns a 6
  stack = makeStack(dummyTensor(TensorType1()));
  kernel.call(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(6, stack[0].toInt());
}

class KernelWithConstructorArg final : public OperatorKernel {
public:
  explicit KernelWithConstructorArg(int64_t offset)
  : offset_(offset) {}

  int64_t operator()(const Tensor&, int64_t input) {
    return input + offset_;
  }

private:
  int64_t offset_;
};

FunctionSchema opWithConstructorArgsSchema(
    "_test::offset_op",
    "",
    (std::vector<Argument>{Argument("tensor"),
                           Argument("input", IntType::get())}),
    (std::vector<Argument>{Argument("output", IntType::get())}));

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithConstructorArg_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithConstructorArgsSchema, kernel<KernelWithConstructorArg>(2), dispatchKey(TensorType1()))
      .op(opWithConstructorArgsSchema, kernel<KernelWithConstructorArg>(4), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::offset_op", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(6, outputs[0].toInt());

  outputs = callOp(*op, dummyTensor(TensorType2()), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(8, outputs[0].toInt());
}

class KernelWithMultipleConstructorArgs final : public OperatorKernel {
public:
  explicit KernelWithMultipleConstructorArgs(int64_t offset1, int64_t offset2)
  : offset_(offset1 + offset2) {}

  int64_t operator()(const Tensor&, int64_t input) {
    return input + offset_;
  }

private:
  int64_t offset_;
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithMultipleConstructorArgs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op(opWithConstructorArgsSchema, kernel<KernelWithMultipleConstructorArgs>(2, 3), dispatchKey(TensorType1()))
      .op(opWithConstructorArgsSchema, kernel<KernelWithMultipleConstructorArgs>(4, 5), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::offset_op", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(9, outputs[0].toInt());

  outputs = callOp(*op, dummyTensor(TensorType2()), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(13, outputs[0].toInt());
}


}
