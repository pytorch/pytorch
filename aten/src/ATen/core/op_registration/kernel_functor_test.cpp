#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

using c10::RegisterOperators;
using c10::OperatorKernel;
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
  int64_t operator()(const Tensor&, int64_t) {
    EXPECT_TRUE(false); // this kernel should never be called
    return 0;
  }
};

struct IncrementKernel final : OperatorKernel {
  int64_t operator()(const Tensor& tensor, int64_t input) {
    return input + 1;
  }
};

struct DecrementKernel final : OperatorKernel {
  int64_t operator()(const Tensor& tensor, int64_t input) {
    return input - 1;
  }
};

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
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<IncrementKernel>(), dispatchKey(TensorType1()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", kernel<IncrementKernel>(), dispatchKey(TensorType1()))
      .op("_test::my_op(Tensor dummy, int input) -> int", kernel<ErrorKernel>(), dispatchKey(TensorType2()))
      .op("_test::error(Tensor dummy, int input) -> int", kernel<ErrorKernel>(), dispatchKey(TensorType1()))
      .op("_test::error(Tensor dummy, int input) -> int", kernel<ErrorKernel>(), dispatchKey(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<IncrementKernel>(), dispatchKey(TensorType1()));
  auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<ErrorKernel>(), dispatchKey(TensorType2()));
  auto registrar3 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", kernel<ErrorKernel>(), dispatchKey(TensorType1()));
  auto registrar4 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", kernel<ErrorKernel>(), dispatchKey(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<IncrementKernel>(), dispatchKey(TensorType1()));
    {
      auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<DecrementKernel>(), dispatchKey(TensorType2()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", kernel<KernelWithoutOutput>(), dispatchKey(TensorType1()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", kernel<KernelWithZeroOutputs>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::zero_outputs", "");
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

struct KernelWithIntOutput final : OperatorKernel {
  int64_t operator()(Tensor, int64_t a, int64_t b) {
    return a + b;
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int", kernel<KernelWithIntOutput>(), dispatchKey(TensorType1()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::returning_tensor(Tensor input) -> Tensor", kernel<KernelWithTensorOutput>(), dispatchKey(TensorType1()))
      .op("_test::returning_tensor(Tensor input) -> Tensor", kernel<KernelWithTensorOutput>(), dispatchKey(TensorType2()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", kernel<KernelWithTensorListOutput>(), dispatchKey(TensorType1()));

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
  std::vector<int64_t> operator()(const Tensor&, int64_t input1, int64_t input2, int64_t input3) {
    return {input1, input2, input3};
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", kernel<KernelWithIntListOutput>(), dispatchKey(TensorType1()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[])", kernel<KernelWithMultipleOutputs>(), dispatchKey(TensorType1()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", kernel<KernelWithTensorInputByReferenceWithOutput>(), dispatchKey(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> Tensor", kernel<KernelWithTensorInputByReferenceWithOutput>(), dispatchKey(TensorType2()));

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
      .op("_test::tensor_input(Tensor input) -> Tensor", kernel<KernelWithTensorInputByValueWithOutput>(), dispatchKey(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> Tensor", kernel<KernelWithTensorInputByValueWithOutput>(), dispatchKey(TensorType2()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", kernel<KernelWithTensorInputByReferenceWithoutOutput>(), dispatchKey(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> ()", kernel<KernelWithTensorInputByReferenceWithoutOutput>(), dispatchKey(TensorType2()));

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
      .op("_test::tensor_input(Tensor input) -> ()", kernel<KernelWithTensorInputByValueWithoutOutput>(), dispatchKey(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> ()", kernel<KernelWithTensorInputByValueWithoutOutput>(), dispatchKey(TensorType2()));

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

struct KernelWithIntInputWithoutOutput final : OperatorKernel {
  void operator()(Tensor, int64_t input1) {
    captured_int_input = input1;
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()", kernel<KernelWithIntInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  captured_int_input = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_int_input);
}

struct KernelWithIntInputWithOutput final : OperatorKernel {
  int64_t operator()(Tensor, int64_t input1) {
    return input1 + 1;
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int", kernel<KernelWithIntInputWithOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

struct KernelWithIntListInputWithoutOutput final : OperatorKernel {
  void operator()(Tensor, ArrayRef<int64_t> input1) {
    captured_input_list_size = input1.size();
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", kernel<KernelWithIntListInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

struct KernelWithIntListInputWithOutput final : OperatorKernel {
  int64_t operator()(Tensor, ArrayRef<int64_t> input1) {
    return input1.size();
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", kernel<KernelWithIntListInputWithOutput>(), dispatchKey(TensorType1()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", kernel<KernelWithTensorListInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

struct KernelWithTensorListInputWithOutput final : OperatorKernel {
  int64_t operator()(ArrayRef<Tensor> input1) {
    return input1.size();
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", kernel<KernelWithTensorListInputWithOutput>(), dispatchKey(TensorType1()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithCache_thenCacheIsKeptCorrectly) {
  auto registrar = RegisterOperators()
      .op("_test::cache_op(Tensor input) -> int", kernel<KernelWithCache>(), dispatchKey(TensorType1()));

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

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithConstructorArg_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::offset_op(Tensor tensor, int input) -> int", kernel<KernelWithConstructorArg>(2), dispatchKey(TensorType1()))
      .op("_test::offset_op(Tensor tensor, int input) -> int", kernel<KernelWithConstructorArg>(4), dispatchKey(TensorType2()));

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
      .op("_test::offset_op(Tensor tensor, int input) -> int", kernel<KernelWithMultipleConstructorArgs>(2, 3), dispatchKey(TensorType1()))
      .op("_test::offset_op(Tensor tensor, int input) -> int", kernel<KernelWithMultipleConstructorArgs>(4, 5), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::offset_op", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(9, outputs[0].toInt());

  outputs = callOp(*op, dummyTensor(TensorType2()), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(13, outputs[0].toInt());
}

bool called = false;

struct KernelWithoutInputs final : OperatorKernel {
  void operator()() {
    called = true;
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", kernel<KernelWithoutInputs>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_tensor_args", "");
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

struct KernelWithoutTensorInputs final : OperatorKernel {
  int64_t operator()(int64_t arg) {
    return arg + 1;
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", kernel<KernelWithoutTensorInputs>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_tensor_args", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

struct KernelForSchemaInference final : OperatorKernel {
  std::tuple<int64_t, Tensor> operator()(Tensor arg1, int64_t arg2, ArrayRef<Tensor> arg3) {
    return {};
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", kernel<KernelForSchemaInference>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_schema_specified", "");
  ASSERT_TRUE(op.has_value());

  c10::assertSchemasHaveSameSignature(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
}

template<class Return, class... Args> struct KernelFunc final : OperatorKernel{
  Return operator()(Args...) { return {}; }
};
template<class... Args> struct KernelFunc<void, Args...> final : OperatorKernel {
  void operator()(Args...) {}
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", kernel<KernelFunc<int64_t, Tensor>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", kernel<KernelFunc<int64_t, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of arguments is different. Specified 2 but inferred 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", kernel<KernelFunc<void, Tensor, Tensor>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", kernel<KernelFunc<void, Tensor, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of arguments is different. Specified 0 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", kernel<KernelFunc<void, Tensor, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of arguments is different. Specified 1 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", kernel<KernelFunc<void, Tensor, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of arguments is different. Specified 3 but inferred 2"
  );
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", kernel<KernelFunc<int64_t, Tensor, int64_t>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", kernel<KernelFunc<int64_t, Tensor, int64_t>>(), dispatchKey(TensorType1()));
    }, "Type mismatch in argument 2: specified float but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", kernel<KernelFunc<int64_t, Tensor, int64_t>>(), dispatchKey(TensorType1()));
    }, "Type mismatch in argument 1: specified int but inferred Tensor"
  );
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", kernel<KernelFunc<int64_t, Tensor>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", kernel<KernelFunc<int64_t, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 0 but inferred 1"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", kernel<KernelFunc<int64_t, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 2 but inferred 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", kernel<KernelFunc<void, Tensor>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", kernel<KernelFunc<void, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 1 but inferred 0"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", kernel<KernelFunc<void, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 2 but inferred 0"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 0 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 1 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 3 but inferred 2"
  );
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", kernel<KernelFunc<int64_t, Tensor>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", kernel<KernelFunc<int64_t, Tensor>>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 1: specified Tensor but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", kernel<KernelFunc<int64_t, Tensor>>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 1: specified float but inferred int"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", kernel<KernelFunc<Tensor, Tensor>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", kernel<KernelFunc<Tensor, Tensor>>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 1: specified float but inferred Tensor"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 2: specified float but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 1: specified int but inferred Tensor"
  );
}

}
