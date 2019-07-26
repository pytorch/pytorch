#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

using c10::RegisterOperators;
using c10::OperatorKernel;
using c10::TensorTypeId;
using c10::KernelCache;
using c10::Stack;
using c10::guts::make_unique;
using c10::intrusive_ptr;
using c10::Dict;
using at::Tensor;
using std::unique_ptr;
using std::string;

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
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

void expectCallsDecrement(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(4, result[0].toInt());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(TensorType1()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(TensorType1()))
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(TensorType2()))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(TensorType1()))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(TensorType1()));
  auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(TensorType2()));
  auto registrar3 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(TensorType1()));
  auto registrar4 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(TensorType1()));
    {
      auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<DecrementKernel>(TensorType2()));

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
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", RegisterOperators::options().kernel<KernelWithoutOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
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
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", RegisterOperators::options().kernel<KernelWithZeroOutputs>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
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
      .op("_test::int_output(Tensor dummy, int a, int b) -> int", RegisterOperators::options().kernel<KernelWithIntOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
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
      .op("_test::returning_tensor(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorOutput>(TensorType1()))
      .op("_test::returning_tensor(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorOutput>(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType1(), result[0].toTensor().type_id());

  result = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
}

struct KernelWithTensorListOutput final : OperatorKernel {
  c10::List<Tensor> operator()(const Tensor& input1, const Tensor& input2, const Tensor& input3) {
    return c10::List<Tensor>({input1, input2, input3});
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", RegisterOperators::options().kernel<KernelWithTensorListOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), dummyTensor(TensorType2()), dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toTensorListRef().size());
  EXPECT_EQ(TensorType1(), result[0].toTensorListRef()[0].type_id());
  EXPECT_EQ(TensorType2(), result[0].toTensorListRef()[1].type_id());
  EXPECT_EQ(TensorType1(), result[0].toTensorListRef()[2].type_id());
}

struct KernelWithIntListOutput final : OperatorKernel {
  c10::List<int64_t> operator()(const Tensor&, int64_t input1, int64_t input2, int64_t input3) {
    return c10::List<int64_t>({input1, input2, input3});
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", RegisterOperators::options().kernel<KernelWithIntListOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), 2, 4, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toIntListRef().size());
  EXPECT_EQ(2, result[0].toIntListRef()[0]);
  EXPECT_EQ(4, result[0].toIntListRef()[1]);
  EXPECT_EQ(6, result[0].toIntListRef()[2]);
}

struct KernelWithMultipleOutputs final : OperatorKernel {
  std::tuple<Tensor, int64_t, c10::List<Tensor>, c10::optional<int64_t>, Dict<string, Tensor>> operator()(Tensor) {
    Dict<string, Tensor> dict;
    dict.insert("first", dummyTensor(TensorType1()));
    dict.insert("second", dummyTensor(TensorType2()));
    return std::tuple<Tensor, int64_t, c10::List<Tensor>, c10::optional<int64_t>, Dict<string, Tensor>>(
      dummyTensor(TensorType2()),
      5,
      c10::List<Tensor>({dummyTensor(TensorType1()), dummyTensor(TensorType2())}),
      c10::optional<int64_t>(c10::in_place, 0),
      dict
    );
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", RegisterOperators::options().kernel<KernelWithMultipleOutputs>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::multiple_outputs", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(5, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
  EXPECT_EQ(5, result[1].toInt());
  EXPECT_EQ(2, result[2].toTensorListRef().size());
  EXPECT_EQ(TensorType1(), result[2].toTensorListRef()[0].type_id());
  EXPECT_EQ(TensorType2(), result[2].toTensorListRef()[1].type_id());
  EXPECT_EQ(0, result[3].toInt());
  auto result_dict = c10::impl::toTypedDict<string, Tensor>(result[4].toGenericDict());
  EXPECT_EQ(2, result_dict.size());
  EXPECT_EQ(TensorType1(), result_dict.at("first").type_id());
  EXPECT_EQ(TensorType2(), result_dict.at("second").type_id());
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
      .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorInputByReferenceWithOutput>(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorInputByReferenceWithOutput>(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
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
      .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorInputByValueWithOutput>(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorInputByValueWithOutput>(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
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
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<KernelWithTensorInputByReferenceWithoutOutput>(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<KernelWithTensorInputByReferenceWithoutOutput>(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
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
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<KernelWithTensorInputByValueWithoutOutput>(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<KernelWithTensorInputByValueWithoutOutput>(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
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
      .op("_test::int_input(Tensor dummy, int input) -> ()", RegisterOperators::options().kernel<KernelWithIntInputWithoutOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
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
      .op("_test::int_input(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<KernelWithIntInputWithOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

struct KernelWithIntListInputWithoutOutput final : OperatorKernel {
  void operator()(Tensor, const c10::List<int64_t>& input1) {
    captured_input_list_size = input1.size();
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", RegisterOperators::options().kernel<KernelWithIntListInputWithoutOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

struct KernelWithIntListInputWithOutput final : OperatorKernel {
  int64_t operator()(Tensor, const c10::List<int64_t>& input1) {
    return input1.size();
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", RegisterOperators::options().kernel<KernelWithIntListInputWithOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(3, outputs[0].toInt());
}

struct KernelWithTensorListInputWithoutOutput final : OperatorKernel {
  void operator()(const c10::List<Tensor>& input1) {
    captured_input_list_size = input1.size();
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", RegisterOperators::options().kernel<KernelWithTensorListInputWithoutOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

struct KernelWithTensorListInputWithOutput final : OperatorKernel {
  int64_t operator()(const c10::List<Tensor>& input1) {
    return input1.size();
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", RegisterOperators::options().kernel<KernelWithTensorListInputWithOutput>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

int captured_dict_size = 0;

struct KernelWithDictInputWithoutOutput final : OperatorKernel {
  void operator()(Dict<string, Tensor> input1) {
    captured_dict_size = input1.size();
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel<KernelWithDictInputWithoutOutput>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_dict_size = 0;
  Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(TensorType1()));
  dict.insert("key2", dummyTensor(TensorType2()));
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_dict_size);
}

struct KernelWithDictInputWithOutput final : OperatorKernel {
  string operator()(Dict<string, string> input1) {
    return input1.at("key2");
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel<KernelWithDictInputWithOutput>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toString()->string());
}

struct KernelWithDictOutput final : OperatorKernel {
  Dict<string, string> operator()(Dict<string, string> input) {
    return input;
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel<KernelWithDictOutput>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  ASSERT_TRUE(op.has_value());

  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());

  EXPECT_EQ(2, output.size());
  EXPECT_EQ("value1", output.at("key1"));
  EXPECT_EQ("value2", output.at("key2"));
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
      .op("_test::cache_op(Tensor input) -> int", RegisterOperators::options().kernel<KernelWithCache>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::cache_op", ""});
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
      .op("_test::offset_op(Tensor tensor, int input) -> int", RegisterOperators::options().kernel<KernelWithConstructorArg>(TensorType1(), 2))
      .op("_test::offset_op(Tensor tensor, int input) -> int", RegisterOperators::options().kernel<KernelWithConstructorArg>(TensorType2(), 4));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::offset_op", ""});
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
      .op("_test::offset_op(Tensor tensor, int input) -> int", RegisterOperators::options().kernel<KernelWithMultipleConstructorArgs>(TensorType1(), 2, 3))
      .op("_test::offset_op(Tensor tensor, int input) -> int", RegisterOperators::options().kernel<KernelWithMultipleConstructorArgs>(TensorType2(), 4, 5));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::offset_op", ""});
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
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<KernelWithoutInputs>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
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
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<KernelWithoutTensorInputs>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

c10::optional<Tensor> called_arg2;
c10::optional<int64_t> called_arg3;
c10::optional<std::string> called_arg4;

struct KernelWithOptInputWithoutOutput final : OperatorKernel {
  void operator()(Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
    called = true;
    called_arg2 = arg2;
    called_arg3 = arg3;
    called_arg4 = arg4;
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", RegisterOperators::options().kernel<KernelWithOptInputWithoutOutput>(TensorType1()));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), dummyTensor(TensorType2()), c10::IValue(), std::string("text"));
  EXPECT_EQ(0, outputs.size());

  EXPECT_TRUE(called);
  EXPECT_TRUE(called_arg2.has_value());
  EXPECT_EQ(called_arg2->type_id(), TensorType2());
  EXPECT_FALSE(called_arg3.has_value());
  EXPECT_TRUE(called_arg4.has_value());
  EXPECT_EQ(*called_arg4, "text");

  called = false;
  outputs = callOp(*op, dummyTensor(TensorType1()), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(0, outputs.size());

  EXPECT_TRUE(called);
  EXPECT_FALSE(called_arg2.has_value());
  EXPECT_TRUE(called_arg3.has_value());
  EXPECT_EQ(*called_arg3, 4);
  EXPECT_FALSE(called_arg4.has_value());
}

struct KernelWithOptInputWithOutput final : OperatorKernel {
  c10::optional<Tensor> operator()(Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
    called = true;
    called_arg2 = arg2;
    called_arg3 = arg3;
    called_arg4 = arg4;
    return arg2;
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", RegisterOperators::options().kernel<KernelWithOptInputWithOutput>(TensorType1()));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), dummyTensor(TensorType2()), c10::IValue(), std::string("text"));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(TensorType2(), outputs[0].toTensor().type_id());

  EXPECT_TRUE(called);
  EXPECT_TRUE(called_arg2.has_value());
  EXPECT_EQ(called_arg2->type_id(), TensorType2());
  EXPECT_FALSE(called_arg3.has_value());
  EXPECT_TRUE(called_arg4.has_value());
  EXPECT_EQ(*called_arg4, "text");

  called = false;
  outputs = callOp(*op, dummyTensor(TensorType1()), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(1, outputs.size());
  EXPECT_TRUE(outputs[0].isNone());

  EXPECT_TRUE(called);
  EXPECT_FALSE(called_arg2.has_value());
  EXPECT_TRUE(called_arg3.has_value());
  EXPECT_EQ(*called_arg3, 4);
  EXPECT_FALSE(called_arg4.has_value());
}

struct KernelWithOptInputWithMultipleOutputs final : OperatorKernel {
  std::tuple<c10::optional<Tensor>, c10::optional<int64_t>, c10::optional<std::string>>
  operator()(Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
    return std::make_tuple(arg2, arg3, arg4);
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", RegisterOperators::options().kernel<KernelWithOptInputWithMultipleOutputs>(TensorType1()));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), dummyTensor(TensorType2()), c10::IValue(), std::string("text"));
  EXPECT_EQ(3, outputs.size());
  EXPECT_EQ(TensorType2(), outputs[0].toTensor().type_id());
  EXPECT_TRUE(outputs[1].isNone());
  EXPECT_EQ("text", outputs[2].toString()->string());

  outputs = callOp(*op, dummyTensor(TensorType1()), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(3, outputs.size());
  EXPECT_TRUE(outputs[0].isNone());
  EXPECT_EQ(4, outputs[1].toInt());
  EXPECT_TRUE(outputs[2].isNone());
}

struct ConcatKernel final : OperatorKernel {
  explicit ConcatKernel(std::string prefix): prefix_(std::move(prefix)) {}

  std::string operator()(const Tensor& tensor1, std::string a, const std::string& b, int64_t c) {
    return prefix_ + a + b + c10::guts::to_string(c);
  }

  std::string prefix_;
};

void expectCallsConcatUnboxed(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, int64_t>(*op, type_id, dummyTensor(type_id), "1", "2", 3);
  EXPECT_EQ("prefix123", result);
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegistered_thenCanBeCalledUnboxed) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", RegisterOperators::options().kernel<ConcatKernel>(TensorType1(), "prefix"));
  expectCallsConcatUnboxed(TensorType1());
}

struct KernelForSchemaInference final : OperatorKernel {
  std::tuple<int64_t, Tensor> operator()(Tensor arg1, int64_t arg2, const c10::List<Tensor>& arg3) {
    return {};
  }
};

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", RegisterOperators::options().kernel<KernelForSchemaInference>(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  ASSERT_TRUE(op.has_value());

  c10::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  EXPECT_FALSE(differences.has_value());
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenKernel_whenRegisteredCatchAllWithoutSpecifyingSchema_thenInfersSchema) {
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel<KernelForSchemaInference>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  ASSERT_TRUE(op.has_value());

  c10::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  EXPECT_FALSE(differences.has_value());
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
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(TensorType1()));
    }, "The number of arguments is different. 2 vs 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(TensorType1()));
    }, "The number of arguments is different. 0 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(TensorType1()));
    }, "The number of arguments is different. 1 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(TensorType1()));
    }, "The number of arguments is different. 3 vs 2"
  );
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor, int64_t>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor, int64_t>>(TensorType1()));
    }, "Type mismatch in argument 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor, int64_t>>(TensorType1()));
    }, "Type mismatch in argument 1: int vs Tensor"
  );
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(TensorType1()));
    }, "The number of returns is different. 0 vs 1"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(TensorType1()));
    }, "The number of returns is different. 2 vs 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(TensorType1()));
    }, "The number of returns is different. 1 vs 0"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(TensorType1()));
    }, "The number of returns is different. 2 vs 0"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(TensorType1()));
    }, "The number of returns is different. 0 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(TensorType1()));
    }, "The number of returns is different. 1 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(TensorType1()));
    }, "The number of returns is different. 3 vs 2"
  );
}

TEST(OperatorRegistrationTest_FunctorBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(TensorType1()));
    }, "Type mismatch in return 1: Tensor vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(TensorType1()));
    }, "Type mismatch in return 1: float vs int"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<Tensor, Tensor>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<KernelFunc<Tensor, Tensor>>(TensorType1()));
    }, "Type mismatch in return 1: float vs Tensor"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(TensorType1()));
    }, "Type mismatch in return 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(TensorType1()));
    }, "Type mismatch in return 1: int vs Tensor"
  );
}

}
