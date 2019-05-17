#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

using c10::RegisterOperators;
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
using c10::Dict;
using at::Tensor;
using std::string;
using std::unique_ptr;

namespace {

C10_DECLARE_TENSOR_TYPE(TensorType1);
C10_DEFINE_TENSOR_TYPE(TensorType1);
C10_DECLARE_TENSOR_TYPE(TensorType2);
C10_DEFINE_TENSOR_TYPE(TensorType2);

int64_t errorKernel(const Tensor& tensor, int64_t input) {
  EXPECT_TRUE(false); // this kernel should never be called
  return 0;
}

int64_t incrementKernel(const Tensor& tensor, int64_t input) {
  return input + 1;
}

int64_t decrementKernel(const Tensor& tensor, int64_t input) {
  return input - 1;
}

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

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<decltype(incrementKernel), &incrementKernel>(), dispatchKey(TensorType1()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", kernel<decltype(incrementKernel), &incrementKernel>(), dispatchKey(TensorType1()))
      .op("_test::my_op(Tensor dummy, int input) -> int", kernel<decltype(errorKernel), &errorKernel>(), dispatchKey(TensorType2()))
      .op("_test::error(Tensor dummy, int input) -> int", kernel<decltype(errorKernel), &errorKernel>(), dispatchKey(TensorType1()))
      .op("_test::error(Tensor dummy, int input) -> int", kernel<decltype(errorKernel), &errorKernel>(), dispatchKey(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<decltype(incrementKernel), &incrementKernel>(), dispatchKey(TensorType1()));
  auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<decltype(errorKernel), &errorKernel>(), dispatchKey(TensorType2()));
  auto registrar3 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", kernel<decltype(errorKernel), &errorKernel>(), dispatchKey(TensorType1()));
  auto registrar4 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", kernel<decltype(errorKernel), &errorKernel>(), dispatchKey(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<decltype(incrementKernel), &incrementKernel>(), dispatchKey(TensorType1()));
    {
      auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel<decltype(decrementKernel), &decrementKernel>(), dispatchKey(TensorType2()));

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

void kernelWithoutOutput(const Tensor&) {
  was_called = true;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", kernel<decltype(kernelWithoutOutput), &kernelWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_return", "");
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

std::tuple<> kernelWithZeroOutputs(const Tensor&) {
  was_called = true;
  return std::make_tuple();
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", kernel<decltype(kernelWithZeroOutputs), &kernelWithZeroOutputs>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::zero_outputs", "");
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

int64_t kernelWithIntOutput(Tensor, int64_t a, int64_t b) {
  return a + b;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int", kernel<decltype(kernelWithIntOutput), &kernelWithIntOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

Tensor kernelWithTensorOutput(const Tensor& input) {
  return input;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::returning_tensor(Tensor input) -> Tensor", kernel<decltype(kernelWithTensorOutput), &kernelWithTensorOutput>(), dispatchKey(TensorType1()))
      .op("_test::returning_tensor(Tensor input) -> Tensor", kernel<decltype(kernelWithTensorOutput), &kernelWithTensorOutput>(), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::returning_tensor", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType1(), result[0].toTensor().type_id());

  result = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
}

std::vector<Tensor> kernelWithTensorListOutput(const Tensor& input1, const Tensor& input2, const Tensor& input3) {
  return {input1, input2, input3};
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", kernel<decltype(kernelWithTensorListOutput), &kernelWithTensorListOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::list_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), dummyTensor(TensorType2()), dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toTensorListRef().size());
  EXPECT_EQ(TensorType1(), result[0].toTensorListRef()[0].type_id());
  EXPECT_EQ(TensorType2(), result[0].toTensorListRef()[1].type_id());
  EXPECT_EQ(TensorType1(), result[0].toTensorListRef()[2].type_id());
}

std::vector<int64_t> kernelWithIntListOutput(const Tensor&, int64_t input1, int64_t input2, int64_t input3) {
  return {input1, input2, input3};
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", kernel<decltype(kernelWithIntListOutput), &kernelWithIntListOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::list_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), 2, 4, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toIntListRef().size());
  EXPECT_EQ(2, result[0].toIntListRef()[0]);
  EXPECT_EQ(4, result[0].toIntListRef()[1]);
  EXPECT_EQ(6, result[0].toIntListRef()[2]);
}

std::tuple<Tensor, int64_t, std::vector<Tensor>, c10::optional<int64_t>, Dict<string, Tensor>> kernelWithMultipleOutputs(Tensor) {
  Dict<string, Tensor> dict;
  dict.insert("first", dummyTensor(TensorType1()));
  dict.insert("second", dummyTensor(TensorType2()));
  return std::tuple<Tensor, int64_t, std::vector<Tensor>, c10::optional<int64_t>, Dict<string, Tensor>>(
    dummyTensor(TensorType2()),
    5,
    {dummyTensor(TensorType1()), dummyTensor(TensorType2())},
    c10::optional<int64_t>(c10::in_place, 0),
    dict
  );
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", kernel<decltype(kernelWithMultipleOutputs), &kernelWithMultipleOutputs>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::multiple_outputs", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(5, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
  EXPECT_EQ(5, result[1].toInt());
  EXPECT_EQ(2, result[2].toTensorListRef().size());
  EXPECT_EQ(TensorType1(), result[2].toTensorListRef()[0].type_id());
  EXPECT_EQ(TensorType2(), result[2].toTensorListRef()[1].type_id());
  EXPECT_EQ(0, result[3].toInt());
  auto result_dict = c10::impl::toTypedDict<string, Tensor>(std::move(result[4].toGenericDict()->elements()));
  EXPECT_EQ(2, result_dict.size());
  EXPECT_EQ(TensorType1(), result_dict.at("first").type_id());
  EXPECT_EQ(TensorType2(), result_dict.at("second").type_id());
}

Tensor kernelWithTensorInputByReferenceWithOutput(const Tensor& input1) {
  return input1;
}

Tensor kernelWithTensorInputByValueWithOutput(Tensor input1) {
  return input1;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", kernel<decltype(kernelWithTensorInputByReferenceWithOutput), &kernelWithTensorInputByReferenceWithOutput>(), dispatchKey(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> Tensor", kernel<decltype(kernelWithTensorInputByReferenceWithOutput), &kernelWithTensorInputByReferenceWithOutput>(), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType1(), result[0].toTensor().type_id());

  result = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(TensorType2(), result[0].toTensor().type_id());
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", kernel<decltype(kernelWithTensorInputByValueWithOutput), &kernelWithTensorInputByValueWithOutput>(), dispatchKey(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> Tensor", kernel<decltype(kernelWithTensorInputByValueWithOutput), &kernelWithTensorInputByValueWithOutput>(), dispatchKey(TensorType2()));

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

void kernelWithTensorInputByReferenceWithoutOutput(const Tensor& input1) {
  captured_input = input1;
}

void kernelWithTensorInputByValueWithoutOutput(Tensor input1) {
  captured_input = input1;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", kernel<decltype(kernelWithTensorInputByReferenceWithoutOutput), &kernelWithTensorInputByReferenceWithoutOutput>(), dispatchKey(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> ()", kernel<decltype(kernelWithTensorInputByReferenceWithoutOutput), &kernelWithTensorInputByReferenceWithoutOutput>(), dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType1(), captured_input.type_id());

  outputs = callOp(*op, dummyTensor(TensorType2()));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(TensorType2(), captured_input.type_id());
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", kernel<decltype(kernelWithTensorInputByValueWithoutOutput), &kernelWithTensorInputByValueWithoutOutput>(), dispatchKey(TensorType1()))
      .op("_test::tensor_input(Tensor input) -> ()", kernel<decltype(kernelWithTensorInputByValueWithoutOutput), &kernelWithTensorInputByValueWithoutOutput>(), dispatchKey(TensorType2()));

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

void kernelWithIntInputWithoutOutput(Tensor, int64_t input1) {
  captured_int_input = input1;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()", kernel<decltype(kernelWithIntInputWithoutOutput), &kernelWithIntInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  captured_int_input = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_int_input);
}

int64_t kernelWithIntInputWithOutput(Tensor, int64_t input1) {
  return input1 + 1;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int", kernel<decltype(kernelWithIntInputWithOutput), &kernelWithIntInputWithOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

void kernelWithIntListInputWithoutOutput(Tensor, ArrayRef<int64_t> input1) {
  captured_input_list_size = input1.size();
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", kernel<decltype(kernelWithIntListInputWithoutOutput), &kernelWithIntListInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

int64_t kernelWithIntListInputWithOutput(Tensor, ArrayRef<int64_t> input1) {
  return input1.size();
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", kernel<decltype(kernelWithIntListInputWithOutput), &kernelWithIntListInputWithOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(3, outputs[0].toInt());
}

void kernelWithTensorListInputWithoutOutput(ArrayRef<Tensor> input1) {
  captured_input_list_size = input1.size();
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", kernel<decltype(kernelWithTensorListInputWithoutOutput), &kernelWithTensorListInputWithoutOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

int64_t kernelWithTensorListInputWithOutput(ArrayRef<Tensor> input1) {
  return input1.size();
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", kernel<decltype(kernelWithTensorListInputWithOutput), &kernelWithTensorListInputWithOutput>(), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

int captured_dict_size = 0;

void kernelWithDictInputWithoutOutput(Dict<string, Tensor> input1) {
  captured_dict_size = input1.size();
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", kernel<decltype(kernelWithDictInputWithoutOutput), &kernelWithDictInputWithoutOutput>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_input", "");
  ASSERT_TRUE(op.has_value());

  captured_dict_size = 0;
  Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(TensorType1()));
  dict.insert("key2", dummyTensor(TensorType2()));
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_dict_size);
}

string kernelWithDictInputWithOutput(Dict<string, string> input1) {
  return input1.at("key2");
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", kernel<decltype(kernelWithDictInputWithOutput), &kernelWithDictInputWithOutput>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_input", "");
  ASSERT_TRUE(op.has_value());

  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toString()->string());
}

Dict<string, string> kernelWithDictOutput(Dict<string, string> input) {
  return input;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", kernel<decltype(kernelWithDictOutput), &kernelWithDictOutput>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_output", "");
  ASSERT_TRUE(op.has_value());

  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  auto output = c10::impl::toTypedDict<string, string>(std::move(outputs[0].toGenericDict()->elements()));

  EXPECT_EQ(2, output.size());
  EXPECT_EQ("value1", output.at("key1"));
  EXPECT_EQ("value2", output.at("key2"));
}

bool called = false;

void kernelWithoutInputs() {
  called = true;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", kernel<decltype(kernelWithoutInputs), &kernelWithoutInputs>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_tensor_args", "");
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

int64_t kernelWithoutTensorInputs(int64_t arg) {
  return arg + 1;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", kernel<decltype(kernelWithoutTensorInputs), &kernelWithoutTensorInputs>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_tensor_args", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

c10::optional<Tensor> called_arg2;
c10::optional<int64_t> called_arg3;
c10::optional<std::string> called_arg4;

void kernelWithOptInputWithoutOutput(Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
  called = true;
  called_arg2 = arg2;
  called_arg3 = arg3;
  called_arg4 = arg4;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", kernel<decltype(kernelWithOptInputWithoutOutput), &kernelWithOptInputWithoutOutput>(), dispatchKey(TensorType1()));
  auto op = c10::Dispatcher::singleton().findSchema("_test::opt_input", "");
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

c10::optional<Tensor> kernelWithOptInputWithOutput(Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
  called = true;
  called_arg2 = arg2;
  called_arg3 = arg3;
  called_arg4 = arg4;
  return arg2;
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", kernel<decltype(kernelWithOptInputWithOutput), &kernelWithOptInputWithOutput>(), dispatchKey(TensorType1()));
  auto op = c10::Dispatcher::singleton().findSchema("_test::opt_input", "");
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

std::tuple<c10::optional<Tensor>, c10::optional<int64_t>, c10::optional<std::string>>
kernelWithOptInputWithMultipleOutputs(Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
  return std::make_tuple(arg2, arg3, arg4);
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", kernel<decltype(kernelWithOptInputWithMultipleOutputs), &kernelWithOptInputWithMultipleOutputs>(), dispatchKey(TensorType1()));
  auto op = c10::Dispatcher::singleton().findSchema("_test::opt_input", "");
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

std::tuple<int64_t, Tensor> kernelForSchemaInference(Tensor arg1, int64_t arg2, ArrayRef<Tensor> arg3) {
  return {};
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", kernel<decltype(kernelForSchemaInference), &kernelForSchemaInference>());

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_schema_specified", "");
  ASSERT_TRUE(op.has_value());

  c10::assertSchemasHaveSameSignature(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
}

template<class Return, class... Args> struct kernel_func final {
  static Return func(Args...) { return {}; }
};
template<class... Args> struct kernel_func<void, Args...> final {
  static void func(Args...) {}
};

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of arguments is different. Specified 2 but inferred 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of arguments is different. Specified 0 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of arguments is different. Specified 1 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of arguments is different. Specified 3 but inferred 2"
  );
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(), dispatchKey(TensorType1()));
    }, "Type mismatch in argument 2: specified float but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(), dispatchKey(TensorType1()));
    }, "Type mismatch in argument 1: specified int but inferred Tensor"
  );
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 0 but inferred 1"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 2 but inferred 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 1 but inferred 0"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 2 but inferred 0"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 0 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 1 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "The number of returns is different. Specified 3 but inferred 2"
  );
}

TEST(OperatorRegistrationTest_FunctionBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 1: specified Tensor but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 1: specified float but inferred int"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", kernel<decltype(kernel_func<Tensor, Tensor>::func), &kernel_func<Tensor, Tensor>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", kernel<decltype(kernel_func<Tensor, Tensor>::func), &kernel_func<Tensor, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 1: specified float but inferred Tensor"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(), dispatchKey(TensorType1()));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 2: specified float but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(), dispatchKey(TensorType1()));
    }, "Type mismatch in return 1: specified int but inferred Tensor"
  );
}

}
