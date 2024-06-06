#include <gtest/gtest.h>
#include <ATen/core/boxing/impl/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/library.h>

#include <ATen/core/LegacyTypeDispatch.h>

using c10::RegisterOperators;
using c10::DispatchKey;
using c10::Stack;
using std::make_unique;
using c10::intrusive_ptr;
using c10::Dict;
using at::Tensor;
using std::string;
using std::unique_ptr;

namespace {

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

void expectCallsIncrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

void expectCallsDecrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(4, result[0].toInt());
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU));
  expectCallsIncrement(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegisteredWithTorchLibraryAndTorchFn_thenCanBeCalled) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("my_op(Tensor dummy, int input) -> int");
  m.impl("my_op", DispatchKey::CPU, TORCH_FN(incrementKernel));
  expectCallsIncrement(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenCatchAllKernel_whenRegisteredWithTorchLibraryAndTorchFn_thenCanBeCalled) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("my_op(Tensor dummy, int input) -> int", TORCH_FN(incrementKernel));
  expectCallsIncrement(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU)
                                                                                      .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CPU)
                                                                                      .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
  expectCallsIncrement(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU)
                                                                                                                       .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
  auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CPU)
                                                                                                                       .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
  expectCallsIncrement(DispatchKey::CPU);
}


TEST(NewOperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto m = MAKE_TORCH_LIBRARY(_test);
    m.def("_test::my_op(Tensor dummy, int input) -> int");
    auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
    m_cpu.impl("my_op", DispatchKey::CPU, TORCH_FN(incrementKernel));
    {
      auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
      m_cuda.impl("my_op", DispatchKey::CUDA, TORCH_FN(decrementKernel));

      // assert that schema and cpu kernel are present
      expectCallsIncrement(DispatchKey::CPU);
      expectCallsDecrement(DispatchKey::CUDA);
    }

    // now registrar2 is destructed. Assert that schema is still present but cpu kernel is not
    expectCallsIncrement(DispatchKey::CPU);
    expectDoesntFindKernel("_test::my_op", DispatchKey::CUDA);
  }

  // now both registrars are destructed. Assert that the whole schema is gone
  expectDoesntFindOperator("_test::my_op");
}

bool was_called = false;

void kernelWithoutOutput(const Tensor&) {
  was_called = true;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", RegisterOperators::options().kernel<decltype(kernelWithoutOutput), &kernelWithoutOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

std::tuple<> kernelWithZeroOutputs(const Tensor&) {
  was_called = true;
  return std::make_tuple();
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", RegisterOperators::options().kernel<decltype(kernelWithZeroOutputs), &kernelWithZeroOutputs>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

int64_t kernelWithIntOutput(Tensor, int64_t a, int64_t b) {
  return a + b;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int", RegisterOperators::options().kernel<decltype(kernelWithIntOutput), &kernelWithIntOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

Tensor kernelWithTensorOutput(const Tensor& input) {
  return input;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::returning_tensor(Tensor input) -> Tensor", RegisterOperators::options().kernel<decltype(kernelWithTensorOutput), &kernelWithTensorOutput>(DispatchKey::CPU)
                                                                                         .kernel<decltype(kernelWithTensorOutput), &kernelWithTensorOutput>(DispatchKey::CUDA));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

c10::List<Tensor> kernelWithTensorListOutput(const Tensor& input1, const Tensor& input2, const Tensor& input3) {
  return c10::List<Tensor>({input1, input2, input3});
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
    auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", RegisterOperators::options().kernel<decltype(kernelWithTensorListOutput), &kernelWithTensorListOutput>(DispatchKey::CUDA));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toTensorVector().size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[0]));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensorVector()[1]));
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[2]));
}

c10::List<int64_t> kernelWithIntListOutput(const Tensor&, int64_t input1, int64_t input2, int64_t input3) {
  return c10::List<int64_t>({input1, input2, input3});
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", RegisterOperators::options().kernel<decltype(kernelWithIntListOutput), &kernelWithIntListOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 2, 4, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toIntVector().size());
  EXPECT_EQ(2, result[0].toIntVector()[0]);
  EXPECT_EQ(4, result[0].toIntVector()[1]);
  EXPECT_EQ(6, result[0].toIntVector()[2]);
}

std::tuple<Tensor, int64_t, c10::List<Tensor>, std::optional<int64_t>, Dict<string, Tensor>> kernelWithMultipleOutputs(Tensor) {
  Dict<string, Tensor> dict;
  dict.insert("first", dummyTensor(DispatchKey::CPU));
  dict.insert("second", dummyTensor(DispatchKey::CUDA));
  return std::tuple<Tensor, int64_t, c10::List<Tensor>, std::optional<int64_t>, Dict<string, Tensor>>(
    dummyTensor(DispatchKey::CUDA),
    5,
    c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)}),
    std::optional<int64_t>(std::in_place, 0),
    dict
  );
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", RegisterOperators::options().kernel<decltype(kernelWithMultipleOutputs), &kernelWithMultipleOutputs>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::multiple_outputs", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(5, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
  EXPECT_EQ(5, result[1].toInt());
  EXPECT_EQ(2, result[2].toTensorVector().size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[2].toTensorVector()[0]));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[2].toTensorVector()[1]));
  EXPECT_EQ(0, result[3].toInt());
  auto result_dict = c10::impl::toTypedDict<string, Tensor>(result[4].toGenericDict());
  EXPECT_EQ(2, result_dict.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result_dict.at("first")));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result_dict.at("second")));
}

Tensor kernelWithTensorInputByReferenceWithOutput(const Tensor& input1) {
  return input1;
}

Tensor kernelWithTensorInputByValueWithOutput(Tensor input1) {
  return input1;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByReferenceWithOutput), &kernelWithTensorInputByReferenceWithOutput>(DispatchKey::CPU)
                                                                                     .kernel<decltype(kernelWithTensorInputByReferenceWithOutput), &kernelWithTensorInputByReferenceWithOutput>(DispatchKey::CUDA));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByValueWithOutput), &kernelWithTensorInputByValueWithOutput>(DispatchKey::CPU)
                                                                                     .kernel<decltype(kernelWithTensorInputByValueWithOutput), &kernelWithTensorInputByValueWithOutput>(DispatchKey::CUDA));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

Tensor captured_input;

void kernelWithTensorInputByReferenceWithoutOutput(const Tensor& input1) {
  captured_input = input1;
}

void kernelWithTensorInputByValueWithoutOutput(Tensor input1) {
  captured_input = input1;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByReferenceWithoutOutput), &kernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CPU)
                                                                                 .kernel<decltype(kernelWithTensorInputByReferenceWithoutOutput), &kernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CUDA));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByValueWithoutOutput), &kernelWithTensorInputByValueWithoutOutput>(DispatchKey::CPU)
                                                                                 .kernel<decltype(kernelWithTensorInputByValueWithoutOutput), &kernelWithTensorInputByValueWithoutOutput>(DispatchKey::CUDA));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

int64_t captured_int_input = 0;

void kernelWithIntInputWithoutOutput(Tensor, int64_t input1) {
  captured_int_input = input1;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithIntInputWithoutOutput), &kernelWithIntInputWithoutOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_int_input = 0;
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_int_input);
}

int64_t kernelWithIntInputWithOutput(Tensor, int64_t input1) {
  return input1 + 1;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(kernelWithIntInputWithOutput), &kernelWithIntInputWithOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

void kernelWithIntListInputWithoutOutput(Tensor, const c10::List<int64_t>& input1) {
  captured_input_list_size = input1.size();
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithIntListInputWithoutOutput), &kernelWithIntListInputWithoutOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

int64_t kernelWithIntListInputWithOutput(Tensor, const c10::List<int64_t>& input1) {
  return input1.size();
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", RegisterOperators::options().kernel<decltype(kernelWithIntListInputWithOutput), &kernelWithIntListInputWithOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(3, outputs[0].toInt());
}

void kernelWithTensorListInputWithoutOutput(const c10::List<Tensor>& input1) {
  captured_input_list_size = input1.size();
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithTensorListInputWithoutOutput), &kernelWithTensorListInputWithoutOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

int64_t kernelWithTensorListInputWithOutput(const c10::List<Tensor>& input1) {
  return input1.size();
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", RegisterOperators::options().kernel<decltype(kernelWithTensorListInputWithOutput), &kernelWithTensorListInputWithOutput>(DispatchKey::CPU));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

int captured_dict_size = 0;

void kernelWithDictInputWithoutOutput(Dict<string, Tensor> input1) {
  captured_dict_size = input1.size();
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictInputWithoutOutput), &kernelWithDictInputWithoutOutput>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_dict_size = 0;
  Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(DispatchKey::CPU));
  dict.insert("key2", dummyTensor(DispatchKey::CUDA));
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_dict_size);
}

string kernelWithDictInputWithOutput(Dict<string, string> input1) {
  return input1.at("key2");
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictInputWithOutput), &kernelWithDictInputWithOutput>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toStringRef());
}

Dict<string, string> kernelWithDictOutput(Dict<string, string> input) {
  return input;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictOutput), &kernelWithDictOutput>());

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

bool called = false;

void kernelWithoutInputs() {
  called = true;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<decltype(kernelWithoutInputs), &kernelWithoutInputs>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

int64_t kernelWithoutTensorInputs(int64_t arg) {
  return arg + 1;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<decltype(kernelWithoutTensorInputs), &kernelWithoutTensorInputs>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

std::optional<Tensor> called_arg2 = c10::nullopt;
std::optional<int64_t> called_arg3 = c10::nullopt;
std::optional<std::string> called_arg4 = c10::nullopt;

void kernelWithOptInputWithoutOutput(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  called = true;
  called_arg2 = arg2;
  called_arg3 = arg3;
  called_arg4 = arg4;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithoutOutput), &kernelWithOptInputWithoutOutput>(DispatchKey::CPU));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(0, outputs.size());

  EXPECT_TRUE(called);
  EXPECT_TRUE(called_arg2.has_value());
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
  EXPECT_FALSE(called_arg3.has_value());
  EXPECT_TRUE(called_arg4.has_value());
  EXPECT_EQ(*called_arg4, "text");

  called = false;
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(0, outputs.size());

  EXPECT_TRUE(called);
  EXPECT_FALSE(called_arg2.has_value());
  EXPECT_TRUE(called_arg3.has_value());
  EXPECT_EQ(*called_arg3, 4);
  EXPECT_FALSE(called_arg4.has_value());
}

std::optional<Tensor> kernelWithOptInputWithOutput(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  called = true;
  called_arg2 = arg2;
  called_arg3 = arg3;
  called_arg4 = arg4;
  return arg2;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithOutput), &kernelWithOptInputWithOutput>(DispatchKey::CPU));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));

  EXPECT_TRUE(called);
  EXPECT_TRUE(called_arg2.has_value());
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
  EXPECT_FALSE(called_arg3.has_value());
  EXPECT_TRUE(called_arg4.has_value());
  EXPECT_EQ(*called_arg4, "text");

  called = false;
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(1, outputs.size());
  EXPECT_TRUE(outputs[0].isNone());

  EXPECT_TRUE(called);
  EXPECT_FALSE(called_arg2.has_value());
  EXPECT_TRUE(called_arg3.has_value());
  EXPECT_EQ(*called_arg3, 4);
  EXPECT_FALSE(called_arg4.has_value());
}

std::tuple<std::optional<Tensor>, std::optional<int64_t>, std::optional<std::string>>
kernelWithOptInputWithMultipleOutputs(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  return std::make_tuple(arg2, arg3, arg4);
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithMultipleOutputs), &kernelWithOptInputWithMultipleOutputs>(DispatchKey::CPU));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(3, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));
  EXPECT_TRUE(outputs[1].isNone());
  EXPECT_EQ("text", outputs[2].toStringRef());

  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(3, outputs.size());
  EXPECT_TRUE(outputs[0].isNone());
  EXPECT_EQ(4, outputs[1].toInt());
  EXPECT_TRUE(outputs[2].isNone());
}

std::string concatKernel(const Tensor& tensor1, std::string a, const std::string& b, int64_t c) {
  return a + b + std::to_string(c);
}

void expectCallsConcatUnboxed(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, int64_t>(*op, dummyTensor(dispatch_key), "1", "2", 3);
  EXPECT_EQ("123", result);
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegistered_thenCanBeCalledUnboxed) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", RegisterOperators::options().kernel<decltype(concatKernel), &concatKernel>(DispatchKey::CPU));
  expectCallsConcatUnboxed(DispatchKey::CPU);
}

std::tuple<int64_t, Tensor> kernelForSchemaInference(Tensor arg1, int64_t arg2, const c10::List<Tensor>& arg3) {
  return {};
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel<decltype(kernelForSchemaInference), &kernelForSchemaInference>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  ASSERT_TRUE(op.has_value());

  std::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  EXPECT_FALSE(differences.has_value());
}

template<class Return, class... Args> struct kernel_func final {
  static Return func(Args...) { return {}; }
};
template<class... Args> struct kernel_func<void, Args...> final {
  static void func(Args...) {}
};

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "The number of arguments is different. 2 vs 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
    }, "The number of arguments is different. 0 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
    }, "The number of arguments is different. 1 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
    }, "The number of arguments is different. 3 vs 2"
  );
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(DispatchKey::CPU));
    }, "Type mismatch in argument 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(DispatchKey::CPU));
    }, "Type mismatch in argument 1: int vs Tensor"
  );
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 0 vs 1"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 2 vs 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 1 vs 0"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 2 vs 0"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 0 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 1 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 3 vs 2"
  );
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 1: Tensor vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 1: float vs int"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<Tensor, Tensor>::func), &kernel_func<Tensor, Tensor>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<decltype(kernel_func<Tensor, Tensor>::func), &kernel_func<Tensor, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 1: float vs Tensor"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(DispatchKey::CPU));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 1: int vs Tensor"
  );
}

}
