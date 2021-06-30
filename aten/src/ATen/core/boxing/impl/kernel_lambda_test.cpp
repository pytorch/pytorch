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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t i) {return i+1;}));
  expectCallsIncrement(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenOutOfLineKernel_whenRegistered_thenCanBeCalled) {
  auto my_kernel = [] (Tensor, int64_t i) {return i+1;};
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, my_kernel));
  expectCallsIncrement(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t i) {return i+1;})
                                                                                      .kernel(DispatchKey::CUDA, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;}))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;})
                                                                                      .kernel(DispatchKey::CUDA, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;}));
  expectCallsIncrement(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t i) {return i+1;})
                                                                                                                       .kernel(DispatchKey::CUDA, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;}));
  auto registrar3 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;})
                                                                                                                       .kernel(DispatchKey::CUDA, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;}));
  expectCallsIncrement(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto m = MAKE_TORCH_LIBRARY(_test);
    m.def("_test::my_op(Tensor dummy, int input) -> int");
    auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
    m_cpu.impl("my_op", DispatchKey::CPU, [] (Tensor, int64_t i) {return i+1;});
    {
      auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
      m_cuda.impl("my_op", DispatchKey::CUDA, [] (Tensor, int64_t i) {return i-1;});

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool was_called = false;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()",
    RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const Tensor&) -> void {was_called = true;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()",
    RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor&) -> std::tuple<> {was_called = true; return {};}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t a, int64_t b) {return a+b;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::returning_tensor(Tensor input) -> Tensor",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) {return a;})
                                    .kernel(DispatchKey::CUDA, [] (const Tensor& a) {return a;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]",
        RegisterOperators::options().kernel(DispatchKey::CUDA, [] (const Tensor& a, const Tensor& b, const Tensor& c) -> c10::List<Tensor> {return c10::List<Tensor>({a, b, c});}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toTensorVector().size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[0]));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensorVector()[1]));
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[2]));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor&, int64_t a, int64_t b, int64_t c) -> c10::List<int64_t> {return c10::List<int64_t>({a,b,c});}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 2, 4, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toIntVector().size());
  EXPECT_EQ(2, result[0].toIntVector()[0]);
  EXPECT_EQ(4, result[0].toIntVector()[1]);
  EXPECT_EQ(6, result[0].toIntVector()[2]);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))",
       RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, int64_t, c10::List<Tensor>, c10::optional<int64_t>, Dict<string, Tensor>> {
         Dict<string, Tensor> dict;
         dict.insert("first", dummyTensor(DispatchKey::CPU));
         dict.insert("second", dummyTensor(DispatchKey::CUDA));
         return std::tuple<Tensor, int64_t, c10::List<Tensor>, c10::optional<int64_t>, Dict<string, Tensor>>(
           dummyTensor(DispatchKey::CUDA),
           5,
           c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)}),
           c10::optional<int64_t>(c10::in_place, 0),
           dict
         );
       }));

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) {return a;})
                                    .kernel(DispatchKey::CUDA, [] (const Tensor& a) {return a;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor a) {return a;})
                                    .kernel(DispatchKey::CUDA, [] (Tensor a) {return a;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
Tensor captured_input;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) -> void {captured_input = a;})
                                    .kernel(DispatchKey::CUDA, [] (const Tensor& a) -> void {captured_input = a;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor a) -> void {captured_input = a;})
                                    .kernel(DispatchKey::CUDA, [] (Tensor a) -> void {captured_input = a;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int64_t captured_int_input = 0;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t a) -> void {captured_int_input = a;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_int_input = 0;
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_int_input);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t a) {return a + 1;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int64_t captured_input_list_size = 0;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, const c10::List<int64_t>& a) {captured_input_list_size = a.size();}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, const c10::List<int64_t>& a) -> int64_t {return a.size();}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(3, outputs[0].toInt());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const c10::List<Tensor>& a) -> void {captured_input_list_size = a.size();}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int",
         RegisterOperators::options().kernel(DispatchKey::CPU, [] (const c10::List<Tensor>& a) -> int64_t {return a.size();}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int captured_dict_size = 0;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel([] (Dict<string, Tensor> input1) {
        captured_dict_size = input1.size();
      }));

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel([] (Dict<string, string> input1) {
        return input1.at("key2");
      }));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toString()->string());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
    .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel([] (Dict<string, string> input) {
      return input;
    }));

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool called = false;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel([] () {called = true;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel([] (int64_t arg) {return arg + 1;}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
c10::optional<Tensor> called_arg2 = c10::nullopt;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
c10::optional<int64_t> called_arg3 = c10::nullopt;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
c10::optional<std::string> called_arg4 = c10::nullopt;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()",
    RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
      called = true;
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
    }));
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?",
    RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
      called = true;
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
      return arg2;
    }));
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)",
    RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
      return std::make_tuple(arg2, arg3, arg4);
    }));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(3, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));
  EXPECT_TRUE(outputs[1].isNone());
  EXPECT_EQ("text", outputs[2].toString()->string());

  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(3, outputs.size());
  EXPECT_TRUE(outputs[0].isNone());
  EXPECT_EQ(4, outputs[1].toInt());
  EXPECT_TRUE(outputs[2].isNone());
}

void expectCallsConcatUnboxed(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, int64_t>(*op, dummyTensor(dispatch_key), "1", "2", 3);
  EXPECT_EQ("123", result);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernel_whenRegistered_thenCanBeCalledUnboxed) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", torch::RegisterOperators::options()
    .kernel(DispatchKey::CPU, [] (const Tensor& tensor1, std::string a, const std::string& b, int64_t c) {
      return a + b + c10::guts::to_string(c);
    }));
  expectCallsConcatUnboxed(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel([] (Tensor arg1, int64_t arg2, const c10::List<Tensor>& arg3) -> std::tuple<int64_t, Tensor> {return {};}));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  ASSERT_TRUE(op.has_value());

  c10::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  EXPECT_FALSE(differences.has_value());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "The number of arguments is different. 2 vs 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
    }, "The number of arguments is different. 0 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
    }, "The number of arguments is different. 1 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
    }, "The number of arguments is different. 3 vs 2"
  );
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {return {};}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {return {};}));
    }, "Type mismatch in argument 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {return {};}));
    }, "Type mismatch in argument 1: int vs Tensor"
  );
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "The number of returns is different. 0 vs 1"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "The number of returns is different. 2 vs 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));
    }, "The number of returns is different. 1 vs 0"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));
    }, "The number of returns is different. 2 vs 0"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
    }, "The number of returns is different. 0 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
    }, "The number of returns is different. 1 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
    }, "The number of returns is different. 3 vs 2"
  );
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_LambdaBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "Type mismatch in return 1: Tensor vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "Type mismatch in return 1: float vs int"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> Tensor {return {};}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> Tensor {return {};}));
    }, "Type mismatch in return 1: float vs Tensor"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};}));

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};}));
    }, "Type mismatch in return 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};}));
    }, "Type mismatch in return 1: int vs Tensor"
  );
}

}
