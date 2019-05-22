#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

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
using c10::TensorTypeId;
using c10::KernelCache;
using c10::Stack;
using c10::guts::make_unique;
using c10::ivalue::TensorList;
using c10::ivalue::IntList;
using c10::intrusive_ptr;
using c10::Dict;
using at::Tensor;
using std::string;
using std::unique_ptr;

namespace {

C10_DECLARE_TENSOR_TYPE(TensorType1);
C10_DEFINE_TENSOR_TYPE(TensorType1);
C10_DECLARE_TENSOR_TYPE(TensorType2);
C10_DEFINE_TENSOR_TYPE(TensorType2);

void expectCallsIncrement(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema("_test::my_op", "");
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernel_whenRegisteredInConstructor_thenCanBeCalled) {
  auto registrar = RegisterOperators("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
          return input + 1;
        })
      .op("_test::error(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
          EXPECT_TRUE(false); // this kernel should never be called
          return 0;
        });
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
      EXPECT_TRUE(false); // this kernel should never be called
      return 0;
    });
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
        return input + 1;
      });

    expectCallsIncrement(TensorType1());
  }

  // now the registrar is destructed. Assert that the schema is gone.
  expectDoesntFindOperator("_test::my_op");
}

bool was_called = false;

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", [] (const Tensor&) -> void {
    was_called = true;
  });

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_return", "");
  ASSERT_TRUE(op.has_value());
  was_called = false;
  auto result = callOp(*op, dummyTensor(TensorType1()));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", [] (const Tensor&) -> std::tuple<> {
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int", [] (Tensor, int64_t a, int64_t b) -> int64_t {
        return a + b;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_output", "");
  ASSERT_TRUE(op.has_value());

  auto result = callOp(*op, dummyTensor(TensorType1()), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::returning_tensor(Tensor input) -> Tensor", [] (const Tensor& input) -> Tensor {
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", [] (const Tensor& input1, const Tensor& input2, const Tensor& input3) -> std::vector<Tensor> {
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", [](const Tensor&, int64_t input1, int64_t input2, int64_t input3) -> std::vector<int64_t> {
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", [] (Tensor) -> std::tuple<Tensor, int64_t, std::vector<Tensor>, c10::optional<int64_t>, Dict<string, Tensor>> {
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
     });

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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", [] (const Tensor& input1) -> Tensor {
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
      .op("_test::tensor_input(Tensor input) -> Tensor", [](Tensor input1) -> Tensor {
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", [] (const Tensor& input1) -> void {
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
      .op("_test::tensor_input(Tensor input) -> ()", [] (Tensor input1) -> void {
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()", [](Tensor, int64_t input1) -> void {
        captured_int_input = input1;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  captured_int_input = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_int_input);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int", [] (Tensor, int64_t input1) -> int64_t {
        return input1 + 1;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", [] (Tensor, const std::vector<int64_t>& input1) -> void {
        captured_input_list_size = input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", [](Tensor, const std::vector<int64_t>& input1) -> int64_t {
        return input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::int_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, dummyTensor(TensorType1()), IntList::create({2, 4, 6}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(3, outputs[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", [] (const std::vector<Tensor>& input1) -> void {
        captured_input_list_size = input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithTensorListRefInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", [] (const std::vector<Tensor>& input1) -> int64_t {
        return input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithLegacyTensorListRefInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", [] (const std::vector<Tensor>& input1) -> void {
        captured_input_list_size = input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithLegacyTensorListRefInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", [] (const std::vector<Tensor>& input1) -> int64_t {
        return input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithLegacyTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", [] (std::vector<Tensor> input1) -> void {
        captured_input_list_size = input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithLegacyTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", [] (std::vector<Tensor> input1) -> int64_t {
        return input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::tensor_list_input", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType1()), dummyTensor(TensorType1())}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithStringListOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::stringlist_output(str[] input) -> str[]", [](std::vector<std::string> input) {
        return input;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::stringlist_output", "");
  ASSERT_TRUE(op.has_value());

  std::vector<std::string> list{"value1", "value2"};
  auto outputs = callOp(*op, list);
  EXPECT_EQ(1, outputs.size());
  auto output = std::move(outputs[0].toGenericList()->elements());

  EXPECT_EQ(2, output.size());
  EXPECT_EQ("value1", output[0].toString()->string());
  EXPECT_EQ("value2", output[1].toString()->string());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  int captured_dict_size = 0;

  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", [&] (Dict<string, Tensor> input1) {
        captured_dict_size = input1.size();
      });

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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", [&] (Dict<string, string> input1) {
        return input1.at("key2");
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_input", "");
  ASSERT_TRUE(op.has_value());

  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toString()->string());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
    .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", [] (Dict<string, string> input) {
      return input;
    });

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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithUnorderedMapInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  int captured_dict_size = 0;

  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", [&] (std::unordered_map<string, Tensor> input1) {
        captured_dict_size = input1.size();
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_input", "");
  ASSERT_TRUE(op.has_value());

  captured_dict_size = 0;
  std::unordered_map<string, Tensor> dict;
  dict.emplace("key1", dummyTensor(TensorType1()));
  dict.emplace("key2", dummyTensor(TensorType2()));
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_dict_size);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithUnorderedMapInput_withOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", [&] (std::unordered_map<string, string> input1) {
        return input1.at("key2");
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_input", "");
  ASSERT_TRUE(op.has_value());

  std::unordered_map<string, string> dict;
  dict.emplace("key1", "value1");
  dict.emplace("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toString()->string());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithUnorderedMapOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
    .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", [] (std::unordered_map<string, string> input) {
      return input;
    });

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_output", "");
  ASSERT_TRUE(op.has_value());

  std::unordered_map<string, string> dict;
  dict.emplace("key1", "value1");
  dict.emplace("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  auto output = c10::impl::toTypedDict<string, string>(std::move(outputs[0].toGenericDict()->elements()));

  EXPECT_EQ(2, output.size());
  EXPECT_EQ("value1", output.at("key1"));
  EXPECT_EQ("value2", output.at("key2"));
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithMapOfList_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, int[]) input) -> Dict(str, int[])", [](std::unordered_map<string, std::vector<int64_t>> input) {
        return input;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_output", "");
  ASSERT_TRUE(op.has_value());

  std::unordered_map<string, std::vector<int64_t>> dict;
  dict.insert({"key1", std::vector<int64_t>{10, 20}});
  dict.insert({"key2", std::vector<int64_t>{30, 40}});
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  auto output = c10::impl::toTypedDict<string, std::vector<int64_t>>(std::move(outputs[0].toGenericDict()->elements()));

  EXPECT_EQ(2, output.size());
  EXPECT_EQ(2, output.at("key1").size());
  EXPECT_EQ(10, output.at("key1")[0]);
  EXPECT_EQ(20, output.at("key1")[1]);
  EXPECT_EQ(2, output.at("key2").size());
  EXPECT_EQ(30, output.at("key2")[0]);
  EXPECT_EQ(40, output.at("key2")[1]);
}


TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithMapOfListOfMap_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, Dict(int,str)[]) input) -> Dict(str, Dict(int,str)[])", [](std::unordered_map<string, std::vector<std::unordered_map<int64_t, string>>> input) {
        return input;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::dict_output", "");
  ASSERT_TRUE(op.has_value());

  std::unordered_map<string, std::vector<std::unordered_map<int64_t, string>>> dict;
  dict.insert({"key1", {{{10, "10"}, {20, "20"}}}});
  dict.insert({"key2", {{{30, "30"}, {40, "40"}}}});
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  auto output = c10::impl::toTypedDict<string, std::vector<std::unordered_map<int64_t, string>>>(std::move(outputs[0].toGenericDict()->elements()));

  EXPECT_EQ(2, output.size());
  EXPECT_EQ(1, output.at("key1").size());
  EXPECT_EQ(2, output.at("key1")[0].size());
  EXPECT_EQ("10", output.at("key1")[0][10]);
  EXPECT_EQ("20", output.at("key1")[0][20]);
  EXPECT_EQ(2, output.at("key2")[0].size());
  EXPECT_EQ("30", output.at("key2")[0][30]);
  EXPECT_EQ("40", output.at("key2")[0][40]);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithListOfMap_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Dict(str, int)[] input) -> Dict(str, int)[]", [](std::vector<std::unordered_map<string, int64_t>> input) {
        return input;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::list_output", "");
  ASSERT_TRUE(op.has_value());

  std::vector<std::unordered_map<string, int64_t>> list{{{"1", 1}, {"2", 2}}, {{"3", 3}, {"4", 4}}};
  auto outputs = callOp(*op, list);
  EXPECT_EQ(1, outputs.size());
  std::vector<c10::IValue> output = std::move(outputs[0].toGenericList()->elements());

  EXPECT_EQ(2, output.size());
  EXPECT_EQ(2, output[0].toGenericDictRef().size());
  EXPECT_EQ(1, output[0].toGenericDictRef().at("1").toInt());
  EXPECT_EQ(2, output[0].toGenericDictRef().at("2").toInt());
  EXPECT_EQ(2, output[1].toGenericDictRef().size());
  EXPECT_EQ(3, output[1].toGenericDictRef().at("3").toInt());
  EXPECT_EQ(4, output[1].toGenericDictRef().at("4").toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithListOfMapOfIntList_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators()
      .op("_test::list_output(Dict(str, int[])[] input) -> Dict(str, int[])[]", [](std::vector<std::unordered_map<string, std::vector<int64_t>>> input) {
        return input;
      });

  auto op = c10::Dispatcher::singleton().findSchema("_test::list_output", "");
  ASSERT_TRUE(op.has_value());

  std::vector<std::unordered_map<string, std::vector<int64_t>>> list{{{"1", {1, 2}}, {"3", {3, 4}}}, {{"5", {5, 6}}, {"7", {7, 8}}}};
  auto outputs = callOp(*op, list);
  EXPECT_EQ(1, outputs.size());
  std::vector<c10::IValue> output = std::move(outputs[0].toGenericList()->elements());

  EXPECT_EQ(2, output.size());
  EXPECT_EQ(2, output[0].toGenericDictRef().size());
  EXPECT_EQ(2, output[0].toGenericDictRef().at("1").toIntListRef().size());
  EXPECT_EQ(1, output[0].toGenericDictRef().at("1").toIntListRef()[0]);
  EXPECT_EQ(2, output[0].toGenericDictRef().at("1").toIntListRef()[1]);
  EXPECT_EQ(2, output[0].toGenericDictRef().at("3").toIntListRef().size());
  EXPECT_EQ(3, output[0].toGenericDictRef().at("3").toIntListRef()[0]);
  EXPECT_EQ(4, output[0].toGenericDictRef().at("3").toIntListRef()[1]);
  EXPECT_EQ(2, output[1].toGenericDictRef().at("5").toIntListRef().size());
  EXPECT_EQ(5, output[1].toGenericDictRef().at("5").toIntListRef()[0]);
  EXPECT_EQ(6, output[1].toGenericDictRef().at("5").toIntListRef()[1]);
  EXPECT_EQ(2, output[1].toGenericDictRef().at("7").toIntListRef().size());
  EXPECT_EQ(7, output[1].toGenericDictRef().at("7").toIntListRef()[0]);
  EXPECT_EQ(8, output[1].toGenericDictRef().at("7").toIntListRef()[1]);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  bool called = false;
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", [&] () {called = true;});

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_tensor_args", "");
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", [] (int64_t arg) {return arg + 1;});

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_tensor_args", "");
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  bool called;
  c10::optional<Tensor> called_arg2;
  c10::optional<int64_t> called_arg3;
  c10::optional<std::string> called_arg4;

  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()",
    [&] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
      called = true;
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
    });
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  bool called;
  c10::optional<Tensor> called_arg2;
  c10::optional<int64_t> called_arg3;
  c10::optional<std::string> called_arg4;

  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?",
    [&] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
      called = true;
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
      return arg2;
    });
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  bool called;
  c10::optional<Tensor> called_arg2;
  c10::optional<int64_t> called_arg3;
  c10::optional<std::string> called_arg4;

  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)",
    [] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<int64_t> arg3, c10::optional<std::string> arg4) {
      return std::make_tuple(arg2, arg3, arg4);
    });
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

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", [] (Tensor arg1, int64_t arg2, const std::vector<Tensor>& arg3) -> std::tuple<int64_t, Tensor> {return {};});

  auto op = c10::Dispatcher::singleton().findSchema("_test::no_schema_specified", "");
  ASSERT_TRUE(op.has_value());

  c10::assertSchemasHaveSameSignature(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", [] (Tensor) -> int64_t {return 0;});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", [] (Tensor) -> int64_t {return 0;});
    }, "The number of arguments is different. Specified 2 but inferred 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", [] (Tensor, Tensor) -> void {});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. Specified 0 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. Specified 1 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. Specified 3 but inferred 2"
  );
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", [] (Tensor, int64_t) -> int64_t {return 0;});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", [] (Tensor, int64_t) -> int64_t {return 0;});
    }, "Type mismatch in argument 2: specified float but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", [] (Tensor, int64_t) -> int64_t {return 0;});
    }, "Type mismatch in argument 1: specified int but inferred Tensor"
  );
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", [] (Tensor) -> int64_t {return 0;});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", [] (Tensor) -> int64_t {return 0;});
    }, "The number of returns is different. Specified 0 but inferred 1"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", [] (Tensor) -> int64_t {return 0;});
    }, "The number of returns is different. Specified 2 but inferred 1"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", [] (Tensor) -> void {});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", [] (Tensor) -> void {});
    }, "The number of returns is different. Specified 1 but inferred 0"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", [] (Tensor) -> void {});
    }, "The number of returns is different. Specified 2 but inferred 0"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. Specified 0 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. Specified 1 but inferred 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. Specified 3 but inferred 2"
  );
}

TEST(OperatorRegistrationTest_LegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", [] (Tensor) -> int64_t {return 0;});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", [] (Tensor) -> int64_t {return 0;});
    }, "Type mismatch in return 1: specified Tensor but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", [] (Tensor) -> int64_t {return 0;});
    }, "Type mismatch in return 1: specified float but inferred int"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", [] (Tensor) -> Tensor {return {};});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", [] (Tensor) -> Tensor {return {};});
    }, "Type mismatch in return 1: specified float but inferred Tensor"
  );

  // assert this does not fail because it matches
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});

  // and now a set of mismatching schemas
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});
    }, "Type mismatch in return 2: specified float but inferred int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});
    }, "Type mismatch in return 1: specified int but inferred Tensor"
  );
}

}

#pragma GCC diagnostic pop
