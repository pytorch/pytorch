#include <gtest/gtest.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>

using c10::ArrayRef;
using c10::ivalue::TensorList;
using c10::OperatorKernel;
using c10::RegisterOperators;
using c10::FunctionSchema;
using c10::Argument;
using c10::IntType;
using c10::FloatType;
using c10::ListType;
using c10::IValue;
using c10::kernel;
using c10::dispatchKey;
using c10::CPUTensorId;
using c10::CUDATensorId;
using c10::DeviceType;
using c10::TensorTypeId;
using c10::OperatorHandle;
using at::Tensor;

namespace {

C10_DECLARE_TENSOR_TYPE(TensorType1);
C10_DEFINE_TENSOR_TYPE(TensorType1);
C10_DECLARE_TENSOR_TYPE(TensorType2);
C10_DEFINE_TENSOR_TYPE(TensorType2);

template<class... Inputs>
std::vector<IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

at::Tensor dummyTensor(TensorTypeId tensorTypeId) {
  auto* allocator = at::getCPUAllocator();
  int64_t nelements = 1;
  auto dtype = caffe2::TypeMeta::Make<float>();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizable=*/true);
  return at::detail::make_tensor<c10::TensorImpl>(storage_impl, tensorTypeId, false);
}

template<class... Args>
std::vector<IValue> callOp(const OperatorHandle& op, Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  auto kernel = c10::Dispatcher::singleton().lookup(op, &stack);
  kernel.call(&stack);
  return stack;
}

class DummyKernel final : public OperatorKernel {
public:
  void operator()(const Tensor& tensor) {}
};

FunctionSchema dummyOpSchema(
    "_test::dummy_op",
    "",
    (std::vector<Argument>{Argument("input")}),
    (std::vector<Argument>{}));

template<class Result, class... Args>
class MockKernel final : public OperatorKernel {
public:
  explicit MockKernel(std::function<Result (Args...)> implementation)
  : implementation_(std::move(implementation)) {}


  Result operator()(const Args&... args) const {
    return implementation_(args...);
  }

private:

  std::function<Result (Args...)> implementation_;
};

TEST(OperatorRegistrationTest_KernelFunctor, givenDummyKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(dummyOpSchema, kernel<DummyKernel>(), dispatchKey(TensorType1()));
  auto op = c10::Dispatcher::singleton().findSchema("_test::dummy_op", "");
  ASSERT_TRUE(op.has_value());
  callOp(*op, dummyTensor(TensorType1()));
}

TEST(OperatorRegistrationTest_KernelFunctor, givenDummyKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar1 = RegisterOperators().op(dummyOpSchema, kernel<DummyKernel>(), dispatchKey(TensorType1()));
    {
      auto registrar2 = RegisterOperators().op(dummyOpSchema, kernel<DummyKernel>(), dispatchKey(TensorType2()));

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema("_test::dummy_op", "");
      ASSERT_TRUE(op.has_value());
      callOp(*op, dummyTensor(TensorType1()));
      callOp(*op, dummyTensor(TensorType2()));
    }

    // now registrar2 is destructed. Assert that schema is still present but cpu kernel is not
    auto op = c10::Dispatcher::singleton().findSchema("_test::dummy_op", "");
    ASSERT_TRUE(op.has_value());
    callOp(*op, dummyTensor(TensorType1()));
    EXPECT_ANY_THROW(
      callOp(*op, dummyTensor(TensorType2()));
    );
  }

  // now both registrars are destructed. Assert that the whole schema is gone
  auto op = c10::Dispatcher::singleton().findSchema("_test::dummy_op", "");
  EXPECT_FALSE(op.has_value());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withoutOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input"),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{})),
    kernel<MockKernel<void, Tensor, int, int>>([] (Tensor input, int arg1, int arg2) {
      EXPECT_EQ(TensorType1(), input.type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
    }),
    dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3, 4);
  EXPECT_EQ(0, outputs.size());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withTensorInSecondPos_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("arg1", IntType::get()),
                             Argument("input"),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{})),
    kernel<MockKernel<void, int, Tensor, int>>([] (int arg1, Tensor input, int arg2) {
      EXPECT_EQ(TensorType1(), input.type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
    }),
    dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, 3, dummyTensor(TensorType1()), 4);
  EXPECT_EQ(0, outputs.size());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input"),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{})),
    kernel<MockKernel<std::tuple<>, Tensor, int, int>>([] (Tensor input, int arg1, int arg2) {
      EXPECT_EQ(TensorType1(), input.type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
      return std::tuple<>();
    }),
    dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3, 4);
  EXPECT_EQ(0, outputs.size());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input"),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{Argument("output")})),
    kernel<MockKernel<Tensor, Tensor, int, int>>([] (Tensor input, int arg1, int arg2) {
      EXPECT_EQ(TensorType1(), input.type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
      return dummyTensor(TensorType2());
    }),
    dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3, 4);
  ASSERT_EQ(1, outputs.size());
  EXPECT_EQ(TensorType2(), outputs[0].toTensor().type_id());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input"),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{Argument("output", IntType::get())})),
    kernel<MockKernel<int, Tensor, int, int>>([] (Tensor input, int arg1, int arg2) {
      EXPECT_EQ(TensorType1(), input.type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
      return 5;
    }),
    dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3, 4);
  ASSERT_EQ(1, outputs.size());
  EXPECT_EQ(5, outputs[0].toInt());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input"),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{Argument("output1"),
                             Argument("output2", IntType::get()),
                             Argument("output3", FloatType::get())})),
    kernel<MockKernel<std::tuple<Tensor, int, double>, Tensor, int, int>>([] (Tensor input, int arg1, int arg2) {
      EXPECT_EQ(TensorType1(), input.type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
      return std::tuple<Tensor, int, double>(dummyTensor(TensorType2()), 5, 2.5);
    }),
    dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, dummyTensor(TensorType1()), 3, 4);
  ASSERT_EQ(3, outputs.size());
  EXPECT_EQ(TensorType2(), outputs[0].toTensor().type_id());
  EXPECT_EQ(5, outputs[1].toInt());
  EXPECT_EQ(2.5, outputs[2].toDouble());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withListInput_withoutOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input", ListType::ofTensors()),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{})),
    kernel<MockKernel<void, ArrayRef<Tensor>, int, int>>([] (ArrayRef<Tensor> input, int arg1, int arg2) {
      EXPECT_EQ(2, input.size());
      EXPECT_EQ(TensorType2(), input[0].type_id());
      EXPECT_EQ(TensorType1(), input[1].type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
    }),
    dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType2()), dummyTensor(TensorType1())}), 3, 4);
  EXPECT_EQ(0, outputs.size());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withListInputInSecondPos_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("arg1", IntType::get()),
                             Argument("input", ListType::ofTensors()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{})),
    kernel<MockKernel<void, int, ArrayRef<Tensor>, int>>([] (int arg1, ArrayRef<Tensor> input, int arg2) {
      EXPECT_EQ(2, input.size());
      EXPECT_EQ(TensorType2(), input[0].type_id());
      EXPECT_EQ(TensorType1(), input[1].type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
    }),
    dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, 3, TensorList::create({dummyTensor(TensorType2()), dummyTensor(TensorType1())}), 4);
  EXPECT_EQ(0, outputs.size());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withListInput_withZeroOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input", ListType::ofTensors()),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{})),
    kernel<MockKernel<std::tuple<>, ArrayRef<Tensor>, int, int>>([] (ArrayRef<Tensor> input, int arg1, int arg2) {
      EXPECT_EQ(2, input.size());
      EXPECT_EQ(TensorType2(), input[0].type_id());
      EXPECT_EQ(TensorType1(), input[1].type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
      return std::tuple<>();
    }),
    dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType2()), dummyTensor(TensorType1())}), 3, 4);
  EXPECT_EQ(0, outputs.size());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withListInput_withTensorOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input", ListType::ofTensors()),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{Argument("output")})),
    kernel<MockKernel<Tensor, ArrayRef<Tensor>, int, int>>([] (ArrayRef<Tensor> input, int arg1, int arg2) {
      EXPECT_EQ(2, input.size());
      EXPECT_EQ(TensorType2(), input[0].type_id());
      EXPECT_EQ(TensorType1(), input[1].type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
      return dummyTensor(TensorType2());
    }),
    dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType2()), dummyTensor(TensorType1())}), 3, 4);
  ASSERT_EQ(1, outputs.size());
  EXPECT_EQ(TensorType2(), outputs[0].toTensor().type_id());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withListInput_withIntOutput_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input", ListType::ofTensors()),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{Argument("output", IntType::get())})),
    kernel<MockKernel<int, ArrayRef<Tensor>, int, int>>([] (ArrayRef<Tensor> input, int arg1, int arg2) {
      EXPECT_EQ(2, input.size());
      EXPECT_EQ(TensorType2(), input[0].type_id());
      EXPECT_EQ(TensorType1(), input[1].type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
      return 5;
    }),
    dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType2()), dummyTensor(TensorType1())}), 3, 4);
  ASSERT_EQ(1, outputs.size());
  EXPECT_EQ(5, outputs[0].toInt());
}

TEST(OperatorRegistrationTest_KernelFunctor, givenKernelFunctor_withListInput_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(FunctionSchema(
      "_test::mock_op",
      "",
      (std::vector<Argument>{Argument("input", ListType::ofTensors()),
                             Argument("arg1", IntType::get()),
                             Argument("arg2", IntType::get())}),
      (std::vector<Argument>{Argument("output1"),
                             Argument("output2", IntType::get()),
                             Argument("output3", FloatType::get())})),
    kernel<MockKernel<std::tuple<Tensor, int, double>, ArrayRef<Tensor>, int, int>>([] (ArrayRef<Tensor> input, int arg1, int arg2) {
      EXPECT_EQ(2, input.size());
      EXPECT_EQ(TensorType2(), input[0].type_id());
      EXPECT_EQ(TensorType1(), input[1].type_id());
      EXPECT_EQ(3, arg1);
      EXPECT_EQ(4, arg2);
      return std::tuple<Tensor, int, double>(dummyTensor(TensorType2()), 5, 2.5);
    }),
    dispatchKey(TensorType2()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::mock_op", "");
  ASSERT_TRUE(op.has_value());
  auto outputs = callOp(*op, TensorList::create({dummyTensor(TensorType2()), dummyTensor(TensorType1())}), 3, 4);
  ASSERT_EQ(3, outputs.size());
  EXPECT_EQ(TensorType2(), outputs[0].toTensor().type_id());
  EXPECT_EQ(5, outputs[1].toInt());
  EXPECT_EQ(2.5, outputs[2].toDouble());
}


}

// TODO Test functor without constructor arguments
// TODO Test different ops, each with different dispatch keys, registered in same registrar
// TODO Test different ops, each with different dispatch keys, registered in different registrar
