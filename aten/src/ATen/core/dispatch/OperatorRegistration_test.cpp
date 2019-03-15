#include <gtest/gtest.h>

#include <ATen/core/dispatch/OperatorRegistration.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>

using c10::OperatorKernel;
using c10::RegisterOperators;
using c10::FunctionSchema;
using c10::Argument;
using c10::IntType;
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
    /*resizeable=*/true);
  return at::detail::make_tensor<c10::TensorImpl>(storage_impl, tensorTypeId, false);
}

template<class... Args>
void callOp(const OperatorHandle& op, Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  auto kernel = c10::Dispatcher::singleton().lookup(op, &stack);
  kernel.call(&stack);
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

TEST(OperatorRegistrationTest, givenDummyKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op(dummyOpSchema, kernel<DummyKernel>(), dispatchKey(TensorType1()));
  auto op = c10::Dispatcher::singleton().findSchema("_test::dummy_op", "");
  ASSERT_TRUE(op.has_value());
  callOp(*op, dummyTensor(TensorType1()));
}

TEST(OperatorRegistrationTest, givenDummyKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
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

TEST(OperatorRegistrationTest, givenKernelFunctor_whenRegistered_thenCanBeCalled) {
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
  callOp(*op, dummyTensor(TensorType1()), 3, 4);
}

}
