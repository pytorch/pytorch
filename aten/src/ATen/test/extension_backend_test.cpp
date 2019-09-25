#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>

#include <ATen/core/ATenDispatch.h>

#include <torch/csrc/jit/operator.h>

using namespace at;

static int test_int;

Tensor empty_override(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> optional_memory_format) {
  test_int = 1;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(nullptr, Device(DeviceType::MSNPU, 1)), nullptr, false),
      TensorTypeId::MSNPUTensorId);
  return Tensor(std::move(tensor_impl));
}

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  test_int = 2;
  return a;
}

TEST(BackendExtensionTest, TestRegisterOp) {
  EXPECT_ANY_THROW(empty({5, 5}, at::kMSNPU));
  auto registry1 = torch::RegisterOperators()
    .op(torch::RegisterOperators::options()
      .schema("aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(empty_override), &empty_override>(TensorTypeId::MSNPUTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
  Tensor a = empty({5, 5}, at::kMSNPU);
  ASSERT_EQ(a.device().type(), at::kMSNPU);
  ASSERT_EQ(a.device().index(), 1);
  ASSERT_EQ(a.dtype(), caffe2::TypeMeta::Make<float>());
  ASSERT_EQ(test_int, 1);

  Tensor b = empty_like(a, at::kMSNPU);
  ASSERT_EQ(b.device().type(), at::kMSNPU);
  ASSERT_EQ(b.device().index(), 1);
  ASSERT_EQ(b.dtype(), caffe2::TypeMeta::Make<float>());

  EXPECT_ANY_THROW(add(a, b));
  auto registry2 = torch::RegisterOperators()
    .op(torch::RegisterOperators::options()
      .schema("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(add_override), &add_override>(TensorTypeId::MSNPUTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
  add(a, b);
  ASSERT_EQ(test_int, 2);

  // Ensure that non-MSNPU operator still works
  Tensor d = empty({5, 5}, at::kCPU);
  ASSERT_EQ(d.device().type(), at::kCPU);

  // Attempt to register on a schema that has already has a function
  EXPECT_ANY_THROW(
    torch::RegisterOperators()
      .op(torch::RegisterOperators::options()
        .schema("aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
        .impl_unboxedOnlyKernel<decltype(empty_override), &empty_override>(TensorTypeId::MSNPUTensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  );
}

// This is a "shitty" version of lazy tensor, where we run every operator by
// running it on JIT
struct ViaJitTensorImpl : public c10::TensorImpl {
  explicit ViaJitTensorImpl(at::Tensor rep)
    : TensorImpl(
        c10::TensorTypeSet(c10::TensorTypeId::TestingOnly_WrapperTensorId),
        rep.dtype(),
        rep.device()
      )
    , rep_(std::move(rep)) {}
  at::Tensor rep_;
};

void generic_override(const char* schema_str, torch::jit::Stack* stack) {
  auto schema = torch::jit::parseSchema(schema_str);
  // TODO: This is a bit circuitous
  auto s = Symbol::fromQualString(schema.name());
  auto operators = torch::jit::getAllOperatorsFor(s);
  // Find the exact match
  std::shared_ptr<torch::jit::Operator> op;
  for (const auto& candidate_op : operators) {
    auto candidate_schema = candidate_op->schema();
    // NB: this is a VERY slow equality test
    if (candidate_schema == schema) {
      op = candidate_op;
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(op);
  auto operation = op->getOperation();
  // TODO: what does the int mean
  operation(*stack);
}

TEST(BackendExtensionTest, TestBoxedFallback) {
  globalATenDispatch().registerFallbackBoxedOp(TensorTypeId::XLATensorId, &generic_override);
  Tensor a = empty({5, 5}, at::DeviceType::XLA);
}
