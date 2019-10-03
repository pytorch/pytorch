#include <gtest/gtest.h>

#include <c10/core/TensorTypeId.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/ATenDispatch.h>

#include <torch/csrc/jit/operator.h>

using namespace at;

// This test file gives an example of a simple use case for "wrapper"
// and "mode" style tensor type ids.  In both cases, the implementation
// of the wrapper/mode simply passes through the call to underlying JIT
// implementation (so the wrapper/mode doesn't actually do anything),
// but this could be used as a starting point to do more interesting things.

// TODO: This to be rewritten when bwasti sets up direct access to
// JIT data structures
std::shared_ptr<torch::jit::Operator> getOperator(const char* schema_str) {
  auto schema = torch::jit::parseSchema(schema_str);
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
  return op;
}

// Global counter for ease of testing
static int64_t override_call_count = 0;

// Mode implementation

void generic_mode_fallback(const char* schema_str, torch::jit::Stack* stack) {
  override_call_count++;
  auto operation = getOperator(schema_str)->getOperation();
  c10::impl::ExcludeTensorTypeIdGuard guard(TensorTypeId::TESTING_ONLY_GenericModeTensorId);
  auto offset = operation(*stack);
  TORCH_INTERNAL_ASSERT(offset == 0);
}

// Wrapper implementation

struct GenericWrapperTensorImpl : public c10::TensorImpl {
  explicit GenericWrapperTensorImpl(at::Tensor rep)
    : TensorImpl(
        c10::TensorTypeSet(c10::TensorTypeId::TESTING_ONLY_GenericWrapperTensorId),
        rep.dtype(),
        rep.device()
        // TODO: propagate size!
      )
    , rep_(std::move(rep)) {}

  at::Tensor rep_;
};

void generic_wrapper_fallback(const char* schema_str, torch::jit::Stack* stack) {
  override_call_count++;
  auto op = getOperator(schema_str);
  auto operation = op->getOperation();

  const auto& schema = op->schema();
  auto num_arguments = schema.arguments().size();
  auto num_returns = schema.returns().size();

  // Unwrap all arguments
  auto args = torch::jit::pop(*stack, num_arguments);
  for (size_t i = 0; i < num_arguments; i++) {
    // TODO: Handle tensor list
    if (args[i].isTensor()) {
      auto* impl = args[i].unsafeToTensorImpl();
      if (impl->type_set().has(TensorTypeId::TESTING_ONLY_GenericWrapperTensorId)) {
        auto* wrapper = static_cast<GenericWrapperTensorImpl*>(impl);
        torch::jit::push(*stack, wrapper->rep_);  // no move!
      } else {
        torch::jit::push(*stack, std::move(args[i]));
      }
    } else {
      torch::jit::push(*stack, std::move(args[i]));
    }
  }

  auto offset = operation(*stack);

  // Rewrap outputs
  auto rets = torch::jit::pop(*stack, num_returns);
  for (size_t i = 0; i < num_returns; i++) {
    // TODO: Handle tensor list
    if (args[i].isTensor()) {
      torch::jit::push(*stack, at::detail::make_tensor<GenericWrapperTensorImpl>(std::move(std::move(args[i]).toTensor())) );  // yes move!
    } else {
      torch::jit::push(*stack, std::move(args[i]));
    }
  }

  TORCH_INTERNAL_ASSERT(offset == 0);
}

// As the current API does not support unregistering fallback boxed ops,
// settings of these values are PROCESS global.  Therefore the environment
// here.
class Environment : public ::testing::Environment {
 public:
  virtual ~Environment() {}

  void SetUp() override {
    globalATenDispatch().registerFallbackBoxedOp(TensorTypeId::TESTING_ONLY_GenericWrapperTensorId, &generic_wrapper_fallback);
    globalATenDispatch().registerFallbackBoxedOp(TensorTypeId::TESTING_ONLY_GenericModeTensorId, &generic_mode_fallback);
  }

  void TearDown() override {}
};

::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new Environment);

// There's a case to be made that a more comprehensive test suite would be able
// to capture many more edge cases.  This test suite is just to show that
// basic functionality works.

TEST(BoxedFallbackTest, TestBoxedFallbackWithMode) {
  c10::impl::IncludeTensorTypeIdGuard guard(TensorTypeId::TESTING_ONLY_GenericModeTensorId);

  override_call_count = 0;
  Tensor a = ones({5, 5}, kDouble);
  Tensor b = batch_norm(a, {}, {}, {}, {}, true, 0.1, 1e-05, false);
  ASSERT_EQ(override_call_count, 2);
}

TEST(BoxedFallbackTest, TestBoxedFallbackWithWrapper) {
  override_call_count = 0;
  Tensor a = at::detail::make_tensor<GenericWrapperTensorImpl>(ones({5, 5}, kDouble));
  Tensor b = batch_norm(a, {}, {}, {}, {}, true, 0.1, 1e-05, false);
  ASSERT_EQ(override_call_count, 1);
}
