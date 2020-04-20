#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>

using namespace at;

namespace {

// This test file gives an example of a simple use case for "wrapper"
// and "mode" style tensor type ids.  In both cases, the implementation
// of the wrapper/mode simply passes through the call to underlying JIT
// implementation (so the wrapper/mode doesn't actually do anything),
// but this could be used as a starting point to do more interesting things.

// Global counter for ease of testing
static int64_t override_call_count = 0;

// TODO Remove callBoxedWorkaround once op.callBoxed(stack) works for all ops.
//      Once callBoxedWorkaround is removed, we can move this file to the location
//      where it actually belongs, i.e. next to Dispatcher.h. The only reason for
//      this not being there yet is that callBoxedWorkaround depends on JIT.

void callBoxedWorkaround(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // This should just be op.callBoxed(stack), but that doesn't work for all ops yet.
  // Note: If op.callBoxed(stack) works for you, then that is preferrable because
  // it's much faster and doesn't come with a dependency on JIT code.
  // Instead, we take a path through the JIT operator registry, which has a boxed
  // calling mechanism that works for all ops from native_functions.yaml.

  auto s = Symbol::fromQualString(op.schema().name());
  auto operators = torch::jit::getAllOperatorsFor(s);
  // Find the exact match
  std::shared_ptr<torch::jit::Operator> jit_op;
  for (const auto& candidate_op : operators) {
    auto candidate_schema = candidate_op->schema();
    // NB: this is a VERY slow equality test
    if (candidate_schema == op.schema()) {
      jit_op = candidate_op;
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(jit_op);

  auto offset = jit_op->getOperation()(*stack);
  TORCH_INTERNAL_ASSERT(offset == 0);
}

// Mode implementation

void generic_mode_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  override_call_count++;
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericModeTensorId);
  callBoxedWorkaround(op, stack);
}

// Wrapper implementation

struct GenericWrapperTensorImpl : public c10::TensorImpl {
  explicit GenericWrapperTensorImpl(at::Tensor rep)
    : TensorImpl(
        c10::DispatchKeySet(c10::DispatchKey::TESTING_ONLY_GenericWrapperTensorId),
        rep.dtype(),
        rep.device()
        // TODO: propagate size!
      )
    , rep_(std::move(rep)) {}

  at::Tensor rep_;
};

void generic_wrapper_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  override_call_count++;

  auto num_arguments = op.schema().arguments().size();
  auto num_returns = op.schema().returns().size();

  // Unwrap all arguments
  auto args = torch::jit::pop(*stack, num_arguments);
  for (size_t i = 0; i < num_arguments; i++) {
    // TODO: Handle tensor list
    if (args[i].isTensor()) {
      auto* impl = args[i].unsafeToTensorImpl();
      if (impl->key_set().has(DispatchKey::TESTING_ONLY_GenericWrapperTensorId)) {
        auto* wrapper = static_cast<GenericWrapperTensorImpl*>(impl);
        torch::jit::push(*stack, wrapper->rep_);  // no move!
      } else {
        torch::jit::push(*stack, std::move(args[i]));
      }
    } else {
      torch::jit::push(*stack, std::move(args[i]));
    }
  }

  callBoxedWorkaround(op, stack);

  // Rewrap outputs
  auto rets = torch::jit::pop(*stack, num_returns);
  for (size_t i = 0; i < num_returns; i++) {
    // TODO: Handle tensor list
    if (args[i].isTensor()) {
      torch::jit::push(*stack, at::detail::make_tensor<GenericWrapperTensorImpl>(std::move(args[i]).toTensor()));  // yes move!
    } else {
      torch::jit::push(*stack, std::move(args[i]));
    }
  }
}

TEST(BackendFallbackTest, TestBackendFallbackWithMode) {
  auto registry = c10::Dispatcher::singleton()
    .registerFallback(
      DispatchKey::TESTING_ONLY_GenericModeTensorId,
      KernelFunction::makeFromBoxedFunction<&generic_mode_fallback>()
    );

  c10::impl::IncludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericModeTensorId);

  override_call_count = 0;
  Tensor a = ones({5, 5}, kDouble);
  Tensor b = batch_norm(a, {}, {}, {}, {}, true, 0.1, 1e-05, false);
  ASSERT_EQ(override_call_count, 2);
}

TEST(BackendFallbackTest, TestBackendFallbackWithWrapper) {
  auto registry = c10::Dispatcher::singleton().registerFallback(
      DispatchKey::TESTING_ONLY_GenericWrapperTensorId,
      KernelFunction::makeFromBoxedFunction<&generic_wrapper_fallback>()
  );

  override_call_count = 0;
  Tensor a = at::detail::make_tensor<GenericWrapperTensorImpl>(ones({5, 5}, kDouble));
  Tensor b = batch_norm(a, {}, {}, {}, {}, true, 0.1, 1e-05, false);
  ASSERT_EQ(override_call_count, 1);
}

TEST(BackendFallbackTest, TestFallthroughBackendFallback) {
  // By default fallthrough
  auto registry = c10::import()
    .fallback(
          c10::dispatch(DispatchKey::TESTING_ONLY_GenericModeTensorId,
                        c10::CppFunction::makeFallthrough()))
    .impl("aten::mul.Tensor",
          c10::dispatch(DispatchKey::TESTING_ONLY_GenericModeTensorId,
                        c10::CppFunction::makeFromBoxedFunction<&generic_mode_fallback>()));

  c10::impl::IncludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericModeTensorId);

  override_call_count = 0;
  // Doesn't trigger, as we fallthrough
  Tensor a = zeros({5, 5}, kDouble);
  ASSERT_EQ(override_call_count, 0);
  // Does trigger, because we explicitly set it
  Tensor b = mul(a, a);
  ASSERT_EQ(override_call_count, 1);
}

}
