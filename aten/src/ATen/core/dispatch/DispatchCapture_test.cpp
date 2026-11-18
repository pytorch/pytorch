#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/DispatchCapture.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <torch/library.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Enter {
  std::string op;
  c10::DispatchKey key;
  bool is_redispatch;
  size_t num_inputs;
  int depth;
};

struct Exit {
  size_t num_outputs;
  bool threw;
};

class RecordingSink final : public c10::DispatchCaptureSink {
 public:
  void onKernelEnter(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet ks,
      bool isRedispatch,
      c10::ArrayRef<const c10::IValue> inputs) override {
    enters.push_back(Enter{
        toString(op.operator_name()),
        ks.highestPriorityTypeId(),
        isRedispatch,
        inputs.size(),
        depth++});
  }

  void onKernelExit(
      const c10::OperatorHandle& /*op*/,
      c10::ArrayRef<const c10::IValue> outputs,
      bool threw) override {
    depth--;
    exits.push_back(Exit{outputs.size(), threw});
  }

  std::vector<Enter> enters;
  std::vector<Exit> exits;
  int depth = 0;
};

// Installs a sink for the duration of a scope.
struct Installed {
  explicit Installed(c10::DispatchCaptureSink* sink)
      : prev(c10::setDispatchCaptureSink(sink)) {}
  Installed(const Installed&) = delete;
  Installed& operator=(const Installed&) = delete;
  ~Installed() {
    c10::setDispatchCaptureSink(prev);
  }
  c10::DispatchCaptureSink* prev;
};

c10::OperatorHandle addTensor() {
  return c10::Dispatcher::singleton().findSchemaOrThrow("aten::add", "Tensor");
}

// What RecordFunction observers are told while a sink is installed.
int observed_starts = 0;
size_t observed_outputs = 0;

std::unique_ptr<at::ObserverContext> onStart(const at::RecordFunction& /*fn*/) {
  observed_starts++;
  return nullptr;
}

void onEnd(const at::RecordFunction& fn, at::ObserverContext* /*ctx*/) {
  observed_outputs += fn.outputs().size();
}

at::CallbackHandle observe(bool needs_outputs) {
  observed_starts = 0;
  observed_outputs = 0;
  return at::addThreadLocalCallback(at::RecordFunctionCallback(onStart, onEnd)
                                        .needsOutputs(needs_outputs)
                                        .scopes({at::RecordScope::FUNCTION}));
}

// Takes three values and returns two.
void variadicKernel(const c10::OperatorHandle& /*op*/, torch::jit::Stack* stack) {
  torch::jit::drop(*stack, 3);
  torch::jit::push(*stack, 1, 2);
}

// Takes one value and returns two.
void variadicReturnsKernel(
    const c10::OperatorHandle& /*op*/,
    torch::jit::Stack* stack) {
  torch::jit::drop(*stack, 1);
  torch::jit::push(*stack, 1, 2);
}

TORCH_LIBRARY(_test_dispatch_capture, m) {
  m.def(
      "_test_dispatch_capture::variadic(...) -> (...)",
      torch::CppFunction::makeFromBoxedFunction<&variadicKernel>());
  m.def(
      "_test_dispatch_capture::variadic_returns(int a) -> (...)",
      torch::CppFunction::makeFromBoxedFunction<&variadicReturnsKernel>());
}

TEST(DispatchCaptureTest, CallReportsEveryStepOfTheChain) {
  const auto x = at::ones({3});
  const auto y = at::ones({3});
  RecordingSink sink;
  {
    Installed installed(&sink);
    at::add(x, y);
  }
  // The Autograd kernel first, then what it redispatches to, down to the
  // backend kernel.
  ASSERT_GE(sink.enters.size(), 2u);
  for (size_t i = 0; i < sink.enters.size(); i++) {
    const auto& e = sink.enters[i];
    EXPECT_EQ(e.op, "aten::add.Tensor");
    EXPECT_EQ(e.is_redispatch, i != 0);
    EXPECT_EQ(e.depth, static_cast<int>(i));
    EXPECT_EQ(e.num_inputs, 3u); // self, other, alpha
  }
  EXPECT_EQ(sink.enters.front().key, c10::DispatchKey::AutogradCPU);
  EXPECT_EQ(sink.enters.back().key, c10::DispatchKey::CPU);
  ASSERT_EQ(sink.exits.size(), sink.enters.size());
  for (const auto& e : sink.exits) {
    EXPECT_FALSE(e.threw);
    EXPECT_EQ(e.num_outputs, 1u);
  }
  EXPECT_EQ(sink.depth, 0);
}

TEST(DispatchCaptureTest, BoxedCallIsReported) {
  const auto x = at::ones({3});
  torch::jit::Stack stack{x, x, 1};
  RecordingSink sink;
  {
    Installed installed(&sink);
    addTensor().callBoxed(&stack);
  }
  ASSERT_EQ(sink.enters.size(), 2u);
  EXPECT_EQ(sink.enters[0].op, "aten::add.Tensor");
  EXPECT_EQ(sink.enters[0].key, c10::DispatchKey::AutogradCPU);
  EXPECT_FALSE(sink.enters[0].is_redispatch);
  EXPECT_EQ(sink.enters[0].num_inputs, 3u);
  EXPECT_EQ(sink.enters[1].key, c10::DispatchKey::CPU);
  EXPECT_TRUE(sink.enters[1].is_redispatch);
  ASSERT_EQ(sink.exits.size(), 2u);
  EXPECT_EQ(sink.exits.back().num_outputs, 1u);
  EXPECT_EQ(sink.depth, 0);
  ASSERT_EQ(stack.size(), 1u);
  EXPECT_TRUE(stack[0].toTensor().equal(at::full({3}, 2.0)));
}

TEST(DispatchCaptureTest, BoxedRedispatchIsReported) {
  const auto x = at::ones({3});
  torch::jit::Stack stack{x, x, 1};
  RecordingSink sink;
  {
    Installed installed(&sink);
    addTensor().redispatchBoxed(
        c10::DispatchKeySet(c10::DispatchKey::CPU), &stack);
  }
  ASSERT_EQ(sink.enters.size(), 1u);
  EXPECT_TRUE(sink.enters[0].is_redispatch);
  EXPECT_EQ(sink.enters[0].key, c10::DispatchKey::CPU);
  ASSERT_EQ(sink.exits.size(), 1u);
  EXPECT_EQ(sink.exits[0].num_outputs, 1u);
}

TEST(DispatchCaptureTest, BoxedCallOfAVariadicSchemaReportsTheWholeStack) {
  const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test_dispatch_capture::variadic", "");
  torch::jit::Stack stack{1, 2, 3};
  RecordingSink sink;
  {
    Installed installed(&sink);
    op.callBoxed(&stack);
  }
  ASSERT_EQ(sink.enters.size(), 1u);
  EXPECT_EQ(sink.enters[0].num_inputs, 3u);
  ASSERT_EQ(sink.exits.size(), 1u);
  EXPECT_EQ(sink.exits[0].num_outputs, 2u);
}

TEST(DispatchCaptureTest, VariadicReturnsStopAtTheArguments) {
  const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test_dispatch_capture::variadic_returns", "");
  // The first value is the caller's, as on a stack shared with an interpreter.
  torch::jit::Stack stack{7, 5};
  RecordingSink sink;
  {
    Installed installed(&sink);
    op.callBoxed(&stack);
  }
  ASSERT_EQ(sink.enters.size(), 1u);
  EXPECT_EQ(sink.enters[0].num_inputs, 1u);
  ASSERT_EQ(sink.exits.size(), 1u);
  EXPECT_EQ(sink.exits[0].num_outputs, 2u);
  EXPECT_EQ(stack.size(), 3u);
}

TEST(DispatchCaptureTest, ThrowingKernelReportsItsExit) {
  const auto x = at::ones({3});
  const auto y = at::ones({4});
  RecordingSink sink;
  {
    Installed installed(&sink);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    EXPECT_THROW(at::add(x, y), c10::Error);
  }
  ASSERT_FALSE(sink.exits.empty());
  ASSERT_EQ(sink.exits.size(), sink.enters.size());
  for (const auto& e : sink.exits) {
    EXPECT_TRUE(e.threw);
    EXPECT_EQ(e.num_outputs, 0u);
  }
  EXPECT_EQ(sink.depth, 0);
}

TEST(DispatchCaptureTest, ThrowingBoxedCallReportsItsExit) {
  torch::jit::Stack stack{at::ones({3}), at::ones({4}), 1};
  RecordingSink sink;
  {
    Installed installed(&sink);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    EXPECT_THROW(addTensor().callBoxed(&stack), c10::Error);
  }
  ASSERT_FALSE(sink.exits.empty());
  ASSERT_EQ(sink.exits.size(), sink.enters.size());
  EXPECT_TRUE(sink.exits.back().threw);
  EXPECT_EQ(sink.exits.back().num_outputs, 0u);
  EXPECT_EQ(sink.depth, 0);
}

TEST(DispatchCaptureTest, RecordFunctionObserversStillSeeTheCall) {
  const auto x = at::ones({3});
  const auto add = addTensor().typed<at::Tensor(
      const at::Tensor&, const at::Tensor&, const at::Scalar&)>();
  for (const bool needs_outputs : {false, true}) {
    const auto handle = observe(needs_outputs);
    RecordingSink sink;
    {
      Installed installed(&sink);
      add.call(x, x, 1);
    }
    at::removeCallback(handle);
    EXPECT_EQ(observed_starts, 1);
    EXPECT_EQ(observed_outputs, needs_outputs ? 1u : 0u);
    ASSERT_FALSE(sink.enters.empty());
    EXPECT_FALSE(sink.enters.front().is_redispatch);
    EXPECT_EQ(sink.exits.size(), sink.enters.size());
  }
}

TEST(DispatchCaptureTest, RecordFunctionObserversStillSeeABoxedCall) {
  const auto x = at::ones({3});
  torch::jit::Stack stack{x, x, 1};
  const auto handle = observe(/*needs_outputs=*/true);
  RecordingSink sink;
  {
    Installed installed(&sink);
    addTensor().callBoxed(&stack);
  }
  at::removeCallback(handle);
  EXPECT_EQ(observed_starts, 1);
  EXPECT_EQ(observed_outputs, 1u);
  EXPECT_EQ(sink.enters.size(), 2u);
}

TEST(DispatchCaptureTest, InstallReturnsThePreviousSink) {
  RecordingSink a;
  RecordingSink b;
  EXPECT_FALSE(c10::impl::dispatch_capture_enabled());
  EXPECT_EQ(c10::setDispatchCaptureSink(&a), nullptr);
  EXPECT_TRUE(c10::impl::dispatch_capture_enabled());
  EXPECT_EQ(c10::setDispatchCaptureSink(&b), &a);
  EXPECT_EQ(c10::setDispatchCaptureSink(nullptr), &b);
  EXPECT_FALSE(c10::impl::dispatch_capture_enabled());
  at::add(at::ones({3}), at::ones({3}));
  EXPECT_TRUE(a.enters.empty());
  EXPECT_TRUE(b.enters.empty());
}

TEST(DispatchCaptureTest, SinkOnlySeesItsOwnThread) {
  RecordingSink sink;
  Installed installed(&sink);
  std::thread other([] {
    // The counter says some thread captures; this one has no sink.
    EXPECT_TRUE(c10::impl::dispatch_capture_enabled());
    EXPECT_EQ(c10::impl::tls_dispatch_capture_sink(), nullptr);
    at::add(at::ones({3}), at::ones({3}));
  });
  other.join();
  EXPECT_TRUE(sink.enters.empty());
}

} // namespace
