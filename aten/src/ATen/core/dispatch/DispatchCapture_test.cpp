#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/DispatchCapture.h>
#include <ATen/core/dispatch/Dispatcher.h>

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
  ASSERT_FALSE(sink.enters.empty());
  EXPECT_EQ(sink.enters.front().op, "aten::add.Tensor");
  EXPECT_FALSE(sink.enters.front().is_redispatch);
  EXPECT_EQ(sink.enters.front().num_inputs, 3u);
  EXPECT_EQ(sink.exits.back().num_outputs, 1u);
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
