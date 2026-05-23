#include <gtest/gtest.h>

#include <ATen/accelerator/Graph.h>

static bool is_capture_begin_called = false;
static bool is_capture_end_called = false;
static bool keep_raw_graph = false;
static bool graph_debug = false;

struct DummyGraphImpl : public at::GraphImplInterface {
  DummyGraphImpl(const at::GraphImplArgs& args = {}) {
    keep_raw_graph = args.keep_graph;
  }

  void capture_begin(
      [[maybe_unused]] at::MempoolId_t pool = {0, 0},
      [[maybe_unused]] at::GraphCaptureMode capture_mode =
          at::GraphCaptureMode::Default) override {
    is_capture_begin_called = true;
  }

  void capture_end() override {
    is_capture_end_called = true;
  }

  void instantiate() override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented");
  }

  void replay() override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented");
  }

  void reset() override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented");
  }

  at::MempoolId_t pool() const override {
    return {10, 0};
  }

  void enable_debug_mode() override {
    graph_debug = true;
  };

  void debug_dump(const std::string& path) override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented");
  }
};

namespace at {
REGISTER_GRAPH_IMPL(DUMMY, DummyGraphImpl)
}

TEST(AcceleratorGraphTest, graphRegistrationAndCapture) {
  EXPECT_EQ(at::has_graph_impl(at::kPrivateUse1), false);
  at::register_privateuse1_backend("DUMMY");
  EXPECT_EQ(at::has_graph_impl(at::kPrivateUse1), true);
  EXPECT_EQ(is_capture_begin_called, false);
  EXPECT_EQ(is_capture_end_called, false);
  EXPECT_EQ(keep_raw_graph, false);
  auto graph = at::accelerator::Graph(true);
  EXPECT_EQ(keep_raw_graph, true);
  graph.capture_begin();
  EXPECT_EQ(is_capture_begin_called, true);
  graph.capture_end();
  EXPECT_EQ(is_capture_end_called, true);
  EXPECT_EQ(graph.pool(), (at::MempoolId_t{10, 0}));
  auto graph1 = at::accelerator::Graph();
  EXPECT_EQ(keep_raw_graph, false);
  EXPECT_EQ(graph_debug, false);
  graph1.enable_debug_mode();
  EXPECT_EQ(graph_debug, true);
  ASSERT_THROW(graph1.debug_dump("abc"), c10::Error);
}
