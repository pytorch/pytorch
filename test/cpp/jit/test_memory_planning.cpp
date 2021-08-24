
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <gtest/gtest.h>
#include <algorithm> // std::copy
#include <chrono>
#include <ctime>
#include <iostream>
#include <regex>

#include <c10/util/Optional.h>
#include <test/cpp/jit/test_utils.h>

namespace torch {
namespace jit {

void start_profiling() {
  torch::autograd::profiler::ProfilerConfig config = {
      torch::autograd::profiler::ProfilerState::KINETO, true, true, true};
  torch::autograd::profiler::ActivityType act_type =
      torch::autograd::profiler::ActivityType::CPU;

  torch::autograd::profiler::prepareProfiler(config, {act_type});
  torch::autograd::profiler::enableProfiler(config, {act_type});
}

Node* findNodeFromHeader(std::string header, std::shared_ptr<Graph>& graph) {
  for (const auto& node : graph->nodes()) {
    if (getHeader(node).compare(header) == 0) {
      return node;
    }
  }
  TORCH_INTERNAL_ASSERT(false);
}

TEST(MemoryPlannerTest, Basic) {
  // setup inputs
  constexpr int batch_size = 4;
  constexpr int input_size = 256;
  int hidden_size = 2 * input_size;
  auto input = at::randn({batch_size, input_size}, at::kCPU);
  auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto w_ih = at::randn({4 * hidden_size, input_size}, at::kCPU).t();
  auto w_hh = at::randn({4 * hidden_size, hidden_size}, at::kCPU).t();
  auto g = build_lstm();
  auto stack = createStack({input, hx, cx, w_ih, w_hh});

  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  Code cd(pr->profiled_graph_, "");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(pr->profiled_graph_->block());
  jit::RemoveProfileNodesAndSpecializeTypes(pr->profiled_graph_);

  for (int strategy = static_cast<int>(Strategy::NAIVE);
       strategy <= static_cast<int>(Strategy::LINEAR_SCAN);
       strategy++) {
    std::cout << "running " << static_cast<Strategy>(strategy) << "\n";
    jit::planMemory(pr->profiled_graph_, static_cast<Strategy>(strategy));
    // run again to test
    Code cd2(pr->profiled_graph_, "");
    InterpreterState is2{cd2};
    stack = createStack({input, hx, cx, w_ih, w_hh});
    is2.run(stack);
  }
}

TEST(MemoryTracingTest, Allocator) {
  constexpr int batch_size = 1;
  constexpr int input_size = 32;

  int hidden_size = 2 * input_size;

  auto g = build_lstm();
  std::vector<MemEvent> mem_events;
  Code cd(g, "lstm");

  {
    c10::WithProfileTracingAllocationsGuard profile_guard(at::kCPU);
    auto input = at::randn({batch_size, input_size}, at::kCPU);
    auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
    auto cx = at::randn({batch_size, hidden_size}, at::kCPU);
    auto w_ih = at::randn({4 * hidden_size, input_size}, at::kCPU).t();
    auto w_hh = at::randn({4 * hidden_size, hidden_size}, at::kCPU).t();
    torch::jit::Inline(*g);
    auto stack = createStack({input, hx, cx, w_ih, w_hh});
    InterpreterState is{cd};
    is.run(stack);
    mem_events = profile_guard.getAllocationTraces();
  }

  std::shared_ptr<Graph> graph(
      cd.instructions_source().front()->owningGraph(), [](Graph*) {});

  for (int strategy = static_cast<int>(Strategy::NAIVE);
       strategy <= static_cast<int>(Strategy::LINEAR_SCAN);
       strategy++) {
    std::cout << "running " << static_cast<Strategy>(strategy) << "\n";
    jit::planMemoryWithTracing(
        graph, static_cast<Strategy>(strategy), mem_events, at::kCPU);
  }
}

} // namespace jit
} // namespace torch
