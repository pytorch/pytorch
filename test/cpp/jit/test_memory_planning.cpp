#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <test/cpp/jit/test_utils.h>

namespace torch {
namespace jit {

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

TEST(MemoryTracingTest, Basic) {
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


} // namespace jit
} // namespace torch
