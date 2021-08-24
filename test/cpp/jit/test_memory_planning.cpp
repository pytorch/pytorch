#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <test/cpp/jit/test_utils.h>

namespace torch {
namespace jit {

std::shared_ptr<Graph> build_small() {
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor):
      %4 : Tensor = aten::mm(%0, %1)
      %5 : Tensor = aten::mm(%2, %3)
      %6 : int = prim::Constant[value=1]()
      %7 : Tensor = aten::add(%4, %5, %6)
      return (%7))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  return g;
}

TEST(MemoryPlannerTest, Small) {
  // setup inputs
  auto in1 = at::randn({10, 10}, at::kCPU);
  auto in2 = at::randn({10, 10}, at::kCPU);
  auto in3 = at::randn({10, 10}, at::kCPU);
  auto in4 = at::randn({10, 10}, at::kCPU);
  auto stack = createStack({in1, in2, in3, in4});

  auto g = build_small();
  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  Code cd(graph, "small");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  jit::planMemory(graph, jit::Strategy::NAIVE);

  auto expected_graph_ir = std::string(R"IR(
    graph(%0 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu),
          %1 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu),
          %2 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu),
          %3 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu)):
        %15 : Tensor = prim::AllocateStorage[total_size=800, device=0]()
        %16 : Tensor = prim::AllocateTensor[size=400, offset=400, sizes=[10, 10], stride=[10, 1], device=0, dtype=6](%15)
        %4 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu) = aten::mm(%0, %1, %16)
        %17 : Tensor = prim::AllocateTensor[size=400, offset=0, sizes=[10, 10], stride=[10, 1], device=0, dtype=6](%15)
        %5 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu) = aten::mm(%2, %3, %17)
        %6 : int = prim::Constant[value=1]()
        %7 : Float(10, 10, strides=[10, 1], requires_grad=0, device=cpu) = aten::add(%4, %5, %6)
        return (%7))IR");
  expected_graph_ir.erase(
      remove_if(expected_graph_ir.begin(), expected_graph_ir.end(), isspace),
      expected_graph_ir.end());
  std::stringstream ss;
  graph->print(ss, false);
  auto actual_graph_ir = ss.str();
  actual_graph_ir.erase(
      remove_if(actual_graph_ir.begin(), actual_graph_ir.end(), isspace),
      actual_graph_ir.end());
  ASSERT_TRUE(actual_graph_ir.compare(expected_graph_ir) == 0);
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

} // namespace jit
} // namespace torch
