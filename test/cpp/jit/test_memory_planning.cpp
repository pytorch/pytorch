#include <gtest/gtest.h>

#include <iostream>

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

struct StorageAttrs {
  int64_t total_size;
  DeviceType device_type;
};

struct AllocAttrs {
  int64_t size;
  int64_t offset;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  DeviceType device_type;
  at::ScalarType dtype;
};

struct PreAllocAttrs {
  int64_t size;
  int64_t offset;
  DeviceType device_type;
};

void checkAllocNodes(
    Graph& graph,
    StorageAttrs expected_storage,
    std::vector<AllocAttrs> expected_allocs,
    std::vector<std::string> expected_successors) {
  int64_t total_size;
  DeviceType device_type;

  int64_t size;
  int64_t offset;
  std::vector<int64_t> strides = {};
  std::vector<int64_t> sizes = {};
  at::ScalarType dtype;

  std::stringstream ss;
  graph.print(ss, false);
  std::unordered_set<const Node*> storage_node_uses;

  auto i = 0;
  for (const auto& node : graph.nodes()) {
    if (node->kind() == prim::AllocateStorage) {
      total_size = node->i(attr::total_size);
      device_type = static_cast<DeviceType>(node->i(attr::device));
      ASSERT_TRUE(
          total_size == expected_storage.total_size &&
          device_type == expected_storage.device_type && node->hasUses())
          << ss.str() << "\n";
      ASSERT_TRUE(node->output()->uses().size() == expected_allocs.size());
      for (auto& use : node->output()->uses()) {
        storage_node_uses.insert(use.user);
      }
      storage_node_uses.size();
    } else if (node->kind() == prim::AllocateTensor) {
      size = node->i(attr::size);
      offset = node->i(attr::offset);
      strides = node->is(attr::stride);
      sizes = node->is(attr::sizes);
      device_type = static_cast<DeviceType>(node->i(attr::device));
      dtype = static_cast<at::ScalarType>(node->i(attr::dtype));
      auto successor = node->next()->getOperator().schema().name();
      ASSERT_TRUE(
          storage_node_uses.count(node) > 0 &&
          size == expected_allocs[i].size &&
          offset == expected_allocs[i].offset &&
          sizes == expected_allocs[i].sizes &&
          strides == expected_allocs[i].strides &&
          device_type == expected_allocs[i].device_type &&
          dtype == expected_allocs[i].dtype &&
          successor == expected_successors[i] &&
          node->output()->uses().size() == 1 &&
          node->output()->uses().front().user == node->next())
          << "i: " << i << "\n"
          << size << ((size == expected_allocs[i].size) ? "==" : "!=")
          << expected_allocs[i].size << "\n"
          << offset << ((offset == expected_allocs[i].offset) ? "==" : "!=")
          << expected_allocs[i].offset << "\n"
          << sizes << ((sizes == expected_allocs[i].sizes) ? "==" : "!=")
          << expected_allocs[i].sizes << "\n"
          << strides << ((strides == expected_allocs[i].strides) ? "==" : "!=")
          << expected_allocs[i].strides << "\n"
          << device_type
          << ((device_type == expected_allocs[i].device_type) ? "==" : "!=")
          << expected_allocs[i].device_type << "\n"
          << dtype << ((dtype == expected_allocs[i].dtype) ? "==" : "!=")
          << expected_allocs[i].dtype << "\n"
          << successor << ((successor == expected_successors[i]) ? "==" : "!=")
          << expected_successors[i] << "\n"
          << ss.str() << "\n";
      i++;
    }
  }

  ASSERT_TRUE(i == (expected_allocs.size()))
      << "i: " << i << ", "
      << "expected_allocs.size() " << expected_allocs.size() << "\n"
      << ss.str() << "\n";
}

void checkPreAllocNodes(
    Graph& graph,
    StorageAttrs expected_storage,
    std::vector<PreAllocAttrs> expected_allocs,
    std::vector<std::string> expected_successors) {
  int64_t total_size;
  DeviceType device_type;

  int64_t size;
  int64_t offset;

  std::stringstream ss;
  graph.print(ss, false);

  auto i = 0;
  std::unordered_set<const Node*> storage_node_uses;
  std::unordered_set<const Value*> pre_alloc_outps;
  Node* collect_node = nullptr;

  for (const auto& node : graph.nodes()) {
    if (node->kind() == prim::AllocateStorage) {
      total_size = node->i(attr::total_size);
      device_type = static_cast<DeviceType>(node->i(attr::device));
      ASSERT_TRUE(
          total_size == expected_storage.total_size &&
          device_type == expected_storage.device_type && node->hasUses())
          << ss.str() << "\n";
      ASSERT_TRUE(node->output()->uses().size() == expected_allocs.size());
      for (auto& use : node->output()->uses()) {
        storage_node_uses.insert(use.user);
      }
      storage_node_uses.size();
    } else if (node->kind() == prim::PreAllocateTensor) {
      size = node->i(attr::size);
      offset = node->i(attr::offset);
      device_type = static_cast<DeviceType>(node->i(attr::device));
      auto successor = node->next()->getOperator().schema().name();
      ASSERT_TRUE(
          storage_node_uses.count(node) > 0 &&
          size == expected_allocs[i].size &&
          offset == expected_allocs[i].offset &&
          device_type == expected_allocs[i].device_type &&
          successor == expected_successors[i])
          << "i: " << i << "\n"
          << size << ((size == expected_allocs[i].size) ? "==" : "!=")
          << expected_allocs[i].size << "\n"
          << offset << ((offset == expected_allocs[i].offset) ? "==" : "!=")
          << expected_allocs[i].offset << "\n"
          << device_type
          << ((device_type == expected_allocs[i].device_type) ? "==" : "!=")
          << expected_allocs[i].device_type << "\n"
          << successor << ((successor == expected_successors[i]) ? "==" : "!=")
          << expected_successors[i] << "\n"
          << ss.str() << "\n";
      pre_alloc_outps.insert(node->output());
      i++;
    } else if (
        node->kind() == prim::Constant && node->hasAttribute(attr::name) &&
        node->s(attr::name) == "CollectAllocatedTensors") {
      collect_node = node;
    }
  }

  ASSERT_TRUE(i == (expected_allocs.size()))
      << "i: " << i << ", "
      << "expected_allocs.size() " << expected_allocs.size() << "\n"
      << ss.str() << "\n";

  ASSERT_TRUE(collect_node != nullptr) << ss.str() << "\n";
  ASSERT_TRUE(

      collect_node->inputs().size() == pre_alloc_outps.size())
      << collect_node->inputs().size() << ", " << pre_alloc_outps.size() << "\n"
      << ss.str() << "\n";
  for (const auto& outp : pre_alloc_outps) {
    ASSERT_TRUE(
        outp->uses().size() == 1 && outp->uses().front().user == collect_node);
  }
}

TEST(MemoryPlannerTest, SmallNaive) {
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

  StorageAttrs expected_storage = {896, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {448, 0, {10, 10}, {10, 1}, DeviceType::CPU, at::ScalarType::Float},
      {448, 448, {10, 10}, {10, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {"aten::mm", "aten::mm"};
  auto graph_copy = graph->copy();
  jit::planMemory(graph_copy, Strategy::NAIVE);
  checkAllocNodes(
      *graph_copy, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryPlannerTest, SmallLinearScan) {
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
  jit::planMemory(graph, Strategy::LINEAR_SCAN);

  StorageAttrs expected_storage = {896, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {448, 0, {10, 10}, {10, 1}, DeviceType::CPU, at::ScalarType::Float},
      {448, 448, {10, 10}, {10, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {"aten::mm", "aten::mm"};
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

std::vector<at::Tensor> buildLSTMInputTensors(
    int batch_size = 1,
    int input_size = 32) {
  int hidden_size = 2 * input_size;

  auto input = at::randn({batch_size, input_size}, at::kCPU);
  auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto w_ih = at::randn({4 * hidden_size, input_size}, at::kCPU).t();
  auto w_hh = at::randn({4 * hidden_size, hidden_size}, at::kCPU).t();

  return {input, hx, cx, w_ih, w_hh};
}

std::pair<std::shared_ptr<Graph>, Stack> buildLSTMWithStack() {
  // setup inputs
  auto vals = buildLSTMInputTensors();
  auto stack = createStack(std::move(vals));
  auto g = build_lstm();
  return std::make_pair(g, stack);
}

TEST(MemoryPlannerTest, LSTMNaive) {
  // run once to type info
  std::shared_ptr<Graph> g;
  Stack stack;
  std::tie(g, stack) = buildLSTMWithStack();
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  Code cd(graph, "lstm");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  jit::planMemory(graph, Strategy::NAIVE);

  StorageAttrs expected_storage = {4864, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 0, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 1024, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 2048, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 3072, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 3328, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 3584, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 3840, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 4096, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 4352, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 4608, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryPlannerTest, LSTMLinearScan) {
  std::shared_ptr<Graph> g;
  Stack stack;
  std::tie(g, stack) = buildLSTMWithStack();
  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  Code cd(graph, "lstm");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  jit::planMemory(graph, Strategy::LINEAR_SCAN);

  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 0, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 1024, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 2048, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 256, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 512, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 768, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1024, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 768, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryPlannerTest, LSTMGreedyBySizeWithSmallestGap) {
  std::shared_ptr<Graph> g;
  Stack stack;
  std::tie(g, stack) = buildLSTMWithStack();
  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  Code cd(graph, "lstm");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  jit::planMemory(graph, Strategy::GREEDY_BY_SIZE_WITH_SMALLEST_GAP);

  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 0, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 1024, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 2048, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 256, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 512, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 768, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1024, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 768, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryPlannerTest, LSTMGreedyBySizeWithFirstGap) {
  std::shared_ptr<Graph> g;
  Stack stack;
  std::tie(g, stack) = buildLSTMWithStack();
  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  Code cd(graph, "lstm");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  jit::planMemory(graph, Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP);

  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 0, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 1024, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 2048, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 256, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 512, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 768, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1024, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 768, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryPlannerTest, LSTMGreedyByLongestAndSizeWithSmallestGap) {
  std::shared_ptr<Graph> g;
  Stack stack;
  std::tie(g, stack) = buildLSTMWithStack();
  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  Code cd(graph, "lstm");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  jit::planMemory(
      graph, Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP);

  StorageAttrs expected_storage = {3328, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 1280, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 2304, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 256, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},

      {256, 1280, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1536, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1792, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 256, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 512, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 256, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryPlannerTest, LSTMGreedyByLongestAndSizeWithFirstGap) {
  std::shared_ptr<Graph> g;
  Stack stack;
  std::tie(g, stack) = buildLSTMWithStack();
  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  Code cd(graph, "lstm");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  jit::planMemory(graph, Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP);

  StorageAttrs expected_storage = {3328, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 1280, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 2304, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 256, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1280, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1536, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1792, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 256, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 512, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 256, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryPlannerTest, LSTMGreedyByBreadth) {
  std::shared_ptr<Graph> g;
  Stack stack;
  std::tie(g, stack) = buildLSTMWithStack();
  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  Code cd(graph, "lstm");
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  jit::planMemory(graph, Strategy::GREEDY_BY_BREADTH);

  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 0, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 1024, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {1024, 2048, {1, 256}, {256, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 1024, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 256, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 512, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 768, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 512, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
      {256, 0, {1, 64}, {64, 1}, DeviceType::CPU, at::ScalarType::Float},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryTracingAllocatorTest, MemEvents) {
  auto g = build_lstm();
  std::vector<MemEvent> mem_events;
  Code cd(g, "lstm");
  torch::jit::Inline(*g);
  InterpreterState is{cd};

  auto inputs = buildLSTMInputTensors();

  {
    c10::WithProfileTracingAllocationsGuard profile_guard(at::kCPU);
    auto stack = createStack(std::move(inputs));
    is.run(stack);
    mem_events = profile_guard.getAllocationTraces();
  }

  std::unordered_map<std::string, size_t> expected_mem_allocs = {
      {"%5 : Tensor = aten::mm(%0, %3)", 1024},
      {"%6 : Tensor = aten::mm(%1, %4)", 1024},
      {"%8 : Tensor = aten::add(%5, %6, %7)", 1024},
      {"%13 : Tensor = aten::sigmoid(%9)", 256},
      {"%14 : Tensor = aten::sigmoid(%12)", 256},
      {"%15 : Tensor = aten::tanh(%11)", 256},
      {"%16 : Tensor = aten::sigmoid(%10)", 256},
      {"%17 : Tensor = aten::mul(%16, %2)", 256},
      {"%18 : Tensor = aten::mul(%13, %15)", 256},
      {"%20 : Tensor = aten::add(%17, %18, %19)", 256},
      {"%21 : Tensor = aten::tanh(%20)", 256},
      {"%22 : Tensor = aten::mul(%14, %21)", 256},
  };

  std::unordered_map<std::string, size_t> expected_mem_frees = {
      {"%8 : Tensor = aten::add(%5, %6, %7)", 1024},
      //      {"%8 : Tensor = aten::add(%5, %6, %7)", 1024},
      {"%16 : Tensor = aten::sigmoid(%10)", 1024},
      {"%17 : Tensor = aten::mul(%16, %2)", 256},
      {"%18 : Tensor = aten::mul(%13, %15)", 256},
      //      {"%18 : Tensor = aten::mul(%13, %15)", 256},
      {"%20 : Tensor = aten::add(%17, %18, %19)", 256},
      //      {"%20 : Tensor = aten::add(%17, %18, %19)", 256},
      {"%22 : Tensor = aten::mul(%14, %21)", 256},
      //      {"%22 : Tensor = aten::mul(%14, %21)", 256},
  };

  std::unordered_map<std::string, size_t> ptr_to_alloc;

  for (const auto& item : mem_events) {
    auto node_header = item.frame_node_id->node_header;
    auto size = item.size;
    auto ptr_addr = item.ptr_addr;
    if (item.type == MemEvent::EventType::Allocate) {
      ASSERT_TRUE(
          expected_mem_allocs.count(node_header) &&
          size == expected_mem_allocs[node_header])
          << node_header << ", " << size << "\n"
          << item << "\n";
      ptr_to_alloc.insert({ptr_addr, size});
    } else {
      ASSERT_TRUE(
          expected_mem_frees.count(node_header) &&
          size == expected_mem_frees[node_header])
          << node_header << ", " << size << "\n"
          << item << "\n";
      ASSERT_TRUE(
          ptr_to_alloc.count(ptr_addr) && ptr_to_alloc[ptr_addr] == size);
    }
  }
}

TEST(MemoryTracingAllocatorTest, LSTMNaive) {
  auto g = build_lstm();
  std::vector<MemEvent> mem_events;
  Code cd(g, "lstm");
  torch::jit::Inline(*g);
  InterpreterState is{cd};

  auto inputs = buildLSTMInputTensors();

  {
    c10::WithProfileTracingAllocationsGuard profile_guard(at::kCPU);
    auto stack = createStack(std::move(inputs));
    is.run(stack);
    mem_events = profile_guard.getAllocationTraces();
  }

  std::shared_ptr<Graph> graph(
      cd.instructions_source().front()->owningGraph(), [](Graph*) {});
  jit::planMemoryWithTracing(graph, Strategy::NAIVE, mem_events, at::kCPU);

  StorageAttrs expected_storage = {4864, DeviceType::CPU};
  std::vector<PreAllocAttrs> expected_allocs = {
      {1024, 0, DeviceType::CPU},
      {1024, 1024, DeviceType::CPU},
      {1024, 2048, DeviceType::CPU},
      {256, 3072, DeviceType::CPU},
      {256, 3328, DeviceType::CPU},
      {256, 3584, DeviceType::CPU},
      {256, 3840, DeviceType::CPU},
      {256, 4096, DeviceType::CPU},
      {256, 4352, DeviceType::CPU},
      {256, 4608, DeviceType::CPU},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkPreAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryTracingAllocatorTest, LSTMLinearScan) {
  auto g = build_lstm();
  std::vector<MemEvent> mem_events;
  Code cd(g, "lstm");
  torch::jit::Inline(*g);
  InterpreterState is{cd};

  auto inputs = buildLSTMInputTensors();

  {
    c10::WithProfileTracingAllocationsGuard profile_guard(at::kCPU);
    auto stack = createStack(std::move(inputs));
    is.run(stack);
    mem_events = profile_guard.getAllocationTraces();
  }

  std::shared_ptr<Graph> graph(
      cd.instructions_source().front()->owningGraph(), [](Graph*) {});
  jit::planMemoryWithTracing(
      graph, Strategy::LINEAR_SCAN, mem_events, at::kCPU);

  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<PreAllocAttrs> expected_allocs = {
      {1024, 0, DeviceType::CPU},
      {1024, 1024, DeviceType::CPU},
      {1024, 2048, DeviceType::CPU},
      {256, 0, DeviceType::CPU},
      {256, 256, DeviceType::CPU},
      {256, 512, DeviceType::CPU},
      {256, 768, DeviceType::CPU},
      {256, 1024, DeviceType::CPU},
      {256, 768, DeviceType::CPU},
      {256, 0, DeviceType::CPU},

  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkPreAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryTracingAllocatorTest, LSTMGreedyBySizeWithSmallestGap) {
  auto g = build_lstm();
  std::vector<MemEvent> mem_events;
  Code cd(g, "lstm");
  torch::jit::Inline(*g);
  InterpreterState is{cd};

  auto inputs = buildLSTMInputTensors();

  {
    c10::WithProfileTracingAllocationsGuard profile_guard(at::kCPU);
    auto stack = createStack(std::move(inputs));
    is.run(stack);
    mem_events = profile_guard.getAllocationTraces();
  }

  std::shared_ptr<Graph> graph(
      cd.instructions_source().front()->owningGraph(), [](Graph*) {});
  jit::planMemoryWithTracing(
      graph, Strategy::GREEDY_BY_SIZE_WITH_SMALLEST_GAP, mem_events, at::kCPU);

  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<PreAllocAttrs> expected_allocs = {
      {1024, 0, DeviceType::CPU},
      {1024, 1024, DeviceType::CPU},
      {1024, 2048, DeviceType::CPU},
      {256, 0, DeviceType::CPU},
      {256, 256, DeviceType::CPU},
      {256, 512, DeviceType::CPU},
      {256, 768, DeviceType::CPU},
      {256, 1024, DeviceType::CPU},
      {256, 768, DeviceType::CPU},
      {256, 0, DeviceType::CPU},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkPreAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryTracingAllocatorTest, LSTMGreedyBySizeWithFirstGap) {
  auto g = build_lstm();
  std::vector<MemEvent> mem_events;
  Code cd(g, "lstm");
  torch::jit::Inline(*g);
  InterpreterState is{cd};

  auto inputs = buildLSTMInputTensors();

  {
    c10::WithProfileTracingAllocationsGuard profile_guard(at::kCPU);
    auto stack = createStack(std::move(inputs));
    is.run(stack);
    mem_events = profile_guard.getAllocationTraces();
  }

  std::shared_ptr<Graph> graph(
      cd.instructions_source().front()->owningGraph(), [](Graph*) {});
  jit::planMemoryWithTracing(
      graph, Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP, mem_events, at::kCPU);

  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<PreAllocAttrs> expected_allocs = {
      {1024, 0, DeviceType::CPU},
      {1024, 1024, DeviceType::CPU},
      {1024, 2048, DeviceType::CPU},
      {256, 0, DeviceType::CPU},
      {256, 256, DeviceType::CPU},
      {256, 512, DeviceType::CPU},
      {256, 768, DeviceType::CPU},
      {256, 1024, DeviceType::CPU},
      {256, 768, DeviceType::CPU},
      {256, 0, DeviceType::CPU},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkPreAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

TEST(MemoryTracingAllocatorTest, LSTMGreedyBySizeAndLongestWithFirstGap) {
  auto g = build_lstm();
  std::vector<MemEvent> mem_events;
  Code cd(g, "lstm");
  torch::jit::Inline(*g);
  InterpreterState is{cd};

  auto inputs = buildLSTMInputTensors();

  {
    c10::WithProfileTracingAllocationsGuard profile_guard(at::kCPU);
    auto stack = createStack(std::move(inputs));
    is.run(stack);
    mem_events = profile_guard.getAllocationTraces();
  }

  std::shared_ptr<Graph> graph(
      cd.instructions_source().front()->owningGraph(), [](Graph*) {});
  jit::planMemoryWithTracing(
      graph,
      Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP,
      mem_events,
      at::kCPU);

  StorageAttrs expected_storage = {3584, DeviceType::CPU};
  std::vector<PreAllocAttrs> expected_allocs = {
      {1024, 1536,  DeviceType::CPU},
      {1024, 2560,  DeviceType::CPU},
      {1024, 512,  DeviceType::CPU},
      {256, 256,  DeviceType::CPU},
      {256, 0,  DeviceType::CPU},
      {256, 1536,  DeviceType::CPU},
      {256, 1792,  DeviceType::CPU},
      {256, 512,  DeviceType::CPU},
      {256, 768,  DeviceType::CPU},
      {256, 256,  DeviceType::CPU},
  };
  std::vector<std::string> expected_successors = {
      "aten::mm",
      "aten::mm",
      "aten::add",
      "aten::sigmoid",
      "aten::sigmoid",
      "aten::tanh",
      "aten::sigmoid",
      "aten::mul",
      "aten::mul",
      "aten::tanh"};
  checkPreAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);
}

} // namespace jit
} // namespace torch
