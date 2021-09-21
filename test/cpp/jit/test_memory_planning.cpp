#include <gtest/gtest.h>

#include <iostream>

#include <ATen/ATen.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/passes/memory_planning/memory_observer.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/interpreter/preprocess_graph.h>
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
      %6 : Tensor = aten::mm(%4, %5)
      %7 : int = prim::Constant[value=1]()
      %8 : Tensor = aten::add(%5, %6, %7)
      return (%8))IR";
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
  c10::TensorTypePtr ttp;
};

void checkAllocNodes(
    Graph& graph,
    StorageAttrs expected_storage,
    std::vector<AllocAttrs> expected_allocs,
    std::vector<std::string> expected_successors) {
  int64_t total_size = 0;
  DeviceType device_type;

  int64_t size = 0;
  int64_t offset = 0;
  c10::TensorTypePtr type;
  std::string successor;

  std::stringstream ss;
  graph.print(ss, false);
  std::unordered_set<const Node*> slab_node_uses;

  auto i = 0;
  for (const auto& node : graph.nodes()) {
    if (node->kind() == prim::AllocateSlab) {
      total_size = node->i(attr::total_size);
      device_type = static_cast<DeviceType>(node->i(attr::device_type));
      ASSERT_TRUE(
          total_size == expected_storage.total_size &&
          device_type == expected_storage.device_type && node->hasUses())
          << ss.str() << "\n";
      ASSERT_TRUE(node->output()->uses().size() == expected_allocs.size() + 1)
          << node->output()->uses().size() << "\n"
          << expected_allocs.size();
      for (auto& use : node->output()->uses()) {
        slab_node_uses.insert(use.user);
      }
    } else if (node->kind() == prim::AllocateTensor) {
      size = node->i(attr::size);
      offset = node->i(attr::offset);
      type = node->ty(attr::profiled_type)->expect<TensorType>();
      successor = node->next()->getOperator().schema().name();
      ASSERT_TRUE(
          slab_node_uses.count(node) > 0 && size == expected_allocs[i].size &&
          offset == expected_allocs[i].offset &&
          type->sizes() == expected_allocs[i].ttp->sizes() &&
          type->strides() == expected_allocs[i].ttp->strides() &&
          type->device() == expected_allocs[i].ttp->device() &&
          type->scalarType() == expected_allocs[i].ttp->scalarType() &&
          successor == expected_successors[i] &&
          node->output()->uses().size() == 1)
          << "i: " << i << "\n"
          << size << ((size == expected_allocs[i].size) ? "==" : "!=")
          << expected_allocs[i].size << "\n"
          << offset << ((offset == expected_allocs[i].offset) ? "==" : "!=")
          << expected_allocs[i].offset << "\n"
          << "node type: " << *type << "\n"
          << "expected type: " << *expected_allocs[i].ttp << "\n"
          << successor << ((successor == expected_successors[i]) ? "==" : "!=")
          << expected_successors[i] << "\n"
          << "uses: " << node->output()->uses().size() << "\n"
          << "user: " << node->output()->uses().front().user << "\n"
          << "next: " << node->next() << "\n"
          << ss.str() << "\n";
      i++;
    } else if (node->kind() == prim::AllocateSlab) {
      ASSERT_TRUE(
          slab_node_uses.count(node) > 0 &&
          node->inputs().size() == expected_allocs.size() + 1);
    }
  }

  ASSERT_TRUE(i == (expected_allocs.size()))
      << "i: " << i << ", "
      << "expected_allocs.size() " << expected_allocs.size() << "\n"
      << ss.str() << "\n";
}

std::tuple<
    std::map<std::string, size_t>,
    std::map<std::string, size_t>,
    std::map<intptr_t, size_t>>
findManagedAndUnmanagedAllocs(
    std::vector<jit::MemoryObserverEvent> events,
    std::string block_fn_name) {
  std::vector<MemoryEvent> mem_events;
  std::vector<FunctionFrameEvent> function_events;
  for (const auto& evt : events) {
    if (evt.type == MemoryObserverEvent::MEMORY_EVENT) {
      mem_events.push_back(evt.mem_event);
    } else if (evt.function_event.fn_name != block_fn_name) {
      function_events.push_back(evt.function_event);
    }
  }
  std::sort(
      mem_events.begin(),
      mem_events.end(),
      [](const auto& evt1, const auto& evt2) { return evt1.ts < evt2.ts; });
  std::sort(
      function_events.begin(),
      function_events.end(),
      [](const auto& evt1, const auto& evt2) {
        return evt1.start_time < evt2.start_time;
      });

  // match mem evt pointers potentially to SSA values (if they're pointer types
  // and are single output)
  std::map<std::string, size_t> managed_allocs;
  std::map<std::string, size_t> unmanaged_allocs;
  std::map<intptr_t, size_t> unexpected_allocs;
  c10::optional<intptr_t> slab_ptr;
  for (auto it = mem_events.begin(); it != mem_events.end(); it++) {
    auto mem_evt = *it;
    if (mem_evt.type == MemoryEvent::EventType::ALLOCATE) {
      if (slab_ptr == mem_evt.addr) {
        // sometimes the slab gets passed back out as an out tensor (e.g. in the
        // small graph above for the first aten::mm
        continue;
      }
      if (mem_evt.frame_node_id &&
          mem_evt.frame_node_id->node->kind() == prim::AllocateSlab) {
        managed_allocs.insert(
            {mem_evt.frame_node_id->node->output()->debugName(), mem_evt.size});
        slab_ptr = mem_evt.addr;
        continue;
      }

      // look for Value debugName associated
      auto val_name_evt = std::find_if(
          function_events.begin(),
          function_events.end(),
          [&mem_evt](auto f_evt) {
            if (f_evt.output_ival_addrs.size() != 1)
              return false;
            auto out_ival_addr = f_evt.output_ival_addrs.front();
            return mem_evt.addr == out_ival_addr;
          });

      // find if alloc is freed (or just leaked)
      // note starting at it is WLOG
      auto freed = std::find_if(it, mem_events.end(), [&mem_evt](auto me) {
        return me.addr == mem_evt.addr &&
            me.type == MemoryEvent::EventType::FREE;
      });

      if (val_name_evt != function_events.end()) {
        TORCH_CHECK(val_name_evt->output_val_names.size() == 1);
        auto val_name = val_name_evt->output_val_names.front();
        if (freed != mem_events.end()) {
          managed_allocs.emplace(val_name, mem_evt.size);
        } else {
          unmanaged_allocs.emplace(val_name, mem_evt.size);
        }
      } else {
        unexpected_allocs.emplace(mem_evt.addr, mem_evt.size);
      }
    }
  }
  return std::make_tuple(managed_allocs, unmanaged_allocs, unexpected_allocs);
}

std::map<std::string, std::string> matchPreprocessedNodesToOrigNodes(
    const Graph& preprocess_graph,
    const Graph& orig_graph) {
  std::map<std::string, std::string> mapping;
  TORCH_CHECK(orig_graph.inputs().size() == preprocess_graph.inputs().size());
  for (auto i = 0; i < orig_graph.inputs().size(); i++) {
    mapping.insert(
        {orig_graph.inputs()[i]->debugName(),
         preprocess_graph.inputs()[i]->debugName()});
  }
  std::vector<const Node*> orig_nodes(
      orig_graph.nodes().begin(), orig_graph.nodes().end());
  std::vector<const Node*> preprocessed_nodes(
      preprocess_graph.nodes().begin(), preprocess_graph.nodes().end());
  TORCH_CHECK(orig_nodes.size() == preprocessed_nodes.size());
  for (auto i = 0; i < orig_nodes.size(); i++) {
    TORCH_CHECK(orig_nodes[i]->kind() == preprocessed_nodes[i]->kind());
    if (orig_nodes[i]->outputs().size() != 1) {
      continue;
    }
    mapping.insert(
        {preprocessed_nodes[i]->output()->debugName(),
         orig_nodes[i]->output()->debugName()});
  }
  return mapping;
}

std::pair<size_t, std::map<std::string, size_t>> getPlannedUnmanagedAllocs(
    std::shared_ptr<Graph>& graph,
    Strategy strategy) {
  size_t total_size = 0;
  std::unordered_map<const Value*, std::pair<UniqueLiveRange, size_t>>
      planned_unmanaged_allocs;
  std::tie(total_size, planned_unmanaged_allocs) =
      jit::planMemory(graph, strategy);
  std::unordered_set<std::string> graph_inputs;
  for (const auto& inp : graph->inputs()) {
    graph_inputs.insert(inp->debugName());
  }

  std::map<std::string, size_t> planned_unmanaged_allocs_str;
  for (const auto& item : planned_unmanaged_allocs) {
    if (graph_inputs.count(item.first->debugName()) ||
        item.second.second == 0) {
      continue;
    }
    planned_unmanaged_allocs_str.insert(
        {item.first->debugName(), item.second.second});
  }

  return std::make_pair(total_size, planned_unmanaged_allocs_str);
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

void test(
    std::shared_ptr<Graph>& g,
    std::vector<at::Tensor> inputs,
    Strategy strategy,
    StorageAttrs expected_storage,
    std::vector<AllocAttrs> expected_allocs,
    std::vector<std::string> expected_successors) {
  // run once to type info
  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;
  std::vector<at::Tensor> baseline;
  {
    // createStack takes &&
    auto stack = createStack(std::vector<at::Tensor>(inputs));

    Code cd(graph, "small1");
    InterpreterState is{cd};
    is.run(stack);
    baseline.reserve(stack.size());
    std::transform(
        stack.begin(),
        stack.end(),
        std::back_inserter(baseline),
        [](c10::IValue v) { return std::move(v.toTensor()); });
  }

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);

  size_t total_size = 0;
  std::map<std::string, size_t> planned_unmanaged_allocs;
  std::tie(total_size, planned_unmanaged_allocs) =
      getPlannedUnmanagedAllocs(graph, strategy);
  checkAllocNodes(
      *graph, expected_storage, expected_allocs, expected_successors);

  // run again to see that everything goes well
  std::vector<at::Tensor> res;
  std::vector<jit::MemoryObserverEvent> planned_events;

  // not within scope so we don't lose the graph that cd holds
  auto stack = createStack(std::vector<at::Tensor>(inputs));
  enableMemoryObserver();
  Code cd(graph, "small2");
  InterpreterState is{cd};
  is.run(stack);
  planned_events = disableMemoryObserver();
  res.reserve(stack.size());
  std::transform(
      stack.begin(), stack.end(), std::back_inserter(res), [](c10::IValue v) {
        return std::move(v.toTensor());
      });

  // we need to do this because CodeImpl preprocesses and renumbers all SSA
  // values
  jit::interpreter::PreprocessGraph preprocessed_graph(*graph);
  auto reconciled_nodes_map =
      matchPreprocessedNodesToOrigNodes(*preprocessed_graph.graph, *graph);

  std::map<std::string, size_t> actual_managed_allocs;
  std::map<std::string, size_t> actual_unmanaged_allocs;
  std::map<intptr_t, size_t> actual_unexpected_allocs;
  std::tie(
      actual_managed_allocs,
      actual_unmanaged_allocs,
      actual_unexpected_allocs) =
      findManagedAndUnmanagedAllocs(planned_events, "small2");

  ASSERT_TRUE(
      actual_managed_allocs.size() == 1 &&
      actual_managed_allocs.begin()->second == total_size);
  for (const auto& item : actual_unmanaged_allocs) {
    ASSERT_TRUE(
        planned_unmanaged_allocs[reconciled_nodes_map[item.first]] ==
        item.second)
        << planned_unmanaged_allocs[reconciled_nodes_map[item.first]] << "\n"
        << item.second;
  }
  assertAllClose(baseline, res);
}

void testSmall(
    StorageAttrs expected_storage,
    std::vector<AllocAttrs> expected_allocs,
    Strategy strategy) {
  auto g = build_small();
  auto in1 = at::randn({10, 10}, at::kCPU);
  auto in2 = 2 * at::randn({10, 10}, at::kCPU);
  auto in3 = 3 * at::randn({10, 10}, at::kCPU);
  auto in4 = 4 * at::randn({10, 10}, at::kCPU);
  std::vector<std::string> expected_successors = {
      "aten::mm", "aten::mm", "aten::mm"};
  test(
      g,
      {in1, in2, in3, in4},
      strategy,
      expected_storage,
      expected_allocs,
      expected_successors);
}

void testLSTM(
    StorageAttrs expected_storage,
    std::vector<AllocAttrs> expected_allocs,
    Strategy strategy) {
  // setup inputs
  auto g = build_lstm();
  auto inputs = buildLSTMInputTensors();
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
  test(
      g,
      inputs,
      strategy,
      expected_storage,
      expected_allocs,
      expected_successors);
}

using Vec = std::vector<int64_t>;
#define FLOAT
#define TTP(x, y) \
  c10::TensorType::create(at::ScalarType::Float, at::kCPU, x, y, false)

TEST(MemoryPlannerTest, SmallNaive) {
  StorageAttrs expected_storage = {1344, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {448, 0, TTP(((Vec{10, 10})), ((Vec{10, 1})))},
      {448, 448, TTP(((Vec{10, 10})), ((Vec{10, 1})))},
      {448, 896, TTP(((Vec{10, 10})), ((Vec{10, 1})))},
  };
  testSmall(expected_storage, expected_allocs, Strategy::NAIVE);
}

TEST(MemoryPlannerTest, LSTMNaive) {
  StorageAttrs expected_storage = {4864, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 0, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 1024, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 2048, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {256, 3072, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 3328, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 3584, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 3840, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 4096, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 4352, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 4608, TTP((Vec{1, 64}), (Vec{64, 1}))},
  };
  testLSTM(expected_storage, expected_allocs, Strategy::NAIVE);
}

TEST(MemoryPlannerTest, SmallLinearScan) {
  StorageAttrs expected_storage = {1344, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {448, 0, TTP((Vec{10, 10}), (Vec{10, 1}))},
      {448, 448, TTP((Vec{10, 10}), (Vec{10, 1}))},
      {448, 896, TTP((Vec{10, 10}), (Vec{10, 1}))},
  };
  testSmall(expected_storage, expected_allocs, Strategy::LINEAR_SCAN);
}

TEST(MemoryPlannerTest, LSTMLinearScan) {
  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 0, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 1024, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 2048, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {256, 0, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 512, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 768, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1024, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 768, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 0, TTP((Vec{1, 64}), (Vec{64, 1}))},
  };
  testLSTM(expected_storage, expected_allocs, Strategy::LINEAR_SCAN);
}

TEST(MemoryPlannerTest, LSTMGreedyBySizeWithSmallestGap) {
  StorageAttrs expected_storage = {3328, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 1280, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 2304, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 256, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {256, 1280, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 0, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1536, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1792, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 512, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
  };
  testLSTM(
      expected_storage,
      expected_allocs,
      Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP);
}

TEST(MemoryPlannerTest, LSTMGreedyBySizeWithFirstGap) {
  StorageAttrs expected_storage = {3328, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 1280, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 2304, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 256, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {256, 1280, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 0, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1536, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1792, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 512, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
  };
  testLSTM(
      expected_storage,
      expected_allocs,
      Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP);
}

TEST(MemoryPlannerTest, LSTMGreedyByLongestAndSizeWithSmallestGap) {
  StorageAttrs expected_storage = {3328, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 1280, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 2304, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 256, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {256, 1280, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 0, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1536, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1792, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 512, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
  };
  testLSTM(
      expected_storage,
      expected_allocs,
      Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP);
}

TEST(MemoryPlannerTest, LSTMGreedyByLongestAndSizeWithFirstGap) {
  StorageAttrs expected_storage = {3328, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 1280, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 2304, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 256, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {256, 1280, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 0, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1536, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1792, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 512, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
  };
  testLSTM(
      expected_storage,
      expected_allocs,
      Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP);
}

TEST(MemoryPlannerTest, LSTMGreedyByBreadth) {
  StorageAttrs expected_storage = {3072, DeviceType::CPU};
  std::vector<AllocAttrs> expected_allocs = {
      {1024, 0, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 1024, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {1024, 2048, TTP((Vec{1, 256}), (Vec{256, 1}))},
      {256, 0, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 1024, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 256, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 512, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 768, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 512, TTP((Vec{1, 64}), (Vec{64, 1}))},
      {256, 0, TTP((Vec{1, 64}), (Vec{64, 1}))},
  };
  testLSTM(
      expected_storage,
      expected_allocs,
      Strategy::GREEDY_BY_BREADTH);
}

} // namespace jit
} // namespace torch
