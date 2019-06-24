#include <thread>

#include "test/cpp/jit/test_utils.h"
#include "caffe2/torch/csrc/autograd/profiler.h"
#include "torch/torch.h"

namespace torch {
namespace jit {
namespace test {

using namespace torch::autograd::profiler;
using namespace torch::jit;
using namespace torch::autograd;
using Var = SymbolicVariable;

const std::string kDEBUGINFO = "TEST_SOURCE_DEBUG_PROFILING";
constexpr int kNumThreads = 4;
constexpr int batch_size = 4;
constexpr int input_size = 256;
constexpr int w_k_n = 256;

namespace {
Var build_fc_body(Var input, Var w) {
  auto out = input.mm(w);
  return out;
}

std::shared_ptr<Graph> build_fc() {
  auto graph_ptr = std::make_shared<Graph>();
  auto& g = *graph_ptr;
  Value* input = g.addInput();
  Value* w = g.addInput();

  Var out;
  out = build_fc_body(input, w);

  out.addAsOutput();
  g.lint();

  return graph_ptr;
}

int64_t add_source_debug_info(std::shared_ptr<Graph> graph_ptr) {
  int64_t num_nodes = 0;
  for (auto n : graph_ptr->nodes()) {
    SourceRange range(kDEBUGINFO);
    n->setSourceRange(range);
    num_nodes++;
  }
  return num_nodes;
}

int64_t num_events_annotated_with_source_debug(thread_event_lists& event_lists) {
  int64_t num_annotated{0};
  std::unique_lock<std::mutex> source_debug_info_lock(module_source_debug_info_lock);
  for (const auto& event_list : event_lists) {
    for (const auto& event : event_list) {
      auto info = event.inst_info();
      std::string source_debug_info = unsafeGetInstructionDebugInfo(event.inst_info());
      std::cout << "debug info:" << info.code_id << ":" << info.instruction_pc << ":" <<  source_debug_info << std::endl;
      if (source_debug_info == kDEBUGINFO) {
        num_annotated++;
      }
    }
  }
  return num_annotated;
}

void run_graph(const std::shared_ptr<Graph> graph_ptr, const std::vector<Variable>& inputs) {
  Code function(graph_ptr);
  InterpreterState interp(function);
  std::vector<IValue> stack(inputs.begin(), inputs.end());
  interp.run(stack);
  auto output = fmap(stack, [](const IValue& i) { return i.toTensor(); });
}

} // namespace

void testProfileSourceDebugInfoSingleThread() {
  auto input = torch::randn({batch_size, input_size}, at::kCPU);
  auto weight = torch::randn({input_size, input_size}, at::kCPU);

  enableProfiler(ProfilerConfig(ProfilerState::CPU, false));
  auto fc_graph_ptr = build_fc();
  int64_t num_nodes = add_source_debug_info(fc_graph_ptr);
  run_graph(fc_graph_ptr, {input, weight});
  auto event_list = disableProfiler();
  int64_t num_annotated = num_events_annotated_with_source_debug(event_list);
  ASSERT_TRUE(num_annotated > num_nodes);
}

void testProfileSourceDebugInfoMultiThreaded() {

  auto input = torch::randn({batch_size, input_size}, at::kCPU);
  auto weight = torch::randn({input_size, input_size}, at::kCPU);

  enableProfiler(ProfilerConfig(ProfilerState::CPU, false));
  auto fc_graph_ptr = build_fc();
  int64_t num_nodes = add_source_debug_info(fc_graph_ptr);
  std::vector<std::thread> interpreter_threads;
  for (int64_t i = 0; i < kNumThreads; ++i) {
    interpreter_threads.emplace_back(std::thread(run_graph, fc_graph_ptr, std::vector<Variable>{input, weight}));
  }
  for (int64_t i = 0; i < kNumThreads; ++i) {
    interpreter_threads[i].join();
  }
  auto event_list = disableProfiler();
  int64_t num_annotated = num_events_annotated_with_source_debug(event_list);
  ASSERT_TRUE(num_annotated > (num_nodes * kNumThreads));
}

} // namespace test
} // namespace jit
} // namespace torch
