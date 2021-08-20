
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <third_party/kineto/libkineto/include/GenericTraceActivity.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <gtest/gtest.h>
#include <algorithm> // std::copy
#include <chrono>
#include <ctime>
#include <iostream>

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

std::vector<torch::jit::MemEvent> stop_profiling() {
  auto res = torch::autograd::profiler::disableProfiler();
  auto activities = res->getActivites();
  std::vector<torch::jit::MemEvent> mem_events{};
  for (const auto& item : *activities) {
    auto gact = dynamic_cast<libkineto::GenericTraceActivity*>(item.get());
    if (!gact->name().compare("[memory_allocate]") &&
        !gact->name().compare("[memory_free]")) {
      continue;
    }
    auto meta_map = gact->getMetadataMap();
    mem_events.emplace_back(torch::jit::MemEvent{
        meta_map.count("pc") > 0 ? std::stoi(meta_map["pc"]) : gact->startTime,
        meta_map.count("Call stack") > 0 ? meta_map["Call stack"] : "",
        meta_map.count("Addr") > 0 ? meta_map["Addr"] : "",
        meta_map.count("NodeSchema") > 0 ? meta_map["NodeSchema"] : "",
        meta_map.count("NodeHeader") > 0 ? meta_map["NodeHeader"] : "",
        meta_map.count("Bytes") > 0 ? std::stoi(meta_map["Bytes"]) : 0,
        gact->name().compare("[memory_allocate]") == 0
            ? torch::jit::MemEvent::EventType::Allocate
            : torch::jit::MemEvent::EventType::Free});
  }

  std::time_t datetime =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::stringstream fp_current_date_time;
  fp_current_date_time << "/tmp/memory_planning_trace_" << std::ctime(&datetime)
                       << ".json";
  res->save(fp_current_date_time.str());
  return mem_events;
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
  Code cd(g, "lstm");
  InterpreterState is{cd};

  start_profiling();
  is.run(stack);
  auto mem_events = stop_profiling();
  for (int strategy = static_cast<int>(Strategy::NAIVE);
       strategy <= static_cast<int>(Strategy::LINEAR_SCAN);
       strategy++) {
    std::cout << "running " << static_cast<Strategy>(strategy) << "\n";
    jit::planMemoryWithTracing(g, static_cast<Strategy>(strategy), mem_events);
    // run again to test
  }
}

} // namespace jit
} // namespace torch
