#include <torch/nativert/graph/passes/pass_manager/PassManager.h>

#include <c10/util/CallOnce.h>

#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/passes/pass_manager/GraphPasses.h>

namespace torch::nativert {

GraphPassManager::GraphPassManager(
    GraphPassPipeline pipeline,
    PassManagerOptions opts)
    : pipeline_(std::move(pipeline)), opts_(opts) {
  static c10::once_flag flag;
  c10::call_once(flag, [&]() { register_base_passes(); });
}

bool GraphPassManager::run(Graph* graph) {
  bool changed = false;
  for (const auto& pass_name : pipeline_) {
    changed |= run_pass(graph, pass_name);
  }
  return changed;
}

bool GraphPassManager::run_pass(Graph* graph, const GraphPassIdentifier& name) {
  const auto& pass = GraphPassRegistry::get().get_pass(name);

  size_t nodesBefore = 0;
  if (VLOG_IS_ON(1)) {
    nodesBefore = graph->nodes().size();
  }

  bool changed = pass_pre_run_hook(graph, pass);
  changed |= (pass.get())(graph);
  changed |= pass_post_run_hook(graph, pass);

  if (VLOG_IS_ON(1)) {
    size_t nodesAfter = graph->nodes().size();
    if (changed) {
      VLOG(1) << "Pass " << name << ": " << nodesBefore << " -> " << nodesAfter
              << " nodes (delta: "
              << static_cast<int64_t>(nodesAfter) -
              static_cast<int64_t>(nodesBefore)
              << ")";
    } else {
      VLOG(1) << "Pass " << name << ": no change (" << nodesAfter << " nodes)";
    }
  }

  return changed;
}

bool GraphPassManager::pass_pre_run_hook(Graph* graph, const GraphPass& pass) {
  if (opts_.logGraphBetweenPasses()) {
    LOG(INFO) << "Before pass: " << pass.name() << '\n'
              << graph->toString() << "-------------------------";
  }
  return false;
}

bool GraphPassManager::pass_post_run_hook(Graph* graph, const GraphPass& pass) {
  if (opts_.logGraphBetweenPasses()) {
    LOG(INFO) << "After pass: " << pass.name() << '\n'
              << graph->toString() << "-------------------------";
  }
  return false;
}

} // namespace torch::nativert
