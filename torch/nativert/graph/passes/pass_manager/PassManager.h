#pragma once

#include <memory>

#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/passes/pass_manager/PassPipeline.h>

namespace torch::nativert {

using torch::nativert::Graph;
using torch::nativert::GraphPass;

class PassManagerOptions {
 public:
  /* GETTERS */
  bool logGraphBetweenPasses() const {
    return log_graph_between_passes_;
  }

  /* SETTERS */
  PassManagerOptions& setLogGraphBetweenPasses(bool log_graph_between_passes) {
    log_graph_between_passes_ = log_graph_between_passes;
    return *this;
  }

 private:
  bool log_graph_between_passes_{false};
};

class GraphPassManager {
 public:
  explicit GraphPassManager(
      GraphPassPipeline pipeline,
      PassManagerOptions opts = {});
  ~GraphPassManager() = default;

  bool run(Graph* graph);

  const GraphPassPipeline& pipeline() const {
    return pipeline_;
  }

  const PassManagerOptions& opts() const {
    return opts_;
  }

 private:
  std::unique_ptr<GraphPass> create_pass(GraphPassIdentifier id);

  bool run_pass(Graph* graph, const GraphPassIdentifier& config);
  bool pass_pre_run_hook(Graph* graph, const GraphPass& pass);
  bool pass_post_run_hook(Graph* graph, const GraphPass& pass);

  const GraphPassPipeline pipeline_;
  const PassManagerOptions opts_;
};

} // namespace torch::nativert
