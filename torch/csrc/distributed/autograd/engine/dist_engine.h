#pragma once

#include <mutex>
#include <unordered_set>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/distributed/autograd/context/context.h>

namespace torch {
namespace distributed {
namespace autograd {

// Forward declaration.
class BackwardPassCleanupGuard;

// This is a singleton class responsible for running distributed backward
// passes. This engine relies heavily on the vanilla autograd engine and tries
// to re-use it as much as possible. This class is mostly responsible for the
// distributed aspects of autograd and tries to hook into the autograd engine
// where convenient.

// Unlike the vanilla autograd engine, the distributed autograd engine
// accumulates the gradients in the appropriate DistAutogradContext. This avoids
// multiple trainer nodes stomping on each others gradients.
class TORCH_API DistEngine {
 public:
  // Retrieve the singleton instance.
  static DistEngine& getInstance();

  // Given a list of root variables, start the distributed backwards pass from
  // these variables and accumulate all the gradients in the current autograd
  // context on each node. This method is used to kickoff distributed autograd
  // on a single node.
  void execute(
      int64_t context_id,
      const torch::autograd::variable_list& roots,
      bool retainGraph);

  // Given a send function to execute in the autograd engine, ensures we compute
  // dependencies once for this node and enqueues the send function for execute
  // in the engine.
  // This method is used to kick off the autograd computation on a node when it
  // receives gradients from the corresponding 'recv' method on another node.
  // The gradients are accumulated in the provided autograd context.
  std::shared_ptr<rpc::FutureMessage> executeSendFunctionAsync(
      const ContextPtr& autogradContext,
      const std::shared_ptr<torch::autograd::Node>& sendFunction,
      bool retainGraph);

  // Number of backward passes currently running for the Distributed Engine.
  size_t numBackwardPasses() const;

  // Returns key-value pairs consisting of useful debugging information related
  // to distributed autograd.
  std::unordered_map<std::string, std::string> getDebugInfo() const;

 private:
  // Make sure this is a singleton.
  DistEngine();
  ~DistEngine() = default;

  DistEngine(const DistEngine&) = delete;
  DistEngine& operator=(const DistEngine&) = delete;
  DistEngine(DistEngine&&) = delete;
  DistEngine& operator=(DistEngine&&) = delete;

  // Validates the input roots for the backward computations and retrieves the
  // appropriate root edges and corresponding gradients. Populates root_edges
  // with the appropriate gradient edges and grads with the gradients for each
  // edge.
  void validateRootsAndRetrieveEdges(
      const torch::autograd::variable_list& roots,
      torch::autograd::edge_list& rootEdges,
      torch::autograd::variable_list& grads);

  // Given the autograd context, root edges and grads, we compute dependencies
  // for the local node and fill out the provided GraphTask and GraphRoot with
  // appropriate information for the local autograd engine.
  // We also determine all leaf nodes(functions) in the graph and accumulate
  // them in outputEdges.
  void computeDependencies(
      const ContextPtr& context,
      const torch::autograd::edge_list& rootEdges,
      const torch::autograd::variable_list& grads,
      const std::shared_ptr<torch::autograd::Node>& graphRoot,
      torch::autograd::edge_list& outputEdges,
      bool retainGraph);

  // Run the local autograd engine using the provided graphTask and graphRoot
  // and accumulate the gradients part 'outputEdges' in the provided autograd
  // context.
  std::shared_ptr<rpc::FutureMessage> runEngineAndAccumulateGradients(
      const ContextPtr& autogradContext,
      const std::shared_ptr<torch::autograd::Node>& graphRoot,
      const torch::autograd::edge_list& outputEdges);

  // Run after the backward pass is done to appropriately cleanup structures.
  void cleanupBackwardPass(const ContextPtr& autogradContext);

  // Set of autograd context_ids, which we have already initialized for
  // distributed autograd on this node (e.g.: already computed dependencies)
  std::unordered_set<int64_t> initializedContextIds_;

  mutable std::mutex initializedContextIdsLock_;

  // Reference to local autograd engine.
  torch::autograd::Engine& engine_;

  friend class BackwardPassCleanupGuard;
};

// Guard to clean up resources once the backward pass is done.
class BackwardPassCleanupGuard {
 public:
  explicit BackwardPassCleanupGuard(const ContextPtr& autogradContext)
      : autogradContext_(autogradContext) {}

  ~BackwardPassCleanupGuard() {
    DistEngine::getInstance().cleanupBackwardPass(autogradContext_);
  }

 private:
  ContextPtr autogradContext_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
