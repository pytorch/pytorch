#include <queue>

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::AccumulateGrad;
using torch::autograd::edge_list;
using torch::autograd::Engine;
using torch::autograd::GraphRoot;
using torch::autograd::GraphTask;
using torch::autograd::Node;
using torch::autograd::validate_outputs;
using torch::autograd::variable_list;

DistEngine::DistEngine()
    : initializedContextIds_(), engine_(Engine::get_default_engine()) {}

DistEngine& DistEngine::getInstance() {
  static DistEngine engine;
  return engine;
}

void DistEngine::validateRootsAndRetrieveEdges(
    const variable_list& roots,
    edge_list& rootEdges,
    variable_list& grads) {
  TORCH_CHECK(!roots.empty(), "No tensors provided for gradient computation.");
  TORCH_INTERNAL_ASSERT(rootEdges.empty());
  TORCH_INTERNAL_ASSERT(grads.empty());

  // Verify roots are all scalar and require gradients.
  for (const auto& root : roots) {
    TORCH_CHECK(
        root.requires_grad(), "requires_grad not set on: ", root.name());
    TORCH_CHECK(
        root.numel() == 1,
        root.name(),
        " is not a scalar, all roots need to be scalar");
    TORCH_CHECK(
        root.grad_fn(),
        root.name(),
        " does not have a valid gradient function.");

    // Compute the root edges and generate the appropriate gradients.
    rootEdges.push_back(root.gradient_edge());
    grads.push_back(at::ones_like(root, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  }

  // Validate rootEdges and grads.
  validate_outputs(
      rootEdges, grads, [](const std::string& msg) { return msg; });
}

void DistEngine::computeDependencies(
    DistAutogradContext& autogradContext,
    const edge_list& rootEdges,
    const variable_list& grads,
    const std::shared_ptr<Node>& graphRoot,
    edge_list& outputEdges) {
  TORCH_INTERNAL_ASSERT(graphRoot, "graphRoot is null!");

  // Build the graph task and graph root.
  auto graphTask = std::make_shared<GraphTask>(
      /* keep_graph */ false,
      /* create_graph */ false,
      /* depth */ 0,
      /* exit_on_error */ true);

  // Run BFS to traverse the graph locally. The roots of the graph are
  // GraphRoot and all send functions for this autograd context.
  std::unordered_set<Node*> seen;
  std::queue<Node*> queue;
  queue.push(static_cast<Node*>(graphRoot.get()));

  auto sendFunctions = autogradContext.sendFunctions();

  // Add all the send functions to the queue as roots.
  for (const auto& mapEntry : sendFunctions) {
    // Increment 'outstanding_tasks_' for GraphTask for each send_function
    // since we want the local autograd engine to wait for all of them.
    graphTask->outstanding_tasks_++;
    queue.push(mapEntry.second.get());
  }

  edge_list recvBackwardEdges;
  // Traverse the graph.
  auto& dependencies = graphTask->dependencies_;
  while (!queue.empty()) {
    auto fn = queue.front();
    queue.pop();

    for (const auto& edge : fn->next_edges()) {
      if (auto nextFn = edge.function.get()) {
        dependencies[nextFn] += 1;
        const bool wasInserted = seen.insert(nextFn).second;
        if (wasInserted) {
          // Seeing this function for the first time.
          queue.push(nextFn);

          if (nextFn->next_edges().empty()) {
            TORCH_INTERNAL_ASSERT(
                dynamic_cast<AccumulateGrad*>(nextFn) ||
                dynamic_cast<RecvRpcBackward*>(nextFn));
            // We have found a leaf node which should be either AccumulateGrad
            // or RecvRpcBackward. Record the function
            // to ensure we don't execute it and instead accumulate the grads on
            // the autograd context. These functions would be passed in as the
            // 'outputs' parameter of the vanilla autograd engine.

            // We don't accumulate any grads in the context for RecvRpcBackward.
            // RecvRpcBackward is added as an output edge to indicate it is a
            // leaf node and this helps in properly computing dependencies for
            // the local autograd graph. Putting RecvRpcBackward in
            // 'outputEdges' means that this function needs to be executed
            // (inline with our assumption for FAST mode that all send/recv
            // functions are valid in the backward pass), and as a result all of
            //  its ancestors need to be executed as well.
            if (dynamic_cast<RecvRpcBackward*>(nextFn)) {
              recvBackwardEdges.emplace_back(edge);
            }
            outputEdges.emplace_back(edge);
          }
        }
      }
    }
  }

  // Now lets compute which functions need to be executed. The algorithm is as
  // follows:
  // 1. Create a dummy GraphRoot which points to all 'send' functions for this
  //    context and the original graphRoot. Run 'init_to_execute' with the
  //    outputEdges and the dummy GraphRoot. This ensures we mark
  //    appropriate functions as needed if they are reachable only from a
  //    specific 'send' function locally and not necessarily from the provided
  //    roots.
  // 2. For all edges in 'outputEdges' which point to 'RecvRpcBackward', mark
  //    those functions as needed for execution. The reason for this is that
  //    'init_to_execute', will mark these as not needed. But 'RecvRpcBackward'
  //    is unique in the sense that we use it as a leaf node in graph to compute
  //    needed execution accurately, but unlike AccumulateGrad, we do need to
  //    execute this function.
  if (!outputEdges.empty()) {
    // Compute 'needed execution' starting from all 'send' functions and the
    // original graphRoot.
    edge_list edges;
    // Create some dummy edges (input_nr not important for init_to_execute).
    for (const auto& mapEntry : sendFunctions) {
      edges.emplace_back(mapEntry.second, 0);
    }

    // Add the original graphRoot as an edge.
    edges.emplace_back(graphRoot, 0);

    // Create a dummy GraphRoot and run init_to_execute with it.
    GraphRoot dummyRoot(edges, {});
    graphTask->init_to_execute(dummyRoot, outputEdges);

    // Mark all 'RecvRPCBackward' as needing execution.
    for (const auto& recvBackwardEdge : recvBackwardEdges) {
      graphTask->exec_info_[recvBackwardEdge.function.get()].needed_ = true;
    }
  }

  // Let autograd context take ownership of the GraphTask.
  autogradContext.setGraphTask(std::move(graphTask));
}

void DistEngine::runEngineAndAccumulateGradients(
    DistAutogradContext& autogradContext,
    const std::shared_ptr<Node>& graphRoot,
    const edge_list& outputEdges) {
  // Kick off autograd computation with the root node and retrieve all the
  // gradients.
  // TODO: make this non-blocking
  // (https://github.com/pytorch/pytorch/issues/26359)
  variable_list grads = engine_.execute_with_graph_task(
      autogradContext.retrieveGraphTask(), graphRoot);

  // Accumulate all the gradients in the context.
  TORCH_INTERNAL_ASSERT(grads.size() == outputEdges.size());
  for (size_t i = 0; i < grads.size(); i++) {
    // It is possible that the grad is not defined since a separate invocation
    // of the autograd engine on the same node might actually compute this
    // gradient.
    // Also accumulate grads only for AccumulateGrad function.
    if (grads[i].defined() &&
        dynamic_cast<AccumulateGrad*>(outputEdges[i].function.get())) {
      auto& variable =
          std::static_pointer_cast<AccumulateGrad>(outputEdges[i].function)
              ->variable;
      autogradContext.accumulateGrad(variable, grads[i]);
    }
  }
}

void DistEngine::executeSendFunction(
    DistAutogradContext& autogradContext,
    const std::shared_ptr<Node>& sendFunction) {
  std::unique_lock<std::mutex> lock(initializedContextIdsLock_);
  if (initializedContextIds_.find(autogradContext.contextId()) ==
      initializedContextIds_.end()) {
    edge_list outputEdges;
    // Pass in a dummy graphRoot since all send functions are the roots.
    auto dummyRoot = std::make_shared<GraphRoot>(edge_list(), variable_list());
    computeDependencies(autogradContext, {}, {}, dummyRoot, outputEdges);

    // Mark the autograd context id as initialized and unlock.
    initializedContextIds_.insert(autogradContext.contextId());
    lock.unlock();

    // Enqueue the current send function.
    auto graphTask = autogradContext.retrieveGraphTask();
    engine_.enqueue_blocked_task_on_cpu(torch::autograd::NodeTask(
        graphTask, sendFunction, torch::autograd::InputBuffer(0)));

    // Run the autograd engine.
    runEngineAndAccumulateGradients(autogradContext, dummyRoot, outputEdges);

    // Wait for all of the outstanding rpcs to complete.
    autogradContext.clearAndWaitForOutstandingRpcs();
  } else {
    lock.unlock();
    auto graphTask = autogradContext.retrieveGraphTask();
    engine_.enqueue_blocked_task_on_cpu(torch::autograd::NodeTask(
        graphTask, sendFunction, torch::autograd::InputBuffer(0)));
  }
}

void DistEngine::execute(const variable_list& roots) {
  // Get the current context, if exists. This will throw if we don't have a
  // valid context.
  DistAutogradContext& autogradContext =
      DistAutogradContainer::getInstance().currentContext();

  // Perform initial pre-processing.
  edge_list rootEdges;
  variable_list grads;
  validateRootsAndRetrieveEdges(roots, rootEdges, grads);

  std::shared_ptr<Node> graphRoot =
      std::make_shared<GraphRoot>(rootEdges, grads);
  edge_list outputEdges;
  // Compute dependencies locally, starting from all roots and all 'send'
  // functions.
  {
    std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
    // Context should not have been intialized already.
    TORCH_INTERNAL_ASSERT(
        initializedContextIds_.find(autogradContext.contextId()) ==
        initializedContextIds_.end());

    computeDependencies(
        autogradContext, rootEdges, grads, graphRoot, outputEdges);

    // Mark the autograd context id as initialized.
    initializedContextIds_.insert(autogradContext.contextId());
  }

  runEngineAndAccumulateGradients(autogradContext, graphRoot, outputEdges);

  // Wait for all of the outstanding rpcs to complete.
  autogradContext.clearAndWaitForOutstandingRpcs();
}

} // namespace autograd
} // namespace distributed
} // namespace torch
