#include <queue>

#include <ATen/Parallel.h>
#include <c10/core/Event.h>
#include <c10/util/DeadlockDetection.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::AccumulateGrad;
using torch::autograd::edge_list;
using torch::autograd::Engine;
using torch::autograd::GraphRoot;
using torch::autograd::GraphTask;
using torch::autograd::GraphTaskGuard;
using torch::autograd::InputBuffer;
using torch::autograd::Node;
using torch::autograd::NodeTask;
using torch::autograd::ReadyQueue;
using torch::autograd::validate_outputs;
using torch::autograd::variable_list;

static constexpr const char* kNumBackwardPasses = "num_current_backward_passes";
static constexpr const char* kNumAutogradContexts = "num_autograd_contexts";

// This hook does 3 things:
//   1. Call pre hooks of the original AccumulateGrad to modify the input grad.
//   2. Accumurate the guard to RPC context.
//   3. Call post hooks of the original AccumulateGrad.
class DistAccumulateGradCaptureHook
    : public GraphTask::ExecInfo::Capture::GradCaptureHook {
 public:
  DistAccumulateGradCaptureHook(
      std::shared_ptr<AccumulateGrad> accumulateGrad,
      ContextPtr autogradContext)
      : accumulateGrad_(std::move(accumulateGrad)),
        autogradContext_(std::move(autogradContext)) {}

  at::Tensor operator()(const at::Tensor& grad) override {
    ThreadLocalDistAutogradContext contextGuard{ContextPtr(autogradContext_)};
    variable_list inputGrads = {grad};
    // It's intended that pre/post hooks are still called even if the grad is
    // undefined here.
    for (const auto& hook : accumulateGrad_->pre_hooks()) {
      inputGrads = (*hook)(inputGrads);
    }
    // It is possible that the grad is not defined since a separate
    // invocation of the autograd engine on the same node might actually
    // compute this gradient.
    if (inputGrads[0].defined()) {
      // There are 3 internal references to 'inputGrads[0]' at this moment:
      //   1. 'inputGrads[0]' in this function.
      //   2. 'graph_task->captured_vars_' on the callsite in the local engine.
      //   3. 'InputBuffer& inputs' on the callsite as the inputs of the
      //   function node.
      autogradContext_->accumulateGrad(
          accumulateGrad_->variable, inputGrads[0], 3 /* num_expected_refs */);
    }
    const variable_list kEmptyOutput;
    for (const auto& hook : accumulateGrad_->post_hooks()) {
      (*hook)(kEmptyOutput, inputGrads);
    }
    return inputGrads[0];
  }

 private:
  std::shared_ptr<AccumulateGrad> accumulateGrad_;
  ContextPtr autogradContext_;
};

void DistEngine::globalCpuThread(
    const std::shared_ptr<ReadyQueue>& ready_queue) {
  while (true) {
    NodeTask task = ready_queue->pop();
    if (task.isShutdownTask_) {
      // Need to shutdown this thread.
      C10_LOG_API_USAGE_ONCE("torch.autograd.thread_shutdown");
      break;
    }

    auto graphTask = task.base_.lock();
    if (graphTask == nullptr) {
      // GraphTask has expired, ignore and continue processing.
      continue;
    }

    // Launch the execution on a JIT thread.
    at::launch([this,
                graphTask,
                graphRoot = task.fn_,
                variables =
                    InputBuffer::variables(std::move(task.inputs_))]() mutable {
      InputBuffer inputs(variables.size());
      for (const auto i : c10::irange(variables.size())) {
        inputs.add(i, std::move(variables[i]), c10::nullopt, c10::nullopt);
      }
      execute_graph_task_until_ready_queue_empty(
          /*node_task*/ NodeTask(graphTask, graphRoot, std::move(inputs)),
          /*incrementOutstandingTasks*/ false);
    });
  }
}

DistEngine::DistEngine()
    : initializedContextIds_(),
      engine_(Engine::get_default_engine()),
      global_cpu_ready_queue_(std::make_shared<ReadyQueue>()),
      global_cpu_thread_(
          &DistEngine::globalCpuThread,
          this,
          global_cpu_ready_queue_) {
  // Note [GPU to CPU continuations]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Initialize a single CPU thread to execute continuations from GPU
  // tasks. The multithreaded structure for the distributed engine works
  // well only for CPU tasks. If we have an order of tasks like
  // CPU->GPU->CPU, distributed autograd has no thread to execute the last
  // CPU task on. To fix this, we introduce a global CPU thread to handle
  // such situations and it will be responsible for executing these CPU
  // tasks. The CPU thread has its own ready_queue which is used as the
  // cpu_ready_queue for all GraphTasks for DistEngine. This ensures all GPU
  // to CPU continuations are enqueued on this thread. The global CPU thread
  // simply dequeues tasks from the global queue and calls
  // "execute_graph_task_until_ready_queue_empty" on a JIT thread to execute the
  // appropriate task.
  global_cpu_thread_.detach();
}

DistEngine::~DistEngine() {
  // Ensure we shutdown the CPU thread.
  TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP();
  global_cpu_ready_queue_->pushShutdownTask();
  global_cpu_thread_.join();
}

DistEngine& DistEngine::getInstance() {
  // Leaky singleton to avoid module destructor race.
  static DistEngine* engine = new DistEngine();
  return *engine;
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
    TORCH_CHECK(root.requires_grad(), "requires_grad not set on root");
    TORCH_CHECK(
        root.numel() == 1,
        root.name(),
        " is not a scalar, all roots need to be scalar");
    TORCH_CHECK(
        root.grad_fn(),
        root.name(),
        " does not have a valid gradient function.");

    // Compute the root edges and generate the appropriate gradients.
    rootEdges.push_back(torch::autograd::impl::gradient_edge(root));
    grads.push_back(at::ones_like(root, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  }

  // Validate rootEdges and grads.
  validate_outputs(
      rootEdges, grads, [](const std::string& msg) { return msg; });
}

void DistEngine::computeDependencies(
    const ContextPtr& autogradContext,
    const edge_list& rootEdges,
    const variable_list& grads,
    const std::shared_ptr<Node>& graphRoot,
    edge_list& outputEdges,
    bool retainGraph) {
  TORCH_INTERNAL_ASSERT(graphRoot, "graphRoot is null!");

  // Store root nodes so we can traverse through the graph later
  // e.g., for get_current_graph_task_execution_order
  c10::SmallVector<Node*, 4> temp_roots{rootEdges.size()};
  for (const auto i : c10::irange(rootEdges.size())) {
    temp_roots[i] = rootEdges[i].function.get();
  }

  // Build the graph task and graph root.
  // NOTE: we don't need to build and pass a cpu_ready_queue to GraphTask
  // as we use execute_graph_task_until_ready_queue_empty, which will build
  // a separate ReadyQueue for each call.
  auto graphTask = std::make_shared<GraphTask>(
      /* keep_graph */ retainGraph,
      /* create_graph */ false,
      /* depth */ 0,
      /* cpu_ready_queue */ global_cpu_ready_queue_,
      /* graph_roots */ temp_roots,
      /* exit_on_error */ true);

  // Run BFS to traverse the graph locally. The roots of the graph are
  // GraphRoot and all send functions for this autograd context.
  std::unordered_set<Node*> seen;
  std::queue<Node*> queue;
  queue.push(static_cast<Node*>(graphRoot.get()));

  auto sendFunctions = autogradContext->sendFunctions();

  // Add all the send functions to the queue as roots.
  for (const auto& mapEntry : sendFunctions) {
    // Increment 'outstanding_tasks_' for GraphTask for each send_function
    // since we want the local autograd engine to wait for all of them.
    graphTask->outstanding_tasks_++;
    queue.push(mapEntry.second.get());
  }

  bool will_use_accelerator = false;

  edge_list recvBackwardEdges;
  // Traverse the graph.
  auto& dependencies = graphTask->dependencies_;
  while (!queue.empty()) {
    auto fn = queue.front();
    queue.pop();

    if (!will_use_accelerator) {
      will_use_accelerator = fn->stream().has_value();
    }

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

  if (will_use_accelerator) {
    // Collects current streams for CUDA/ROCM devices where this process has a
    // context, so graphTask::exec_post_processing can sync them with
    // leaf_streams.
    graphTask->stash_current_streams();
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
    graphTask->init_to_execute(
        dummyRoot, outputEdges, /*accumulate_grad=*/false, /*min_topo_nr=*/0);
    for (auto& mapEntry : graphTask->exec_info_) {
      auto& execInfo = mapEntry.second;
      if (!execInfo.captures_) {
        continue;
      }
      auto fn = mapEntry.first;
      // There may be nodes other than 'AccumulateGrad', e.g. RecvRPCBackward,
      // to be captured.
      if (auto accumulateGradFn = dynamic_cast<AccumulateGrad*>(fn)) {
        for (auto& capture : *execInfo.captures_) {
          // Capture hooks are technically deprecated, but as an exception below
          // is the single and only instance of capture hooks usage that we
          // support. See NOTE [Deprecated capture hooks] for more context.
          capture.DO_NOT_USE_DEPRECATED_register_capture_hook(
              std::make_unique<DistAccumulateGradCaptureHook>(
                  std::dynamic_pointer_cast<AccumulateGrad>(
                      accumulateGradFn->shared_from_this()),
                  autogradContext));
        }
      }
    }

    // Mark all 'RecvRPCBackward' as needing execution.
    for (const auto& recvBackwardEdge : recvBackwardEdges) {
      graphTask->exec_info_[recvBackwardEdge.function.get()].needed_ = true;
    }
  }

  // Set graph task owner in a single thread since concurrent access to
  // 'owner_' field is not permitted.
  graphTask->owner_ = torch::autograd::CPU_DEVICE;

  // Let autograd context take ownership of the GraphTask.
  autogradContext->setGraphTask(std::move(graphTask));
}

void DistEngine::execute_graph_task_until_ready_queue_empty(
    NodeTask&& node_task,
    bool incrementOutstandingTasks) {
  engine_.initialize_device_threads_pool();
  // Create a ready queue per call to traverse the graph_task from
  // root_to_execute This allow concurrent execution of the same GraphTask from
  // different threads
  std::shared_ptr<ReadyQueue> cpu_ready_queue = std::make_shared<ReadyQueue>();
  auto graph_task = node_task.base_.lock();
  if (graph_task == nullptr) {
    LOG(ERROR) << "GraphTask has expired for NodeTask: "
               << node_task.fn_->name() << ", skipping execution.";
    return;
  }

  cpu_ready_queue->push(std::move(node_task), incrementOutstandingTasks);

  torch::autograd::set_device(torch::autograd::CPU_DEVICE);
  while (!cpu_ready_queue->empty()) {
    std::shared_ptr<GraphTask> local_graph_task;
    {
      // Scope this block of execution since NodeTask is not needed after this
      // block and can be deallocated (release any references to grad tensors
      // as part of inputs_)
      NodeTask task = cpu_ready_queue->pop();
      if (!(local_graph_task = task.base_.lock())) {
        continue;
      }
      if (task.fn_ && !local_graph_task->has_error_.load()) {
        at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);
        try {
          GraphTaskGuard guard(local_graph_task);
          engine_.evaluate_function(
              local_graph_task, task.fn_.get(), task.inputs_, cpu_ready_queue);
        } catch (std::exception& e) {
          engine_.thread_on_exception(local_graph_task, task.fn_, e);
          // break the loop in error so that we immediately stop the execution
          // of this GraphTask, mark it completed if necessary and return the
          // future with proper ErrorMessage
          break;
        }
      }
    }
    // Decrement the outstanding task.
    --local_graph_task->outstanding_tasks_;
  }
  // Check if we've completed execution.
  if (graph_task->completed()) {
    // We don't need to explicitly notify the owner thread, since
    // 'mark_as_completed_and_run_post_processing' would mark the Future as
    // completed and this would notify the owner thread that the task has been
    // completed.
    graph_task->mark_as_completed_and_run_post_processing();
  }
}

c10::intrusive_ptr<c10::ivalue::Future> DistEngine::
    runEngineAndAccumulateGradients(
        const ContextPtr& autogradContext,
        const std::shared_ptr<Node>& graphRoot,
        const edge_list& outputEdges,
        bool incrementOutstandingTasks) {
  // Cleanup previous state for outstanding RPCs. Outstanding RPCs could be
  // lingering if we're running backward multiple times and some of the
  // passes ran into errors.
  autogradContext->clearOutstandingRpcs();
  auto graphTask = autogradContext->retrieveGraphTask();
  at::launch([this, graphTask, graphRoot, incrementOutstandingTasks]() {
    execute_graph_task_until_ready_queue_empty(
        /*node_task*/ NodeTask(graphTask, graphRoot, InputBuffer(0)),
        /*incrementOutstandingTasks*/ incrementOutstandingTasks);
  });
  // Use a reference here to avoid refcount bump on futureGrads.
  auto& futureGrads = graphTask->future_result_;

  // Build a future that waits for the callbacks to execute (since callbacks
  // execute after the original future is completed). This ensures we return a
  // future that waits for all gradient accumulation to finish.
  auto accumulateGradFuture =
      c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

  futureGrads->addCallback([autogradContext, outputEdges, accumulateGradFuture](
                               c10::ivalue::Future& futureGrads) {
    if (futureGrads.hasError()) {
      // Don't accumulate gradients if we receive an error.
      // We must add the node information here since DistEngine::execute
      // waits on accumulateGradFuture and will throw an exception once we
      // set the error below.
      std::string errorMsg = c10::str(
          "Error on Node ",
          DistAutogradContainer::getInstance().getWorkerId(),
          ": ",
          futureGrads.tryRetrieveErrorMessage());
      accumulateGradFuture->setError(std::make_exception_ptr(
          c10::ivalue::Future::FutureError(std::move(errorMsg))));
      return;
    }

    try {
      const variable_list& grads = futureGrads.constValue().toTensorVector();
      TORCH_INTERNAL_ASSERT(grads.size() == outputEdges.size());
      accumulateGradFuture->markCompleted(c10::IValue());
    } catch (std::exception& e) {
      accumulateGradFuture->setErrorIfNeeded(std::current_exception());
    }
  });

  return accumulateGradFuture;
}

c10::intrusive_ptr<c10::ivalue::Future> DistEngine::executeSendFunctionAsync(
    const ContextPtr& autogradContext,
    const std::shared_ptr<SendRpcBackward>& sendFunction,
    bool retainGraph) {
  // Typically the local autograd engine ensures stream synchronizations between
  // nodes in the graph. However, for distributed autograd the sendFunction
  // inputs might have been retrieved over the wire on a separate stream and the
  // sendFunction itself runs on a different stream. As a result, we need to
  // manually synchronize those two streams here.
  const auto& send_backward_stream = sendFunction->stream();
  if (send_backward_stream) {
    for (const auto& grad : sendFunction->getGrads()) {
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      const auto default_stream = guard.getStream(grad.device());
      if (send_backward_stream != default_stream) {
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(default_stream);
        send_backward_stream->wait(event);
      }
    }
  }

  std::unique_lock<std::mutex> lock(initializedContextIdsLock_);
  if (initializedContextIds_.find(autogradContext->contextId()) ==
      initializedContextIds_.end()) {
    edge_list outputEdges;
    // Pass in a dummy graphRoot since all send functions are the roots.
    auto dummyRoot = std::make_shared<GraphRoot>(edge_list(), variable_list());
    computeDependencies(
        autogradContext, {}, {}, dummyRoot, outputEdges, retainGraph);

    // Mark the autograd context id as initialized and unlock.
    initializedContextIds_.insert(autogradContext->contextId());
    lock.unlock();

    // Enqueue the current send function.
    auto graphTask = autogradContext->retrieveGraphTask();
    // Run the autograd engine.
    auto accumulateGradFuture = runEngineAndAccumulateGradients(
        autogradContext,
        sendFunction,
        outputEdges,
        /*incrementOutstandingTasks=*/false);

    // Build the 'uber' future that waits for everything.
    auto callbackFuture =
        c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

    accumulateGradFuture->addCallback(
        [autogradContext,
         callbackFuture](c10::ivalue::Future& accumulateGradFuture) {
          try {
            if (accumulateGradFuture.hasError()) {
              // Perform cleanup at the end of the backward pass (before we mark
              // the future as completed).
              DistEngine::getInstance().cleanupBackwardPass(autogradContext);

              // Skip any further processing on errors.
              callbackFuture->setError(accumulateGradFuture.exception_ptr());
              return;
            }

            // Wait for all RPCs after the autograd engine is done.
            auto rpcFuture =
                autogradContext->clearAndWaitForOutstandingRpcsAsync();
            rpcFuture->addCallback([callbackFuture, autogradContext](
                                       c10::ivalue::Future& rpcFuture) {
              try {
                // Perform cleanup at the end of the backward pass (before
                // we mark the future as completed).
                DistEngine::getInstance().cleanupBackwardPass(autogradContext);
              } catch (std::exception& e) {
                callbackFuture->setErrorIfNeeded(std::current_exception());
                return;
              }

              // Finally mark the 'uber' future as completed.
              if (!rpcFuture.hasError()) {
                callbackFuture->markCompleted(c10::IValue());
              } else {
                callbackFuture->setError(rpcFuture.exception_ptr());
              }
            });
          } catch (std::exception& e) {
            callbackFuture->setErrorIfNeeded(std::current_exception());
          }
        });

    // Return the future which waits for all async processing to be done.
    return callbackFuture;
  } else {
    lock.unlock();
    auto graphTask = autogradContext->retrieveGraphTask();
    at::launch([this, graphTask, sendFunction]() {
      execute_graph_task_until_ready_queue_empty(
          /*node_task*/ NodeTask(graphTask, sendFunction, InputBuffer(0)),
          /*incrementOutstandingTasks*/ false);
    });
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    fut->markCompleted(c10::IValue());
    return fut;
  }
}

void DistEngine::execute(
    int64_t contextId,
    const variable_list& roots,
    bool retainGraph) {
  // Retrieve the context for the given context_id. This will throw if the
  // context_id is invalid.
  auto autogradContext =
      DistAutogradContainer::getInstance().retrieveContext(contextId);

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
    // Context should not have been initialized already.
    TORCH_INTERNAL_ASSERT(
        initializedContextIds_.find(autogradContext->contextId()) ==
        initializedContextIds_.end());

    computeDependencies(
        autogradContext, rootEdges, grads, graphRoot, outputEdges, retainGraph);

    // Mark the autograd context id as initialized.
    initializedContextIds_.insert(autogradContext->contextId());
  }

  BackwardPassCleanupGuard guard(autogradContext);

  // This needs to be blocking and as a result we wait for the future to
  // complete.
  runEngineAndAccumulateGradients(autogradContext, graphRoot, outputEdges)
      ->waitAndThrow();

  // Wait for all of the outstanding rpcs to complete.
  autogradContext->clearAndWaitForOutstandingRpcsAsync()->waitAndThrow();
}

void DistEngine::cleanupBackwardPass(const ContextPtr& autogradContext) {
  // Validate only the GraphTask is holding a reference to the Future
  // which holds gradients for the backward pass. This ensures that
  // after 'resetGraphTask' is called below, there are no remaining
  // references left to the gradients for the backward pass.
  //
  // This ensures our 'use_count' checks in
  // AccumulateGrad::accumulateGrad are correct and we're
  // not leaking any references to the gradients anywhere else.
  const auto& futureGrads =
      autogradContext->retrieveGraphTask()->future_result_;
  TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1);

  // Reset the graph task once we're done with all processing.
  autogradContext->resetGraphTask();

  // Clear any outstanding rpcs.
  autogradContext->clearOutstandingRpcs();

  // Clear the context id once we're done with the autograd engine
  // processing.
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  initializedContextIds_.erase(autogradContext->contextId());
}

size_t DistEngine::numBackwardPasses() const {
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  return initializedContextIds_.size();
}

std::unordered_map<std::string, int> DistEngine::getDebugInfo() const {
  std::unordered_map<std::string, int> debugInfo;
  debugInfo[kNumBackwardPasses] = numBackwardPasses();
  debugInfo[kNumAutogradContexts] =
      DistAutogradContainer::getInstance().numAutogradContexts();
  return debugInfo;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
