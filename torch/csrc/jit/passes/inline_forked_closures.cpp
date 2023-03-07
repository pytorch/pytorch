#include <torch/csrc/jit/passes/inline_forked_closures.h>

#include <torch/csrc/jit/frontend/ir_emitter.h>

namespace torch {
namespace jit {

// Closure nodes are emitted as a tuple of (function %, context tuple %)
// Inside the closure the closure is then unpacked so that all closed over
// values are set. A function closing over a and b would look like:
// def foo(context):
//  a, b = context
//
// To fork the closure, we need to set each value in the context tuple
// as an explicit input to the fork node, and then within the closure
// subgraph, replace the context unpacking value with the new graph input.
// fork(foo) ->
// def foo(a, b):
void inlineForkedClosure(Node* fork_closure, NodeKind genKind) {
  Node* function_context_node = fork_closure->input()->node();

  if (function_context_node->inputs().size() != 2 ||
      function_context_node->inputs().at(0)->node()->kind() != prim::Closure ||
      function_context_node->inputs().at(1)->node()->kind() !=
          prim::TupleConstruct) {
    throw ErrorReport(fork_closure->sourceRange()) << "Cannot fork this value";
  }

  Node* function = function_context_node->inputs().at(0)->node();
  Node* context = function_context_node->inputs().at(1)->node();
  auto fork_graph = function->g(attr::Subgraph)->copy();
  auto g = fork_closure->owningGraph();
  Node* fork_node = g->create(genKind, 1)
                        ->insertAfter(fork_closure)
                        ->setSourceRange(fork_closure->sourceRange());

  if (fork_graph->inputs().size() != 1 ||
      !fork_graph->inputs().at(0)->type()->cast<TupleType>()) {
    throw ErrorReport(fork_node->sourceRange())
        << "Cannot fork lambda with parameters";
  }
  auto fork_graph_context = fork_graph->inputs().at(0);
  AT_ASSERT(fork_graph_context->uses().size() == 1);
  auto fork_graph_unpack = fork_graph_context->uses().at(0).user;

  for (size_t i = 0; i < context->inputs().size(); ++i) {
    auto cont_input = context->inputs().at(i);
    fork_node->addInput(cont_input);
    auto inp = fork_graph->insertInput(i)->copyMetadata(cont_input);
    fork_graph_unpack->outputs().at(i)->replaceAllUsesWith(inp);
  }
  fork_graph_unpack->destroy();
  fork_graph->eraseInput(fork_graph->inputs().size() - 1);
  fork_node->output()->copyMetadata(fork_closure->output());
  fork_closure->output()->replaceAllUsesWith(fork_node->output());
  fork_closure->destroy();
  fork_node->g_(attr::Subgraph, fork_graph);
  runCleanupPasses(fork_graph);
}

void inlineAwaitableThenClosure(Node* then_closure) {
  Node* function_context_node = then_closure->inputs()[0]->node();

  if (function_context_node->inputs().size() != 2 ||
      function_context_node->inputs().at(0)->node()->kind() != prim::Closure ||
      function_context_node->inputs().at(1)->node()->kind() !=
          prim::TupleConstruct) {
    throw ErrorReport(then_closure->sourceRange()) << "Cannot fork this value";
  }

  Node* function = function_context_node->inputs().at(0)->node();
  Node* context = function_context_node->inputs().at(1)->node();
  auto then_graph = function->g(attr::Subgraph)->copy();
  //  ir_emitter produces IR with prim::awaitableThenClosure; but just like the
  //  inlineForkedClosure method above, we need to modify it before running it.
  // We replace the prim::Closure with a prim::awaitable_then, which is a
  // callable. Note that a prim::awaitable_then function takes arguments (%x:
  // Await(T), %y: T). In the calling graph, we will have %x: Await(T)
  // available; it's the awaitable that the ops are operating on. However, we
  // won't have second argument %y: T. (At execution time, %y, the input to the
  // function we're calling as part of our awaitable_then call, will be the
  // output of the previous function queued in the Awaitable object). As a
  // workaround, we use prim::awaitable_then_input to create a dummy argument in
  // the calling graph that we pass into the awaitable_then, which makes the
  // graph valid. Before: %aw : Await(Tensor) = prim::awaitableClosure(%43)
  // ...
  // %1 : NoneType = prim::Closure_1()
  // %2 : () = prim::TupleConstruct()
  // %3 : (NoneType, ()) = prim::TupleConstruct(%1, %2)
  // = prim::awaitableThenClosure(%3, %aw)
  // ...
  // with prim::Closure_1 = graph(%context : (),
  //       %aw_ : Await(Tensor),
  //       %aw_out : Tensor):
  //   %4 : Function = prim::Constant[name="continuation_fn"]()
  //   %5 : Tensor = prim::CallFunction(%4, %aw_, %aw_out)
  //   return (%5)
  // After:
  // %aw : Await(Tensor) = prim::awaitable_0(%0)
  //   ...
  //   %2 : Tensor = prim::awaitable_then_input(%aw)
  //    = prim::awaitable_then_1(%aw, %2)
  //
  // with prim::awaitable_then_1 = graph(%aw_ : Await(Tensor),
  //       %aw_out : Tensor):
  //   %2 : Function = prim::Constant[name="continuation_fn"]()
  //   %4 : Tensor = prim::CallFunction(%2, %aw_, %aw_out)
  //   return (%4)
  auto g = then_closure->owningGraph();
  Node* then_node = g->create(prim::awaitable_then, 0)
                        ->insertAfter(then_closure)
                        ->setSourceRange(then_closure->sourceRange());
  Node* then_input_node = g->create(prim::awaitable_then_input, 1)
                              ->insertBefore(then_node)
                              ->setSourceRange(then_closure->sourceRange());
  TORCH_INTERNAL_ASSERT(then_closure->inputs().size() == 2);
  auto aw = then_closure->inputs().at(1);
  TORCH_INTERNAL_ASSERT(aw->type()->kind() == AwaitType::Kind);
  auto fake_aw_then_input = then_input_node->output();
  then_input_node->addInput(aw);

  then_node->addInput(fake_aw_then_input);

  auto then_graph_context = then_graph->inputs().at(0);
  {
    // Unpacking then graph context from tuple node to individual inputs
    if (then_graph_context->uses().size() == 1) {
      auto then_graph_unpack = then_graph_context->uses().at(0).user;

      for (size_t i = 0; i < context->inputs().size(); ++i) {
        auto cont_input = context->inputs().at(i);
        then_node->addInput(cont_input);
        auto inp = then_graph->addInput()->copyMetadata(cont_input);
        then_graph_unpack->outputs().at(i)->replaceAllUsesWith(inp);
      }
      then_graph_unpack->destroy();
    }
  }

  then_graph->eraseInput(0);
  then_closure->destroy();
  then_node->g_(attr::Subgraph, then_graph);
  runCleanupPasses(then_graph);
}

void inlineForkedClosures(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    switch (n->kind()) {
      case prim::forkClosure: {
        inlineForkedClosure(n, prim::fork);
      } break;
      case prim::awaitableClosure: {
        inlineForkedClosure(n, prim::awaitable);
      } break;
      case prim::awaitableThenClosure: {
        inlineAwaitableThenClosure(n);
      } break;
      default: {
        for (Block* b : n->blocks()) {
          inlineForkedClosures(b);
        }
      } break;
    }
  }
}

void inlineForkedClosures(std::shared_ptr<Graph>& to_clean) {
  inlineForkedClosures(to_clean->block());
}

} // namespace jit
} // namespace torch
