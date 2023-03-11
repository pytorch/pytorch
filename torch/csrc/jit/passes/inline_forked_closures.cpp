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
