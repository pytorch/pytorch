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
  std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
      << " then_closure.node:" << *then_closure
      << std::endl;
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
  std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
      << " then_graph:" << *then_graph
      << "\n function.node:" << *function
      << "\n context.node:" << *context
      << std::endl;

  auto g = then_closure->owningGraph();
  Node* then_node = g->create(prim::awaitable_then, 0)
                        ->insertAfter(then_closure)
                        ->setSourceRange(then_closure->sourceRange());
  Node* then_input_node = g->create(prim::awaitable_then_input, 1)
                        ->insertBefore(then_node)
                        ->setSourceRange(then_closure->sourceRange());
  std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
      << " then_closure->inputs().size():" << then_closure->inputs().size()
      << std::endl;
  TORCH_INTERNAL_ASSERT(then_closure->inputs().size() == 2);
  auto aw = then_closure->inputs().at(1);
  TORCH_INTERNAL_ASSERT(aw->type()->kind() == AwaitType::Kind);
  auto fake_aw_then_input = then_input_node->output();
  then_input_node->addInput(aw);

  then_node->addInput(aw);
  then_node->addInput(fake_aw_then_input);

//  if (then_graph->inputs().size() != 2 ||
//      !then_graph->inputs().at(0)->type()->cast<TupleType>()) {
//    throw ErrorReport(then_node->sourceRange())
//        << "Expect await then closure with 2 arguments: fn context tuple, await out";
//  }
  auto then_graph_context = then_graph->inputs().at(0);
  {
    std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
        << " then_graph_context->uses().size():" << then_graph_context->uses().size()
        << std::endl;
    if (then_graph_context->uses().size() == 1) {
      auto fork_graph_unpack = then_graph_context->uses().at(0).user;
      std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
          << " fork_graph_unpack:" << *fork_graph_unpack
          << std::endl;

      for (size_t i = 0; i < context->inputs().size(); ++i) {
        auto cont_input = context->inputs().at(i);
        then_node->addInput(cont_input);
        auto inp = then_graph->addInput()->copyMetadata(cont_input);
        fork_graph_unpack->outputs().at(i)->replaceAllUsesWith(inp);
      }
      fork_graph_unpack->destroy();
    }
  }
  std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
      << " THEN_GRAPH.before_erase:" << *then_graph
      << std::endl;

  then_graph->eraseInput(0);
  // auto aw_input = then_graph->insertInput(0, "aw");
  // aw_input->setType(aw->type());
  std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
      << " THEN_GRAPH2:" << *then_graph
      << std::endl;
  std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
      << " THEN_GRAPH2.G:" << *g
      << std::endl;
  //then_node->output()->copyMetadata(then_closure->output());
  //then_closure->output()->replaceAllUsesWith(then_node->output());
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
  std::cout << "XXX" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
      << " INLINE_FORKED_CLOSURES:\n" << *to_clean
      << std::endl;
  inlineForkedClosures(to_clean->block());
}

} // namespace jit
} // namespace torch
