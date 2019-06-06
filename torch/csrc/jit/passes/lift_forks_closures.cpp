#include <torch/csrc/jit/passes/lift_forks_closures.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/compiler.h>

namespace torch {
namespace jit {
namespace script {

void lambdaLiftFork(Node* fork_node) {
  // Fork a new graph from its orignal owning graph
  auto forked_graph = std::make_shared<Graph>();
  auto body_block = fork_node->blocks()[0];

  // Make sure we capture everything in the new graph.
  // The uncaptured values will be added to the fork signature.
  std::unordered_map<Value*, Value*> uncaptures_map;
  auto env = [&](Value* v) -> Value* {
    if (!uncaptures_map.count(v)) {
      // Capture values for both graphs
      uncaptures_map[v] = forked_graph->addInput()->copyMetadata(v);
      fork_node->addInput(v);
    }
    return uncaptures_map[v];
  };
  forked_graph->block()->cloneFrom(body_block, env);

  // Separate the subgraph and clean up the orignal one
  fork_node->g_(attr::Subgraph, forked_graph);
  fork_node->eraseBlock(0);
}

void liftFork(Node* fork_node) {
  // forks from tracing or nested forks will already be lifted
  if (fork_node->hasAttribute(attr::Subgraph)) {
    AT_ASSERT(fork_node->blocks().size() == 0);
    return;
  }

  // lambda lift fork is exposed as a separate function
  // so that it can be invoked in the tracer
  lambdaLiftFork(fork_node);
  runCleanupPasses(fork_node->g(attr::Subgraph), /*convert_to_ssa*/ false);
}

// Closures are initially emitted as prim::Function nodes with a single block.
// Here, we convert the block to a subgraph, adding all closed over variables
// as a context tuple input to the closure node.
// At this point the closure has already undergone conversion to SSA,
// so closed over variables will just be value * that are not set in the
// closure block.
// Within the closure subgraph, the context tuple is unpacked and the unpacked
// values are used for closed over values.
void liftClosure(Node* closure) {
  auto block = closure->blocks().at(0);
  auto subgraph = std::make_shared<Graph>();
  // closures/forks can be nested, so use closure owning graph
  auto g = closure->owningGraph();
  Node* pack_context =
      g->create(prim::TupleConstruct, {}, 1)->insertAfter(closure);
  Value* context = subgraph->addInput("context");
  // cannot use createTupleUnpack because the type is not known yet
  Node* unpack_context =
      subgraph->insertNode(subgraph->create(prim::TupleUnpack, {context}, 0));

  std::unordered_map<Value*, Value*> captures;
  auto env = [&](Value* v) -> Value* {
    auto it = captures.find(v);
    if (it != captures.end()) {
      return it->second;
    }
    pack_context->addInput(v);
    Value* r = unpack_context->addOutput()->copyMetadata(v);
    captures[v] = r;
    return r;
  };
  subgraph->block()->cloneFrom(block, env);
  auto context_type = TupleType::create(
      fmap(pack_context->inputs(), [](Value* v) { return v->type(); }));
  pack_context->output()->setType(context_type);
  auto closure_tuple =
      g->create(prim::TupleConstruct, {}, 1)->insertAfter(pack_context);
  closure->output()->replaceAllUsesWith(closure_tuple->output());
  closure_tuple->addInput(closure->output());
  closure_tuple->addInput(pack_context->output());
  closure_tuple->output()->setType(
      TupleType::create({closure->output()->type(), context_type}));
  closure->eraseBlock(0);
  closure->g_(attr::Subgraph, std::move(subgraph));
  runCleanupPasses(closure->g(attr::Subgraph), /*convert_to_ssa*/ false);
}

void liftClosuresAndForks(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    switch (n->kind()) {
      case prim::Function: {
        liftClosure(n);
      } break;
      case prim::fork: {
        liftFork(n);
      } break;
      default: {
        for (Block* b : n->blocks()) {
          liftClosuresAndForks(b);
        }
      }
    }
  }
}

void liftClosuresAndForks(const std::shared_ptr<Graph>& to_clean) {
  liftClosuresAndForks(to_clean->block());
}

} // namespace script
} // namespace jit
} // namespace torch
