#include <torch/csrc/jit/passes/lift_closures.h>

#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir.h>

#include <utility>

namespace torch {
namespace jit {

// Closures are initially emitted as prim::Closure nodes with a single block.
// Here, we convert the block to a subgraph, adding all closed over variables
// as a context tuple input to the closure node.
// At this point the closure has already undergone conversion to SSA,
// so closed over variables will just be value * that are not set in the
// closure block.
// Within the closure subgraph, the context tuple is unpacked and the unpacked
// values are used for closed over values.
static void liftClosure(Node* closure) {
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
  context->setType(context_type);
  pack_context->output()->setType(context_type);
  auto closure_tuple =
      g->create(prim::TupleConstruct, {}, 1)->insertAfter(pack_context);
  closure->output()->replaceAllUsesWith(closure_tuple->output());
  closure_tuple->addInput(closure->output());
  closure_tuple->addInput(pack_context->output());
  closure_tuple->output()->setType(
      TupleType::create({closure->output()->type(), std::move(context_type)}));
  closure->eraseBlock(0);
  closure->g_(attr::Subgraph, std::move(subgraph));
  runCleanupPasses(closure->g(attr::Subgraph));
}

static void liftClosures(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    switch (n->kind()) {
      case prim::Closure: {
        liftClosure(n);
      } break;
      default: {
        for (Block* b : n->blocks()) {
          liftClosures(b);
        }
      }
    }
  }
}

void liftClosures(const std::shared_ptr<Graph>& to_clean) {
  liftClosures(to_clean->block());
}

} // namespace jit
} // namespace torch
