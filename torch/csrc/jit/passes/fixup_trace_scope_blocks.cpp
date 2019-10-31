#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>

#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/script/schema_matching.h>

#include <algorithm>

namespace torch {
namespace jit {

namespace {

bool isEligibleNode(Node* n) {
  return n->kind() == prim::FakeScopeBlock || n->kind() == prim::FakeFork;
}

} // namespace

// Iterate through all the nodes in program order and--for each use--
// if the Value referenced is not in a scope that dominates the node,
// add block and Node outputs to lift it into a scope in which
// it dominates the Use.
struct MakeDefsDominateUses {
  MakeDefsDominateUses() {}

  void run(Block* b) {
    processNode(b->param_node(), b);
    for (Node* n : b->nodes()) {
      processNode(n, b);
    }
    processNode(b->return_node(), b);
  }

 private:
  void processNode(Node* n, Block* b) {
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      Value* inp = n->inputs()[i];
      if (inp->node()->owningBlock() != b) {
        // Find the common ancestor block between this node and the node that
        // produced this input. For this input Use to be valid, the Value's
        // def must be present in this common ancestor node.
        Block* common_ancestor = n->findCommonAncestorBlockWith(inp->node());

        Value* v_itr = inp;
        Block* b = inp->node()->owningBlock();

        // Starting from the initial def for this input, iterate to
        // wider and wider blocks, adding Block outputs and Node outputs
        // along the way. Then, log the lifted values in the remap table
        // so we can make subsequent Uses refer to the lifted value, if
        // the domination condition is met.
        while (b != common_ancestor) {
          // Already lifted to this level, switch to remapped value
          // and continue.
          if (remap.count(v_itr)) {
            v_itr = remap[inp];
            b = v_itr->node()->owningBlock();
            continue;
          }

          b->registerOutput(v_itr);
          Value* remapped = b->owningNode()->addOutput();
          v_itr = remapped;
          b = b->owningNode()->owningBlock();
        }
        // From now on, references to `inp` will be replaced with
        // references to `v_iter`, the lifted Value
        remap[inp] = v_itr;
        n->replaceInput(i, remap[inp]);
      }
    }

    if (isEligibleNode(n)) {
      run(n->blocks()[0]);
    }
  }

  // This holds the mapping between a Value* we would see in a Use
  // and the lifted value, if present. We use this to ensure that
  // Uses refer to a Value* that is in a dominating scope.
  using RemappingTable = std::unordered_map<Value*, Value*>;
  RemappingTable remap;
};

// For all blocks except graph->block(), convert multiple block
// returns to a TupleConstruct. This is required for turning the
// blocks into Methods. (and in the case that self is nullptr,
// it is required to properly inline the blocks).
void convertReturnsToTuples(Block* b) {
  for (Node* n : b->nodes()) {
    if (n->kind() == prim::FakeFork) {
      convertReturnsToTuples(n->blocks()[0]);
    } else if (n->kind() == prim::FakeScopeBlock) {
      TORCH_INTERNAL_ASSERT(n->blocks().size() == 1);
      convertReturnsToTuples(n->blocks()[0]);

      Graph* g = b->owningGraph();
      Block* sub_block = n->blocks()[0];
      if (sub_block->outputs().size() > 1) {
        {
          // Make block returns go through a Tuple
          WithInsertPoint guard(sub_block->return_node());
          Node* return_tup =
              g->insertNode(g->createTuple(sub_block->outputs()));
          while (sub_block->outputs().size()) {
            sub_block->eraseOutput(0);
          }
          sub_block->registerOutput(return_tup->output());
        }

        // Make node outputs a single tuple;
        std::vector<TypePtr> types;
        for (size_t i = 0; i < n->outputs().size(); ++i) {
          types.push_back(n->output(0)->type());
        }
        Value* tup_output = n->addOutput()->setType(TupleType::create(types));
        Node* tup_unpack = g->createTupleUnpack(tup_output)->insertAfter(n);
        for (size_t i = 0; i < tup_unpack->outputs().size(); ++i) {
          auto rev_idx = tup_unpack->outputs().size() - i - 1;
          n->output(rev_idx)->replaceAllUsesWith(tup_unpack->output(i));
          n->eraseOutput(rev_idx);
        }
      } else if (sub_block->outputs().size() == 0) {
        WithInsertPoint guard(sub_block->return_node());
        sub_block->registerOutput(g->insertNode(g->createNone())->output());
        n->addOutput()->setType(NoneType::get());
      }
    }
  }
}

// Lambda lift Values (i.e. add Graph inputs for the purpose of
// referencing values that dominate the block) and convert
// the block to a Graph. blocks()[0] on each FakeScopeBlock then
// appears as a Graph attribute attr::Subgraph
void lambdaLiftBlocksAndConvertToGraph(Block* b) {
  for (Node* n : b->nodes()) {
    if (isEligibleNode(n)) {
      lambdaLiftBlocksAndConvertToGraph(n->blocks()[0]);

      auto graph = std::make_shared<Graph>();
      std::unordered_map<Value*, Value*> remaps;
      graph->block()->cloneFrom(n->blocks()[0], [&](Value* v) {
        if (!remaps.count(v)) {
          remaps[v] = graph->addInput()->copyMetadata(v);
          n->addInput(v);
        }
        return remaps[v];
      });
      LintGraph(graph);
      n->g_(attr::Subgraph, graph);
      n->eraseBlock(0);
    }
  }
}

namespace {

// Find a unique name to add this method as
// We try {method_name}, {method_name}1, {method_name}2, ...
std::string mangleMethodName(
    const std::string& method_name,
    const script::Module& mod) {
  for (size_t method_idx = 0;; method_idx++) {
    auto mangled = method_name;
    if (method_idx != 0) {
      mangled += std::to_string(method_idx);
    }
    if (!mod.find_method(mangled)) {
      return mangled;
    }
  }
  TORCH_INTERNAL_ASSERT(false);
}

} // namespace

// Add `self` argument with the correct Module type to each scope block. This is
// necessary downstream when we emit code to dereference Module values before
// invoking Methods on them.
void addSelfArgsToBlocks(
    Block* b,
    script::Module this_level_self,
    std::vector<std::string> prefix,
    Value* self) {
  // Top level block already has `self`, skip it.
  if (prefix != std::vector<std::string>{"__module"}) {
    b->addInput()->setType(this_level_self.type())->setDebugName("self");
  }

  for (Node* n : b->nodes()) {
    if (n->kind() == prim::FakeFork) {
      addSelfArgsToBlocks(n->blocks()[0], this_level_self, prefix, self);
    } else if (n->kind() == prim::FakeScopeBlock) {
      // First, figure out what module we need to get in scope to call
      // the method
      auto sub_atoms = c10::QualifiedName(n->s(attr::scope)).atoms();
      TORCH_INTERNAL_ASSERT(sub_atoms.size() > prefix.size());

      script::Module callee_mod = this_level_self;
      Value* callee_val = self;

      WithInsertPoint ip(n);

      for (size_t i = 0; i < sub_atoms.size(); ++i) {
        if (i < prefix.size()) {
          TORCH_INTERNAL_ASSERT(sub_atoms[i] == prefix[i]);
        } else {
          callee_val =
              b->owningGraph()->insertGetAttr(callee_val, sub_atoms[i]);
          callee_mod = callee_mod.get_module(sub_atoms[i]);
        } // if (i < prefix.size())
      } // for (size_t i = 0; i < sub_atoms.size(); ++i)

      n->addInput(callee_val);

      addSelfArgsToBlocks(n->blocks()[0], callee_mod, sub_atoms, callee_val);
    } // if (n->kind() == prim::FakeScopeBlock)
  } // for (Node *n : b->nodes())
}

// Register the attr::Subgraph Graph values as Functions in the
// class compilation unit and register that Function as a method
// on the corresponding Module in the Module hierarchy. Note that we
// unique the methods by naming them forward, forward1, forward2...
void createMethodCalls(
    const std::shared_ptr<Graph>& g,
    script::Module this_level_self,
    std::vector<std::string> prefix,
    Value* self) {
  TORCH_INTERNAL_ASSERT(self->type()->isSubtypeOf(this_level_self.type()));

  for (auto node_itr = g->nodes().begin(); node_itr != g->nodes().end();) {
    Node* n = *node_itr++;
    if (n->kind() == prim::FakeFork) {
      createMethodCalls(n->g(attr::Subgraph), this_level_self, prefix, self);
    } else if (n->kind() == prim::FakeScopeBlock) {
      // First, figure out what module we need to get in scope to call
      // the method
      auto sub_atoms = c10::QualifiedName(n->s(attr::scope)).atoms();
      TORCH_INTERNAL_ASSERT(sub_atoms.size() > prefix.size());

      WithInsertPoint ip(n);

      Value* callee_val = n->inputs()[0];
      script::Module callee_mod = this_level_self;
      for (size_t i = 0; i < sub_atoms.size(); ++i) {
        if (i < prefix.size()) {
          TORCH_INTERNAL_ASSERT(sub_atoms[i] == prefix[i]);
        } else {
          callee_mod = callee_mod.get_module(sub_atoms[i]);
        }
      }

      createMethodCalls(
          n->g(attr::Subgraph), callee_mod, sub_atoms, callee_val);

      auto mangled_method_name = mangleMethodName("forward", callee_mod);
      auto qualname =
          c10::QualifiedName(callee_mod.name(), mangled_method_name);
      Function* f = callee_mod.class_compilation_unit()->create_function(
          qualname, n->g(attr::Subgraph));
      callee_mod.type()->addMethod(f);

      std::vector<NamedValue> nvs;
      for (Value* i : n->inputs()) {
        nvs.emplace_back(i->node()->sourceRange(), i);
      }
      auto schema =
          script::matchSchema(f->getSchema(), n->sourceRange(), *g, nvs, {});
      Value* retval = g->insertMethodCall(f->qualname().name(), schema);
      n->output()->replaceAllUsesWith(retval);
      n->destroy();
    }
  }
}

void inlineScopeBlocks(Block* b) {
  for (auto n_itr = b->nodes().begin(); n_itr != b->nodes().end();) {
    Node* n = *n_itr++;
    for (Block* sub_b : n->blocks()) {
      inlineScopeBlocks(sub_b);
    }
    if (n->kind() == prim::FakeScopeBlock) {
      // Convert the block to a graph so we can inline it
      auto graph = std::make_shared<Graph>();
      std::unordered_map<Value*, Value*> remaps;
      graph->block()->cloneFrom(n->blocks()[0], [&](Value* v) {
        remaps[v] = graph->block()->addInput()->copyMetadata(v);
        n->addInput(v);
        return remaps[v];
      });

      inlineCallTo(n, *graph);
    }
  }
}

void convertFakeForksToRealForks(const std::shared_ptr<Graph>& g) {
  for (auto itr = g->nodes().begin(); itr != g->nodes().end();) {
    Node* n = *itr++;
    if (n->kind() == prim::FakeFork) {
      WithInsertPoint guard(n);
      Node* new_fork_node =
          g->insertNode(g->create(prim::fork, n->outputs().size()))
              ->copyAttributes(*n);
      for (Value* i : n->inputs()) {
        new_fork_node->addInput(i);
      }
      for (size_t i = 0; i < new_fork_node->outputs().size(); ++i) {
        new_fork_node->outputs()[i]->copyMetadata(n->outputs()[i]);
        n->outputs()[i]->replaceAllUsesWith(new_fork_node->outputs()[i]);
      }
      n->destroy();
    }
  }
}

// Run a few clean-up passes to make the graph a bit cleaner.
void runCleanupPasses(const std::shared_ptr<Graph>& g) {
  for (Node* n : g->nodes()) {
    if (n->kind() == prim::FakeFork) {
      auto subgraph = n->g(attr::Subgraph);
      if (script::getInlineEverythingMode()) {
        Inline(*subgraph);
      }
      convertFakeForksToRealForks(subgraph);
      LowerSimpleTuples(subgraph);
      EliminateDeadCode(subgraph);
      LintGraph(subgraph);
    }
  }
  if (script::getInlineEverythingMode()) {
    Inline(*g);
  }
  convertFakeForksToRealForks(g);
  LowerSimpleTuples(g);
  EliminateDeadCode(g);
  LintGraph(g);
}

void runCleanupPasses(script::Module* m) {
  auto methods = m->get_methods();
  for (auto module : m->get_modules()) {
    runCleanupPasses(&module.module);
  }
  for (auto& method : methods) {
    runCleanupPasses(method.graph());
  }
}

void FixupTraceScopeBlocks(
    std::shared_ptr<Graph>& graph,
    script::Module* self) {
  MakeDefsDominateUses().run(graph->block());
  convertReturnsToTuples(graph->block());
  if (!self) {
    // We have no Module, so we're just going to inline everything.
    // This should give us a totally flag graph.
    inlineScopeBlocks(graph->block());
    // For FakeFork nodes
    lambdaLiftBlocksAndConvertToGraph(graph->block());
    runCleanupPasses(graph);
  } else {
    addSelfArgsToBlocks(
        graph->block(), *self, {"__module"}, graph->inputs()[0]);
    lambdaLiftBlocksAndConvertToGraph(graph->block());
    createMethodCalls(graph, *self, {"__module"}, graph->inputs()[0]);
    runCleanupPasses(self);
    // `graph` isn't referenced in `self` yet, so we need to run
    // this separately
    runCleanupPasses(graph);
  }
}

} // namespace jit
} // namespace torch
