#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>

#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/script/schema_matching.h>

#include <algorithm>

namespace torch {
namespace jit {

struct BlockInfo {
  int level;
};
using BlockInfoMap = std::unordered_map<Block*, BlockInfo>;

bool explodeTupleOutput(std::shared_ptr<Graph>& graph) {
  bool was_tuple_output = false;
  if (graph->outputs().size() == 1 &&
      graph->outputs()[0]->node()->kind() == prim::TupleConstruct) {
    Node* output_tup_construct = graph->outputs()[0]->node();
    for (Value* o : output_tup_construct->inputs()) {
      graph->registerOutput(o);
    }
    graph->eraseOutput(0);
    output_tup_construct->destroy();
    was_tuple_output = true;
  }
  return was_tuple_output;
}

void inspectBlocks(Block* b, BlockInfoMap& block_info, int level = 0) {
  TORCH_INTERNAL_ASSERT(!block_info.count(b));
  block_info[b] = BlockInfo{level};
  for (Node* n : b->nodes()) {
    if (n->kind() == prim::FakeScopeBlock) {
      inspectBlocks(n->blocks()[0], block_info, level + 1);
    }
  }
}

struct RemappedValueInfo {
  Value* v;
  int level;
};
using RemappingTable = std::unordered_map<Value*, RemappedValueInfo>;

void makeDefsDominateUses(
    Block* b,
    RemappingTable& remap,
    const BlockInfoMap& block_info) {
  auto find_common_ancestor_block_level = [&](Node* n) {
    std::unordered_set<Block*> seen;

    Block* my_provenance = b;
    while (my_provenance) {
      seen.insert(my_provenance);
      if (!my_provenance->owningNode()) {
        my_provenance = nullptr;
        continue;
      }
      my_provenance = my_provenance->owningNode()->owningBlock();
    }

    Block* their_provenance = n->owningBlock();
    while (their_provenance) {
      if (seen.count(their_provenance)) {
        return block_info.at(their_provenance).level;
      }
      if (!their_provenance->owningNode()) {
        their_provenance = nullptr;
        continue;
      }
      their_provenance = their_provenance->owningNode()->owningBlock();
    }
    TORCH_INTERNAL_ASSERT(false);
  };

  auto process_node = [&](Node* n) {
    // inspect uses
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      Value* inp = n->inputs()[i];
      if (inp->node()->owningBlock() != b) {
        int ancestor_level = find_common_ancestor_block_level(inp->node());

        Block* origin_block = inp->node()->owningBlock();
        if (ancestor_level == block_info.at(origin_block).level) {
          remap[inp] = {inp, block_info.at(origin_block).level};
        } else {
          Value* v_itr = inp;
          Block* b = origin_block;

          // Start off where we left off if there's an existing mapping
          if (remap.count(inp)) {
            int existing_level = remap[inp].level;
            while (block_info.at(b).level > existing_level) {
              b = b->owningNode()->owningBlock();
            }
            v_itr = remap[inp].v;
          }

          while (block_info.at(b).level > ancestor_level) {
            b->registerOutput(v_itr);
            Value* remapped = b->owningNode()->addOutput();
            v_itr = remapped;
            b = b->owningNode()->owningBlock();
          }
          remap[inp] = {v_itr, ancestor_level};
        }
        TORCH_INTERNAL_ASSERT(remap.count(inp));
        n->replaceInput(i, remap[inp].v);
      }
    }

    if (n->kind() == prim::FakeScopeBlock) {
      makeDefsDominateUses(n->blocks()[0], remap, block_info);
    }
  };

  process_node(b->param_node());
  for (Node* n : b->nodes()) {
    process_node(n);
  }
  process_node(b->return_node());
}

void convertReturnsToTuples(Block* b) {
  for (Node* n : b->nodes()) {
    if (n->kind() == prim::FakeScopeBlock) {
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
          n->output(0)->replaceAllUsesWith(tup_unpack->output(i));
          n->eraseOutput(0);
        }
      }
    }
  }
}

void lambdaLiftBlocksAndConvertToGraph(Block* b) {
  for (Node* n : b->nodes()) {
    if (n->kind() == prim::FakeScopeBlock) {
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

  // Inline base __module call
  for (auto n_itr = b->nodes().begin(); n_itr != b->nodes().end();) {
    Node* n = *n_itr++;
    if (n->kind() == prim::FakeScopeBlock && n->s(attr::scope) == "__module") {
      inlineCallTo(n, *n->g(attr::Subgraph));
    }
  }
}

void createMethodCalls(const std::shared_ptr<Graph>& g, script::Module* self) {
  // Add `self` as an input on subblocks
  Value* self_val;
  if (g->inputs().size() == 0 || !g->inputs()[0]->type()->cast<ClassType>() ||
      g->inputs()[0]->type() != self->type()) {
    self_val = g->insertInput(0)->setType(self->type())->setDebugName("self");
  } else {
    self_val = g->inputs()[0];
  }

  for (auto itr = g->nodes().begin(); itr != g->nodes().end();) {
    Node* n = *itr++;
    if (n->kind() == prim::FakeScopeBlock) {
      auto submod_name = n->s(attr::scope);
      auto submod = self->get_module(submod_name);
      createMethodCalls(n->g(attr::Subgraph), &submod);

      WithInsertPoint ip(n);
      Value* submod_val = g->insertGetAttr(self_val, submod_name);
      Function *f = nullptr;

      for (size_t method_idx = 0; ; method_idx++) {
        std::string method_name = "forward";
        if (method_idx != 0) {
          method_name += std::to_string(method_idx);
        }
        if (submod.find_method(method_name)) {
          continue;
        } else {
          auto qualname = c10::QualifiedName(submod.name(), method_name);
          f = submod.class_compilation_unit()->create_function(
              qualname, n->g(attr::Subgraph));
          submod.type()->addMethod(f);
          break;
        }
      }
      TORCH_INTERNAL_ASSERT(f);
      std::vector<NamedValue> nvs = {
          NamedValue(submod_val->node()->sourceRange(), submod_val)};
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

void runCleanupPasses(const std::shared_ptr<Graph>& g) {
  if (script::getInlineEverythingMode()) {
    Inline(*g);
  }
  EliminateDeadCode(g);
  LintGraph(g);
}

void runCleanupPasses(script::Module *m) {
  auto methods = m->get_methods();
  for (auto module : m->get_modules()) {
    runCleanupPasses(&module.module);
  }
  for (auto method : methods) {
    runCleanupPasses(method.graph());
  }
}

void FixupTraceScopeBlocks(
    std::shared_ptr<Graph>& graph,
    script::Module* self) {
  bool was_tuple_output = explodeTupleOutput(graph);
  // Gather level information about blocks in the graph
  BlockInfoMap block_info;
  inspectBlocks(graph->block(), block_info);
  RemappingTable table;
  makeDefsDominateUses(graph->block(), table, block_info);
  convertReturnsToTuples(graph->block());
  if (was_tuple_output || graph->outputs().size() > 1) {
    WithInsertPoint guard(graph->return_node());
    Value* out_tup =
        graph->insertNode(graph->createTuple(graph->outputs()))->output();
    while (graph->outputs().size()) {
      graph->eraseOutput(0);
    }
    graph->registerOutput(out_tup);
  }
  if (!self) {
    inlineScopeBlocks(graph->block());
    runCleanupPasses(graph);
  } else {
    lambdaLiftBlocksAndConvertToGraph(graph->block());
    createMethodCalls(graph, self);
    runCleanupPasses(self);
    // `graph` isn't referenced in `self` yet, so we need to run
    // this separately
    runCleanupPasses(graph);
  }
}

} // namespace jit
} // namespace torch
