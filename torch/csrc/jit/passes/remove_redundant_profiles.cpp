#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

void RemoveRedundantProfiles(Block* block, AliasDb& db) {
  for (auto it = block->nodes().end()->reverseIterator();
       it != block->nodes().begin();) {
    Node* n = *it;
    it++;

    for (Block* b : n->blocks()) {
      RemoveRedundantProfiles(b, db);
    }

    // we only check prim::profile and not prim::profile_ivalue bc profile
    // is inserted on each use, while profile_ivalue is inserted on the def
    if (n->kind() != prim::profile ||
        n->input()->node()->kind() != prim::profile) {
      continue;
    }

    Node* input_node = n->input()->node();
    if (input_node->ty(attr::profiled_type) != n->ty(attr::profiled_type)) {
      continue;
    }

    if (!db.moveBeforeTopologicallyValid(input_node, n)) {
      continue;
    }

    n->output()->replaceAllUsesWith(n->input());
    n->destroy();
  }
}

void RemoveRedundantProfiles(std::shared_ptr<Graph>& graph) {
  AliasDb db(graph);
  RemoveRedundantProfiles(graph->block(), db);
}

} // namespace jit
} // namespace torch
