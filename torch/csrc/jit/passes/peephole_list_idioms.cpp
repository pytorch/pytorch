#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

c10::optional<size_t> normalizeIndex(int64_t index, size_t len) {
  if (index < 0) {
    index = index + len;
  }
  if (index >= 0 && index < static_cast<int64_t>(len)) {
    return index;
  } else {
    return c10::nullopt;
  }
}

// This pass only does optimizations on lists which aren't mutated,
// so we first use the Alias Db to collect the set of list values
// which we shouldn't optimize.
struct PeepholeOptimizeListIdiomsImpl {
  PeepholeOptimizeListIdiomsImpl(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)),
        aliasDb_(torch::make_unique<AliasDb>(graph_)) {}

  bool run() {
    collectMutatedLists(graph_->block());
    return runBlock(graph_->block());
  }

 private:
  void checkForMutatedList(Value* v) {
    if (v->type()->cast<ListType>() && aliasDb_->hasWriters(v)) {
      mutated_lists_.insert(v);
    }
  }

  void collectMutatedLists(Block* b) {
    for (Value* v : b->inputs()) {
      checkForMutatedList(v);
    }
    for (Node* n : b->nodes()) {
      for (Value* v : n->outputs()) {
        checkForMutatedList(v);
      }
      for (Block* block : n->blocks()) {
        collectMutatedLists(block);
      }
    }
  }

  bool runBlock(Block* block) {
    bool changed = false;
    for (Node* node : block->nodes()) {
      for (Block* b : node->blocks()) {
        changed |= runBlock(b);
      }

      // only optimizing list ops
      if (node->inputs().size() == 0 ||
          !node->inputs().at(0)->type()->cast<ListType>()) {
        continue;
      }

      auto first_input = node->inputs().at(0);

      // only optimizing ops with unmutated lists
      if (mutated_lists_.count(first_input)) {
        continue;
      }

      if (node->kind() == aten::len) {
        if (first_input->node()->kind() == prim::ListConstruct) {
          WithInsertPoint guard(node);
          node->output()->replaceAllUsesWith(graph_->insertConstant(
              static_cast<int64_t>(first_input->node()->inputs().size())));
          changed = true;
        }
      } else if (node->kind() == aten::__getitem__) {
        auto list_creation_node = first_input->node();
        if (list_creation_node->kind() == prim::ListConstruct) {
          if (auto index = toIValue(node->inputs().at(1))) {
            size_t list_size = list_creation_node->inputs().size();
            if (auto norm_index = normalizeIndex(index->toInt(), list_size)) {
              node->output()->replaceAllUsesWith(
                  list_creation_node->inputs().at(*norm_index));
              changed = true;
            }
          }
        }
      } else if (node->kind() == prim::ListUnpack) {
        auto list_creation_node = first_input->node();
        if (list_creation_node->kind() == prim::ListConstruct) {
          // if sizes are unequal it's a runtime error
          if (list_creation_node->inputs().size() != node->outputs().size()) {
            continue;
          }
          for (size_t i = 0; i < node->outputs().size(); ++i) {
            node->output(i)->replaceAllUsesWith(
                list_creation_node->inputs().at(i));
            changed = true;
          }
        }
      } else if (node->kind() == aten::add) {
        if (node->inputs().size() != 2) {
          continue;
        }
        auto second_input = node->inputs().at(1);
        // already checked first, need to check second
        if (mutated_lists_.count(second_input)) {
          continue;
        }
        if (first_input->node()->kind() != prim::ListConstruct ||
            second_input->node()->kind() != prim::ListConstruct) {
          continue;
        }
        WithInsertPoint guard(node);
        auto list_construct =
            graph_->insertNode(graph_->create(prim::ListConstruct));
        list_construct->output()->setType(node->output()->type());
        for (Value* v : first_input->node()->inputs()) {
          list_construct->addInput(v);
        }
        for (Value* v : second_input->node()->inputs()) {
          list_construct->addInput(v);
        }
        node->output()->replaceAllUsesWith(list_construct->output());
        if (mutated_lists_.count(node->output())) {
          mutated_lists_.insert(list_construct->output());
        }
        changed = true;
      }
    }
    return changed;
  }

  std::unordered_set<Value*> mutated_lists_;
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_;
};

bool PeepholeOptimizeListIdioms(const std::shared_ptr<Graph>& graph) {
  PeepholeOptimizeListIdiomsImpl opt(graph);
  return opt.run();
}

} // namespace jit
} // namespace torch
