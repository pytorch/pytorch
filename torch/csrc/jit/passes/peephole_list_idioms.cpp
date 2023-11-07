#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/value_refinement_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
#include <limits>
#include <utility>

namespace torch {
namespace jit {

static c10::optional<size_t> normalizeIndex(int64_t index, size_t len) {
  if (index < 0) {
    index = index + len;
  }
  if (index >= 0 && index < static_cast<int64_t>(len)) {
    return index;
  } else {
    return c10::nullopt;
  }
}

// see [value refinement algorithm]

struct ListLenRefiner {
  ListLenRefiner(
      std::shared_ptr<Graph> graph,
      std::unordered_set<Value*>& mutated_lists)
      : graph_(std::move(graph)), mutated_lists_(mutated_lists) {}

  bool run() {
    std::unordered_set<Value*> li_with_len_use;
    collectListsToRefine(graph_->block(), li_with_len_use);
    if (lists_to_refine_.empty()) {
      return false;
    }
    ListRefinement refinements;
    RefineListLens(graph_->block(), std::move(refinements));
    return changed_;
  }

  // we only need to analyze lists that have multiple uses of len(), and we can
  // only analyze lists that are not mutated
  void collectListsToRefine(
      Block* b,
      std::unordered_set<Value*>& li_with_len_use) {
    for (Node* n : b->nodes()) {
      for (Block* block : n->blocks()) {
        collectListsToRefine(block, li_with_len_use);
      }

      if (n->kind() != aten::len) {
        continue;
      }

      auto first_input = n->input(0);
      if (first_input->type()->castRaw<ListType>() &&
          !mutated_lists_.count(first_input)) {
        if (!li_with_len_use.count(first_input)) {
          li_with_len_use.insert(first_input);
        } else {
          lists_to_refine_.insert(first_input);
        }
      }
    }
  }

  ListRefinement RefineListLens(Block* b, ListRefinement block_refinements) {
    active_refinements_.push_back(&block_refinements);
    for (Node* n : b->nodes()) {
      if (n->matches("aten::eq(int a, int b) -> bool") ||
          n->matches("aten::ne(int a, int b) -> bool")) {
        // check for one input constant and the other coming from len(li)
        for (size_t const_index : {0, 1}) {
          auto ival = constant_as<int64_t>(n->input(const_index));
          if (!ival) {
            continue;
          }
          auto li_len = n->input(1 - const_index);
          if (!li_len->node()->matches("aten::len.t(t[] a) -> int") ||
              !lists_to_refine_.count(li_len->node()->input())) {
            continue;
          }
          ListRefinement refine;
          refine[li_len->node()->input()] = *ival;
          boolean_value_refinements_[n->output()] = n->kind() == aten::eq
              ? BooleanRefinementMapping::TrueRefinements(std::move(refine))
              : BooleanRefinementMapping::FalseRefinements(std::move(refine));
        }
      } else if (n->kind() == aten::len) {
        if (auto maybe_len = tryFindRefinement(n->input(0))) {
          changed_ = true;
          WithInsertPoint guard(n);
          n->output()->replaceAllUsesWith(
              graph_->insertConstant(static_cast<int64_t>(*maybe_len)));
        }
      } else if (n->kind() == prim::If) {
        IfView if_n(n);
        bool has_cond_ref = boolean_value_refinements_.count(if_n.cond()) != 0;
        ListRefinement empty;
        auto true_block_refinements = RefineListLens(
            if_n.thenBlock(),
            has_cond_ref ? boolean_value_refinements_[if_n.cond()].true_refine()
                         : empty);
        auto false_block_refinements = RefineListLens(
            if_n.elseBlock(),
            has_cond_ref
                ? boolean_value_refinements_[if_n.cond()].false_refine()
                : empty);

        joinIfRefinements(
            n,
            throwing_blocks_,
            block_refinements,
            true_block_refinements,
            false_block_refinements,
            boolean_value_refinements_);
      } else {
        handleCommonRefinentOperators(
            n, throwing_blocks_, boolean_value_refinements_);
      }
    }
    active_refinements_.pop_back();
    return block_refinements;
  };

  c10::optional<int64_t> tryFindRefinement(Value* v) {
    for (const auto& ref : active_refinements_) {
      auto maybe_refinement = ref->find(v);
      if (maybe_refinement != ref->end()) {
        return maybe_refinement->second;
      }
    }
    return c10::nullopt;
  }

  std::shared_ptr<Graph> graph_;
  std::unordered_set<Value*> mutated_lists_;
  // candidate lists for optimizations
  std::unordered_set<Value*> lists_to_refine_;
  // A stack of active refinements, one for each block
  std::vector<ListRefinement*> active_refinements_;
  // A map from Boolean Value * -> associated refinements
  std::unordered_map<Value*, BooleanRefinementMapping>
      boolean_value_refinements_;
  std::unordered_set<Block*> throwing_blocks_;
  bool changed_ = false;
};

// This pass only does optimizations on lists which aren't mutated,
// so we first use the Alias Db to collect the set of list values
// which we shouldn't optimize.
struct PeepholeOptimizeListIdiomsImpl {
  PeepholeOptimizeListIdiomsImpl(
      std::shared_ptr<Graph> graph,
      bool refine_list_len)
      : graph_(std::move(graph)),
        aliasDb_(std::make_unique<AliasDb>(graph_)),
        refine_list_len_(refine_list_len) {}

  bool run() {
    collectMutatedLists(graph_->block());
    bool changed = runBlock(graph_->block());
    if (refine_list_len_) {
      changed |= ListLenRefiner(graph_, mutated_lists_).run();
    }
    return changed;
  }

 private:
  void checkForMutatedList(Value* v) {
    if (v->type()->castRaw<ListType>() && aliasDb_->hasWriters(v)) {
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

  bool optimizeSlice(Node* slice_node, Node* list_construct_node) {
    auto start_val = toIValue(slice_node->input(1));
    auto end_val = toIValue(slice_node->input(2));
    auto step_val = toIValue(slice_node->input(3));

    // All args must be constant to apply this optimization.
    if (start_val == c10::nullopt || end_val == c10::nullopt ||
        step_val == c10::nullopt) {
      return false;
    }

    int64_t start = start_val->isInt() ? start_val->to<int64_t>()
                                       : std::numeric_limits<int64_t>::max();
    int64_t end = end_val->isInt() ? end_val->to<int64_t>()
                                   : std::numeric_limits<int64_t>::max();
    int64_t step = step_val->isInt() ? step_val->to<int64_t>() : 1;

    size_t list_size = list_construct_node->inputs().size();
    size_t num_values = slice_indices_adjust(list_size, &start, &end, step);

    WithInsertPoint guard(slice_node);
    auto slice_list_construct =
        graph_->insertNode(graph_->create(prim::ListConstruct));
    slice_list_construct->output()->setType(slice_node->output()->type());
    for (size_t i = start, j = 0; j < num_values; ++j) {
      slice_list_construct->addInput(list_construct_node->input(i));
      i += step;
    }

    slice_node->output()->replaceAllUsesWith(slice_list_construct->output());
    if (mutated_lists_.count(slice_node->output())) {
      mutated_lists_.insert(slice_list_construct->output());
    }

    return true;
  }

  bool runBlock(Block* block) {
    bool changed = false;
    for (Node* node : block->nodes()) {
      for (Block* b : node->blocks()) {
        changed |= runBlock(b);
      }

      // only optimizing list ops
      if (node->inputs().empty() ||
          !node->input(0)->type()->castRaw<ListType>()) {
        continue;
      }

      auto first_input = node->input(0);

      // only optimizing ops with unmutated lists
      if (mutated_lists_.count(first_input)) {
        continue;
      }

      auto list_creation_node = first_input->node();
      if (list_creation_node->kind() != prim::ListConstruct) {
        continue;
      }

      if (node->kind() == aten::len) {
        WithInsertPoint guard(node);
        node->output()->replaceAllUsesWith(graph_->insertConstant(
            static_cast<int64_t>(first_input->node()->inputs().size())));
        changed = true;
      } else if (node->kind() == aten::__getitem__) {
        if (auto index = toIValue(node->input(1))) {
          size_t list_size = list_creation_node->inputs().size();
          if (auto norm_index = normalizeIndex(index->toInt(), list_size)) {
            node->output()->replaceAllUsesWith(
                list_creation_node->input(*norm_index));
            changed = true;
          }
        }
      } else if (node->kind() == prim::ListUnpack) {
        // if sizes are unequal it's a runtime error
        if (list_creation_node->inputs().size() != node->outputs().size()) {
          continue;
        }
        for (size_t i = 0; i < node->outputs().size(); ++i) {
          node->output(i)->replaceAllUsesWith(list_creation_node->input(i));
          changed = true;
        }
      } else if (node->kind() == aten::add) {
        if (node->inputs().size() != 2) {
          continue;
        }
        auto second_input = node->input(1);
        // already checked first, need to check second
        if (mutated_lists_.count(second_input)) {
          continue;
        }
        if (second_input->node()->kind() != prim::ListConstruct) {
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
      } else if (node->kind() == aten::slice) {
        changed |= optimizeSlice(node, first_input->node());
      }
    }
    return changed;
  }

  std::unordered_set<Value*> mutated_lists_;
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_;
  bool refine_list_len_;
};

bool PeepholeOptimizeListIdioms(
    const std::shared_ptr<Graph>& graph,
    bool refine_list_len) {
  PeepholeOptimizeListIdiomsImpl opt(graph, refine_list_len);
  return opt.run();
}

} // namespace jit
} // namespace torch
