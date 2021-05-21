#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
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

// Refine from Value of type List -> len of list
// If a refinement mapping of List Value * -> len is present in a block
// the list is guaranteed to be that length
// TODO: vector may be faster
using ListRefinement = std::unordered_map<Value*, int64_t>;

// The intersection of the refinements is the Value* which are in both
// refinements and are refined to the same length
// in an example like:
// if cond:
//    x = len(a) == 4 and len(b) == 5
// else:
//    x = len(a) == 4
// For the x output of the node we take the intersection between
// the refinements stored on each block output, which will result
// in only the refinement of len(a) == 4

ListRefinement intersectRefinements(
    const ListRefinement& ref1,
    const ListRefinement& ref2) {
  ListRefinement out;
  for (const auto& pair : ref1) {
    auto val2 = ref2.find(pair.first);
    if (val2 != ref2.end() && val2->second == pair.second) {
      out[pair.first] = pair.second;
    }
  }
  return out;
}

// To union, just take all refinements from both inputs. We do not need to worry
// about len refinements disagreeing because a path like `if len(x) == 4 and
// len(x) == 5` will never be taken
// in an example like:
// if len(a) == 5:
//     x = len(b) == 4
// else:
//     x = False
// For the output x Value, if is true then the refinements present in the true
// block must also be true, so we take the union of `len(a) == 5` and len(b) ==
// 4` and assign them to true refinements of the output x value. This is a very
// common pattern in desugaring of `and` or `or` boolean expressions

ListRefinement unionRefinements(
    const ListRefinement& ref1,
    const ListRefinement& ref2) {
  ListRefinement out = ref1;
  out.insert(ref2.begin(), ref2.end());
  return out;
}

// Represents the refinement information that can be carried on a boolean
struct BooleanRefinementMapping {
  BooleanRefinementMapping(
      ListRefinement true_refine,
      ListRefinement false_refine)
      : true_refine_(std::move(true_refine)),
        false_refine_(std::move(false_refine)){};
  BooleanRefinementMapping() = default; // empty

  static BooleanRefinementMapping FalseRefinements(
      ListRefinement false_refine) {
    return BooleanRefinementMapping({}, std::move(false_refine));
  }

  static BooleanRefinementMapping TrueRefinements(ListRefinement true_refine) {
    return BooleanRefinementMapping(std::move(true_refine), {});
  }

  BooleanRefinementMapping intersectBooleanRefinementMapping(
      BooleanRefinementMapping& other) {
    return BooleanRefinementMapping(
        intersectRefinements(true_refine_, other.true_refine()),
        intersectRefinements(false_refine_, other.false_refine()));
  }

  ListRefinement& true_refine() {
    return true_refine_;
  }

  ListRefinement& false_refine() {
    return false_refine_;
  }

 private:
  ListRefinement true_refine_;
  ListRefinement false_refine_;
};

// When a comparison like `cond = len(x) == 4` or `cond = len(x) != 4` is made,
// `cond` value carries information (refinements) about the len of `x`.
// When `cond` is used as the conditional of an if statement, the information
// it carries for its true value can be inserted into the true block
// and the same for its false value.
// For something like `y = len(x) if len(x) == 1 else 1`, in the true branch
// we can replace len(x) with 1 because the true refinements from `len(x) == 1`
// will be present in the true block.
// Additionally, we can optimize something like:
// if len(x) != 4:
//    raise Exception(...)
// return len(x)
// Because the true block always throws, whatever refinements exist in the false
// block become present in the owning block of the if node. We can also merge
// refinements carried by two different booleans across an if node join by
// taking the intersections of their refinements.
// if cond:
//    z = len(x) == 4 and len(y) == 5
// else:
//    z = len(x) == 4
// Here, z's true value will refine the len(x) to 4, but not len(y).
// If the code was written as:
// if cond:
//    z = len(x) == 4 and len(y) == 5
// else:
//    z = False
//
// Then z's true value would refine x and y, because if z is true it had to have
// come from the true block. Code that is written with `and` or `or` will
// desugar to something similar. Additionally, any True refinements that were
// present on `cond` can also be associated with the if node True output value.

struct ListLenRefiner {
  ListLenRefiner(
      std::shared_ptr<Graph> graph,
      std::unordered_set<Value*>& mutated_lists)
      : graph_(std::move(graph)), mutated_lists_(mutated_lists) {}

  bool run() {
    std::unordered_set<Value*> li_with_len_use;
    collectListsToRefine(graph_->block(), li_with_len_use);
    if (lists_to_refine_.size() == 0) {
      return false;
    }
    ListRefinement refinements;
    RefineListLens(graph_->block(), refinements);
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
      if (first_input->type()->cast<ListType>() &&
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
          auto li_len = n->input(const_index - 1);
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
      }
      if (n->kind() == prim::RaiseException) {
        throwing_blocks_.insert(b);
      }
      if (n->kind() == aten::len) {
        if (auto maybe_len = tryFindRefinement(n->input(0))) {
          changed_ = true;
          WithInsertPoint guard(n);
          n->output()->replaceAllUsesWith(
              graph_->insertConstant(static_cast<int64_t>(*maybe_len)));
        }
      }
      if (n->kind() == aten::__not__ &&
          n->inputs().at(0)->type()->cast<BoolType>()) {
        // __not__(inp) -> reverse refinements
        if (boolean_value_refinements_.count(n->input())) {
          auto& input_ref = boolean_value_refinements_[n->input()];
          boolean_value_refinements_[n->output()] = BooleanRefinementMapping(
              input_ref.false_refine(), input_ref.true_refine());
        }
      }
      if (n->matches("aten::eq(bool a, bool b) -> bool") ||
          (n->matches("aten::ne(bool a, bool b) -> bool"))) {
        for (size_t const_index : {0, 1}) {
          if (n->input(const_index)->node()->kind() != prim::Constant) {
            continue;
          }
          auto const_input = constant_as<bool>(n->input(const_index)).value();
          auto non_const_input = n->input(1 - const_index);
          if (!boolean_value_refinements_.count(non_const_input)) {
            continue;
          }
          // value == False / value != True -> equivalent to __not__ value
          // value == True / value != False -> equivalent to value
          auto& input_ref = boolean_value_refinements_[non_const_input];
          if ((!const_input && n->kind() == aten::eq) ||
              (const_input && n->kind() == aten::ne)) {
            boolean_value_refinements_[n->output()] = BooleanRefinementMapping(
                input_ref.false_refine(), input_ref.true_refine());
          } else {
            boolean_value_refinements_[n->output()] = BooleanRefinementMapping(
                input_ref.true_refine(), input_ref.false_refine());
          }
        }
      }

      if (n->kind() == prim::If) {
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
        bool true_block_throws = throwing_blocks_.count(if_n.thenBlock());
        bool false_block_throws = throwing_blocks_.count(if_n.elseBlock());

        // if one block throws, the refinements for the other block
        // become present in the current block, and all bool outputs
        // of the if node take their refinements from non throwing block
        // output

        if (true_block_throws || false_block_throws) {
          if (true_block_throws && false_block_throws) {
            throwing_blocks_.insert(b);
            continue;
          }
          if (true_block_throws) {
            block_refinements.insert(
                false_block_refinements.begin(), false_block_refinements.end());
          } else {
            block_refinements.insert(
                true_block_refinements.begin(), true_block_refinements.end());
          }
          Block* non_throwing_block =
              true_block_throws ? n->blocks().at(1) : n->blocks().at(0);
          for (size_t i = 0; i < if_n.outputs().size(); ++i) {
            if (boolean_value_refinements_.count(
                    non_throwing_block->outputs().at(i))) {
              boolean_value_refinements_[n->outputs().at(i)] =
                  boolean_value_refinements_[non_throwing_block->outputs().at(
                      i)];
            }
          }
          continue;
        }

        for (size_t i = 0; i < if_n.outputs().size(); ++i) {
          if (!(if_n.outputs().at(i)->type() == BoolType::get())) {
            continue;
          }
          Value* true_v = if_n.thenOutputs().at(i);
          Value* false_v = if_n.elseOutputs().at(i);

          if (!boolean_value_refinements_.count(true_v) &&
              !boolean_value_refinements_.count(false_v) &&
              !constant_as<bool>(true_v) && !constant_as<bool>(false_v)) {
            continue;
          }

          // if either block has a constant bool output, e.g. `true` on the
          // true block, then for the `false` value we can take the false
          // refinements present on the false block and from the other block
          // output value bc if the output is false it had to have come from the
          // false block. if len(a) == 5:
          //     x = len(b) == 4
          // else:
          //     x = False
          // if x is true, then we know both len(a) == 5 and len(b) == 4
          //
          // if neither block has a constant bool value, we just take the
          // intersection of the refinements from boolean outputs.
          // if cond:
          //    x = len(a) == 4 and len(b) == 5
          // else:
          //    x = len(a) == 4
          // here, we know if x is true, then len(a) == 4, but not len(b)
          // == 5, because that refinement is not present in the true block.
          // TODO: could also take intersection of refinements present in
          // both blocks, but it's not a real use case.

          // boolean_value_refinements_[value] is safe to access because
          // BooleanRefinementMapping has a default constructor

          BooleanRefinementMapping out;
          if (auto maybe_bool = constant_as<bool>(true_v)) {
            if (*maybe_bool) {
              out = BooleanRefinementMapping::FalseRefinements(unionRefinements(
                  boolean_value_refinements_[false_v].false_refine(),
                  false_block_refinements));
            } else {
              out = BooleanRefinementMapping::TrueRefinements(unionRefinements(
                  boolean_value_refinements_[false_v].true_refine(),
                  false_block_refinements));
            }
          } else if (auto maybe_bool = constant_as<bool>(false_v)) {
            if (*maybe_bool) {
              out = BooleanRefinementMapping::FalseRefinements(unionRefinements(
                  boolean_value_refinements_[true_v].false_refine(),
                  true_block_refinements));
            } else {
              out = BooleanRefinementMapping::TrueRefinements(unionRefinements(
                  boolean_value_refinements_[true_v].true_refine(),
                  true_block_refinements));
            }
          } else if (
              boolean_value_refinements_.count(true_v) &&
              boolean_value_refinements_.count(false_v)) {
            out = boolean_value_refinements_[true_v]
                      .intersectBooleanRefinementMapping(
                          boolean_value_refinements_[false_v]);
          }
          boolean_value_refinements_[if_n.outputs().at(i)] = out;
        }
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
        aliasDb_(torch::make_unique<AliasDb>(graph_)),
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
          !node->input(0)->type()->cast<ListType>()) {
        continue;
      }

      auto first_input = node->input(0);

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
          if (auto index = toIValue(node->input(1))) {
            size_t list_size = list_creation_node->inputs().size();
            if (auto norm_index = normalizeIndex(index->toInt(), list_size)) {
              node->output()->replaceAllUsesWith(
                  list_creation_node->input(*norm_index));
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
            node->output(i)->replaceAllUsesWith(list_creation_node->input(i));
            changed = true;
          }
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
