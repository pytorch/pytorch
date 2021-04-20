#include <torch/csrc/jit/passes/value_refinement_utils.h>

namespace torch {
namespace jit {

// [value refinement algorithm]

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

// The intersection of the refinements is the Value* which are in both
// refinements and are refined to the same length
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
ListRefinement unionRefinements(
    const ListRefinement& ref1,
    const ListRefinement& ref2) {
  ListRefinement out;
  for (const auto& pair : ref1) {
    out[pair.first] = pair.second;
  }
  for (const auto& pair : ref2) {
    out[pair.first] = pair.second;
  }
  return out;
}

void joinIfRefinements(
    Node* if_node,
    std::unordered_set<Block*>& throwing_blocks,
    ListRefinement& curr_block_refinements,
    ListRefinement& true_block_refinements,
    ListRefinement& false_block_refinements,
    std::unordered_map<Value*, BoolRefinements>& info) {
  IfView if_n(if_node);
  Block* b = if_node->owningBlock();

  bool true_block_throws = throwing_blocks.count(if_n.thenBlock());
  bool false_block_throws = throwing_blocks.count(if_n.elseBlock());

  // if one block throws, the refinements for the other block
  // become present in the current block, and all bool outputs
  // of the if node take their refinements from non throwing block
  // output

  if (true_block_throws || false_block_throws) {
    if (true_block_throws && false_block_throws) {
      throwing_blocks.insert(b);
      return;
    }
    if (true_block_throws) {
      curr_block_refinements.insert(
          false_block_refinements.begin(), false_block_refinements.end());
    } else {
      curr_block_refinements.insert(
          true_block_refinements.begin(), true_block_refinements.end());
    }
    Block* non_throwing_block =
        true_block_throws ? if_n.elseBlock() : if_n.thenBlock();
    for (size_t i = 0; i < if_n.outputs().size(); ++i) {
      if (info.count(non_throwing_block->outputs().at(i))) {
        info[if_n.outputs().at(i)] = info[non_throwing_block->outputs().at(i)];
      }
    }
    return;
  }

  // if either block has a constant bool output, e.g. `true` on the
  // truee block, then for the `false` value we can take the false
  // refinements from the other block and from the other block value bc
  // if the output is false it had to have come from the false block.
  // Otherwise, just take intersection of refinements

  for (size_t i = 0; i < if_n.outputs().size(); ++i) {
    if (!(if_n.outputs().at(i)->type() == BoolType::get())) {
      continue;
    }
    Value* true_v = if_n.thenOutputs().at(i);
    Value* false_v = if_n.elseOutputs().at(i);

    if (!info.count(true_v) && !info.count(false_v)) {
      continue;
    }

    BoolRefinements out;
    if (auto maybe_bool = constant_as<bool>(true_v)) {
      if (*maybe_bool) {
        out = BoolRefinements::FalseRefinements(unionRefinements(
            info[false_v].false_refine(), false_block_refinements));
      } else {
        out = BoolRefinements::TrueRefinements(unionRefinements(
            info[false_v].true_refine(), false_block_refinements));
      }
    } else if (auto maybe_bool = constant_as<bool>(false_v)) {
      if (*maybe_bool) {
        out = BoolRefinements::FalseRefinements(unionRefinements(
            info[true_v].false_refine(), true_block_refinements));
      } else {
        out = BoolRefinements::TrueRefinements(unionRefinements(
            info[true_v].true_refine(), true_block_refinements));
      }
    }
    if (info.count(true_v) && info.count(false_v)) {
      out = info[true_v].intersectBoolRefinements(info[false_v]);
    }
    info[if_n.outputs().at(i)] = out;
  }
}

} // namespace jit
} // namespace torch