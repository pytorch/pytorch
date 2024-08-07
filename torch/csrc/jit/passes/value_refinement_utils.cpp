#include <c10/util/irange.h>
#include <torch/csrc/jit/passes/value_refinement_utils.h>

namespace torch::jit {

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

void joinIfRefinements(
    Node* if_node,
    std::unordered_set<Block*>& throwing_blocks,
    ListRefinement& curr_block_refinements,
    ListRefinement& true_block_refinements,
    ListRefinement& false_block_refinements,
    std::unordered_map<Value*, BooleanRefinementMapping>&
        boolean_value_refinements) {
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
        true_block_throws ? if_node->blocks().at(1) : if_node->blocks().at(0);
    for (const auto i : c10::irange(if_n.outputs().size())) {
      if (boolean_value_refinements.count(
              non_throwing_block->outputs().at(i))) {
        boolean_value_refinements[if_node->outputs().at(i)] =
            boolean_value_refinements[non_throwing_block->outputs().at(i)];
      }
    }
    return;
  }

  for (const auto i : c10::irange(if_n.outputs().size())) {
    if (!(if_n.outputs().at(i)->type() == BoolType::get())) {
      return;
    }
    Value* true_v = if_n.thenOutputs().at(i);
    Value* false_v = if_n.elseOutputs().at(i);

    if (!boolean_value_refinements.count(true_v) &&
        !boolean_value_refinements.count(false_v) &&
        !constant_as<bool>(true_v) && !constant_as<bool>(false_v)) {
      return;
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

    // boolean_value_refinements[value] is safe to access because
    // BooleanRefinementMapping has a default constructor

    BooleanRefinementMapping out;
    if (auto maybe_bool = constant_as<bool>(true_v)) {
      if (*maybe_bool) {
        out = BooleanRefinementMapping::FalseRefinements(unionRefinements(
            boolean_value_refinements[false_v].false_refine(),
            false_block_refinements));
      } else {
        out = BooleanRefinementMapping::TrueRefinements(unionRefinements(
            boolean_value_refinements[false_v].true_refine(),
            false_block_refinements));
      }
    } else if (auto maybe_bool = constant_as<bool>(false_v)) {
      if (*maybe_bool) {
        out = BooleanRefinementMapping::FalseRefinements(unionRefinements(
            boolean_value_refinements[true_v].false_refine(),
            true_block_refinements));
      } else {
        out = BooleanRefinementMapping::TrueRefinements(unionRefinements(
            boolean_value_refinements[true_v].true_refine(),
            true_block_refinements));
      }
    } else if (
        boolean_value_refinements.count(true_v) &&
        boolean_value_refinements.count(false_v)) {
      out = boolean_value_refinements[true_v].intersectBooleanRefinementMapping(
          boolean_value_refinements[false_v]);
    }
    boolean_value_refinements[if_n.outputs().at(i)] = out;
  }
}

bool handleCommonRefinentOperators(
    Node* n,
    std::unordered_set<Block*>& throwing_blocks,
    std::unordered_map<Value*, BooleanRefinementMapping>& info) {
  if (n->kind() == prim::RaiseException) {
    throwing_blocks.insert(n->owningBlock());
    return true;
  }
  if (n->kind() == aten::__not__ &&
      n->inputs().at(0)->type()->cast<BoolType>()) {
    // __not__(inp) -> reverse refinements
    if (info.count(n->input())) {
      auto& input_ref = info[n->input()];
      info[n->output()] = BooleanRefinementMapping(
          input_ref.false_refine(), input_ref.true_refine());
    }
    return true;
  }
  if (n->matches("aten::eq(bool a, bool b) -> bool") ||
      (n->matches("aten::ne(bool a, bool b) -> bool"))) {
    for (size_t const_index : {0, 1}) {
      if (n->input(const_index)->node()->kind() != prim::Constant) {
        continue;
      }
      auto const_input = constant_as<bool>(n->input(const_index)).value();
      auto non_const_input = n->input(1 - const_index);
      if (!info.count(non_const_input)) {
        continue;
      }
      // value == False / value != True -> equivalent to __not__ value
      // value == True / value != False -> equivalent to value
      auto& input_ref = info[non_const_input];
      if ((!const_input && n->kind() == aten::eq) ||
          (const_input && n->kind() == aten::ne)) {
        info[n->output()] = BooleanRefinementMapping(
            input_ref.false_refine(), input_ref.true_refine());
      } else {
        info[n->output()] = BooleanRefinementMapping(
            input_ref.true_refine(), input_ref.false_refine());
      }
    }
    return true;
  }
  return false;
}

} // namespace torch::jit
