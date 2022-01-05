#pragma once

#include <ATen/core/jit_type.h>

#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/frontend/tree_views.h>

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/* =============================================== */
/*                  Program Status                 */
/* =============================================== */

enum NoneStatus { ALWAYS, MAYBE, NEVER };

NoneStatus canBeNone(Value* v);

enum class LoopStatus { NOT_IN_LOOP, IN_LOOP, IN_UNROLLED_LOOP };

struct WithLoopStatus {
  WithLoopStatus(LoopStatus* prev, LoopStatus new_status) {
    prev_value_ = *prev;
    prev_ptr_ = prev;
    *prev = new_status;
  }
  ~WithLoopStatus() {
    *prev_ptr_ = prev_value_;
  }

 private:
  LoopStatus* prev_ptr_;
  LoopStatus prev_value_;
};

/* ============================================= */
/*                    Getters                    */
/*     (Given X, return the corresponding Y)     */
/* ============================================= */

NodeKind getNodeKind(int kind, int ninputs);

NodeKind reverseComparision(NodeKind kind);

std::string getOperatorOverload(int kind, int ninputs);

// Get the appropriate builtin op for this augmented assignment
// If the RHS is a tensor, return the corresponding ATen in-place op
// If it's a list of scalars, then return the corresponding list augment op
Symbol getAugOp(const AugAssign& stmt, const TypePtr& type);

int64_t getAdjTupleIndex(
    const SourceRange& loc,
    const TupleTypePtr& tuple_type,
    int64_t input_index,
    bool allow_out_of_bounds);

int64_t getSliceInd(Value* idx_val, const SourceRange& loc);

// Get a pair of <in place magic method name, out of place magic method name>
// since the out of place method is called if the in place method is not
// present
std::pair<std::string, std::string> getAugMagicMethod(const AugAssign& stmt);

// we consider _N where N is a number, to be a non-meaningful name
// and do not record it as a unique name. This allows python printing to
// be able to export and import more consistently named graphs
bool meaningfulName(const std::string& name);

/* ======================================================== */
/*                      Verification                        */
/* ======================================================== */

void checkApplyNumInputs(Apply& apply, size_t expected_inputs);

void checkApplyNumInputsRange(
    Apply& apply,
    size_t min_expected_inputs,
    size_t max_expected_inputs);

bool isSupportedListElementType(const TypePtr& type);

// Validate that the `lhs` Expr's in an assignment statement are valid. That
// is:
//
// 1) All lhs Expr's are either Var, Tuple or Starred nodes
// 2) There is at most one Starred node in the lhs Expr
// 3) A Starred node can only appear when there is another non-Starred lhs
//    Expr. Concretely this means that `*abc = func()` is illegal. Unpacking
//    all outputs into a tuple is covered by `abc = func()`.
bool validateAssignLhsExpr(const List<Expr>& lhs, const SourceRange& r);

/* =============================================== */
/*                      Casts                      */
/* =============================================== */

Value* asSimple(const SugaredValuePtr& value);

std::shared_ptr<MagicMethod> makeMagic(
    const std::string& name,
    SugaredValuePtr base);

/* ========================================================== */
/*                 `__setstate__` Information                 */
/* ========================================================== */

// see [setstate type]
TypePtr getTypeForSetStateArg(const Def& def, const Self* self);

// see [setstate type]
bool shouldDeriveSetStateType(const Def& def, const FunctionSchema& schema);

/* ================================================== */
/*                 List/Dict Emission                 */
/* ================================================== */

template <class F1, class F2, class F3>
void refineAndSetUnionTypeHintOrPopulateCandidatesVector(
    const TypePtr& type_hint,
    TypePtr* refined_type_hint_ptr,
    std::vector<TypePtr>* all_candidates,
    const std::string& match_repr,
    const Expr& src,
    const F1& type_match,
    const F2& do_if_match,
    const F3& do_if_anytype,
    bool is_dict_constructor = false) {
  if (auto union_type_hint = (*refined_type_hint_ptr)->cast<UnionType>()) {
    // `candidate_types` holds all List types that were in the Union
    // annotation
    std::vector<TypePtr> candidate_types;

    std::copy_if(
        union_type_hint->containedTypes().begin(),
        union_type_hint->containedTypes().end(),
        std::back_inserter(candidate_types),
        [&](TypePtr type_ptr) { return type_match(type_ptr); });

    if (!is_dict_constructor && candidate_types.empty()) {
      throw ErrorReport(src)
          << "Expected an Union type annotation "
          << "with an inner " << match_repr << " type, but got "
          << (*refined_type_hint_ptr)->repr_str();
    } else if (candidate_types.size() == 1) {
      // The Union only had a single type of the container we want to
      // match, so we can unconditionally refine it to that type
      (*refined_type_hint_ptr) = candidate_types[0];
    } else {
      // We can't refine the Union yet, since it contains multiple
      // types of the container we want to match, but we do at least
      // have a list of possible types (e.g. `Union[List[int],
      // List[str], float, str]` -> candidates={List[int], List[str]})
      (*all_candidates) = std::move(candidate_types);
    }
  } else if (
      auto optional_type_hint =
          (*refined_type_hint_ptr)->cast<OptionalType>()) {
    (*refined_type_hint_ptr) = optional_type_hint->getElementType();
  }

  // This case handles code like `dict([(x, y), (a, b)])` that would
  // otherwise fail the following error checks
  if (is_dict_constructor) {
    return;
  }

  // If we had any annotation that was NOT a Union that can hold more
  // than one type of the container we want to match
  if (all_candidates->empty()) {
    if (type_match(*refined_type_hint_ptr)) {
      do_if_match();
    } else if ((*refined_type_hint_ptr)->kind() == AnyType::Kind) {
      do_if_anytype();
    } else {
      throw ErrorReport(src) << "Expected an annotation of type " << match_repr
                             << " but got " << type_hint->repr_str();
    }
  }
}

void refineAndSetListTypeHintFromCandidatesVector(
    const std::vector<TypePtr>& all_candidates,
    const TypePtr& type_hint,
    TypePtr* refined_type_hint_ptr,
    const TypePtr& unified_elem_type,
    const Expr& src);

void refineAndSetDictTypeHintFromCandidatesVector(
    const std::vector<TypePtr>& all_candidates,
    const TypePtr& type_hint,
    TypePtr* refined_type_hint_ptr,
    const TypePtr& known_key_type,
    const TypePtr& known_value_type,
    const Expr& src);

/* ================================================ */
/*                      Misc                        */
/* ================================================ */

template <class T, class Hash>
Value* materializeConstant(
    T val,
    Graph& graph,
    const SourceRange& r,
    std::unordered_map<T, Value*, Hash>& map) {
  auto existing_constant = map.find(val);
  if (existing_constant != map.end()) {
    return existing_constant->second;
  }
  WithInsertPoint guard(graph.block()->nodes().front());
  auto new_constant = graph.insertConstant(val, r);
  map[val] = new_constant;
  return new_constant;
}

} // namespace jit
} // namespace torch
