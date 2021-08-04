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
