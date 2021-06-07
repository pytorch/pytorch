#pragma once

#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/frontend/canonicalize_modified_loop.h>
#include <torch/csrc/jit/frontend/convert_to_ssa.h>
#include <torch/csrc/jit/frontend/environment.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/run_cleanup_passes.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/frontend/type_refinement.h>

#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/passes/normalize_ops.h>

namespace torch {
namespace jit {

struct Environment;

// Information for each def being emitted.
// Defs can be nested to support closures so we need a stack of this information
// Currently records information about the functions return type.
struct DefContext {
  TypePtr declared_return_type_; // nullptr if not annotated
  TypePtr merged_return_type_; // nullptr if a Return has not been seen yet
};

// `to_ir` is the IR Emitter. Given the root of an AST (represented by
// `def`), we output an intermediate representation (the Graph). The
// main entry point of the program is `emitDef`
struct to_ir {
  to_ir(
      const Def& def,
      ResolverPtr resolver_,
      const Self* self,
      Function& method) // method being constructed
      : method(method),
        graph(method.graph()),
        resolver(std::move(resolver_)),
        typeParser_(resolver),
        environment_stack(nullptr) {
    AT_ASSERT(resolver);
    pushFrame(graph->block(), /*starts_def=*/true);

    // Type annotations exclude explicitly typing the "self" parameter, so in
    // the case that this is a method with self we expect one fewer parameter
    // annotation than the number of parameters this Def takes.
    if (self && def.decl().params().size() == 0) {
      throw ErrorReport(def.decl().params().range())
          << "methods must have a self argument";
    }
    method.setSchema(emitDef(def, self, graph->block()));

    // NB ORDERING: SSA conversion has to occur before
    // lifting of closures and forks, this way closures are converted
    // to SSA while part of their original graph, and closures are ready to
    // be inlined into forked closures
    convertToSSA(graph);
    // convert loops with an iter and body condition specified to
    // python-recognize while loops. we do this so they can be exported,
    // and run the pass early to avoid jitter. Like conversion to SSA,
    // it only needs to run once.
    canonicalizeModifiedLoops(graph);

    // Convert Ops to a Normalized Form
    normalizeOps(graph);

    runCleanupPasses(graph);
  }

 private:
  Function& method;
  std::shared_ptr<Graph> graph;
  ResolverPtr resolver;
  std::unordered_map<int64_t, Value*, std::hash<int64_t>> integral_constants;
  std::unordered_map<double, Value*, std::hash<double>> fp_constants;
  std::unordered_map<
      c10::complex<double>,
      Value*,
      c10::hash<c10::complex<double>>>
      complex_constants;
  std::unordered_set<Block*> exit_blocks;
  ScriptTypeParser typeParser_;
  LoopStatus loop_status_ = LoopStatus::NOT_IN_LOOP;
  std::vector<DefContext> def_stack_;
  size_t temp_name_count_ = 0;

  // Singly-linked list of environments. This top element contains a member
  // `next` that points to the most immediate enclosing scope's value.
  std::shared_ptr<Environment> environment_stack;

  // Internal utility methods
  std::string createTempName(const std::string& prefix);
  void checkBreakContinue(const SourceRange& loc, const std::string& stmt_name);

  // IR generation utility methods
  Node* create(Symbol kind, const SourceRange& loc, size_t n_outputs);
  void handleMaybeNoReturn(const Def& def, Block* block);
  void insertRefinements(const SourceRange& loc, const RefinementSet& ref);
  std::vector<NamedValue> emitAttributes(const List<Attribute>& attributes);

  // `Value` Getters
  std::vector<NamedValue> getNamedValues(
      const TreeList& trees,
      bool maybe_unpack);
  std::vector<NamedValue> getNamedValues(
      const List<Expr>& trees,
      bool maybe_unpack);
  std::vector<Value*> getValues(const TreeList& trees, bool maybe_unpack);
  std::vector<Value*> getValues(const List<Expr>& trees, bool maybe_unpack);

  // Environment stack manipulation
  void pushFrame(Block* b, bool starts_def = false);
  std::shared_ptr<Environment> popFrame(bool ends_def = false);

  // IR emission
  FunctionSchema emitDef(const Def& def, const Self* self, Block* block);
  std::vector<Argument> emitFormalArguments(
      const Def& def,
      const Self* self,
      const FunctionSchema& schema,
      Block* block);
  Argument emitOutput(
      const SourceRange& range,
      const FunctionSchema& schema,
      Block* block);
  void emitStatements(const List<Stmt>& statements);
  std::shared_ptr<ClosureValue> emitClosure(
      const std::function<void(Block*)>& emit_body);
  void emitClosure(const Def& def);
  void emitBreak(const Break& stmt);
  void emitContinue(const Continue& stmt);
  void emitDelete(const Delete& stmt);
  void emitReturn(const Return& stmt);
  void emitStatements(
      List<Stmt>::const_iterator begin,
      List<Stmt>::const_iterator end);
  CondValue emitCondExpr(const Expr& expr);
  std::shared_ptr<Environment> emitSingleIfBranch(
      Block* b,
      const List<Stmt>& branch,
      const RefinementSet& refinements);
  Value* emitTernaryIf(
      const TernaryIf& expr,
      const TypePtr& type_hint = nullptr);
  Value* emitListComprehension(const ListComp& lc, const TypePtr& type_hint);
  Value* emitDictComprehension(const DictComp& dc, const TypePtr& type_hint);

  CondValue emitShortCircuitLogical(
      const SourceRange& loc,
      const Expr& first_expr,
      const Expr& second_expr,
      bool is_or);
  Value* emitIfExpr(
      const SourceRange& range,
      const CondValue& cond_value,
      const std::function<Value*()>& true_expr,
      const std::function<Value*()>& false_expr);
  Value* emitToBool(const SourceRange& loc, Value* v);
  void emitIfElseBlocks(
      const SourceRange& loc,
      const CondValue& cond_value,
      const List<Stmt>& trueBranch,
      const List<Stmt>& falseBranch);
  CondValue emitHasAttr(const Expr& objExpr, const Expr& attrExpr);
  CondValue emitIsInstance(const Expr& obj, const Expr& classinfo);
  void emitIf(const If& stmt);
  void emitLoopCommon(
      const SourceRange& range,
      const std::function<void()>& emit_body,
      const SugaredValuePtr& iter_val,
      c10::optional<List<Expr>> targets,
      c10::optional<Expr> cond);
  void emitUnrolledLoop(
      const SourceRange& loc,
      const std::function<void()>& emit_body,
      const SugaredValuePtr& iterable,
      const List<Expr>& targets);
  void emitFor(
      const List<Expr>& targets,
      const List<Expr>& itrs,
      const SourceRange& loc,
      const std::function<void()>& emit_body);
  void emitFor(const For& stmt);
  void emitWhile(const While& stmt);
  void emitWith(const With& stmt);
  void emitRaise(const Raise& raise);
  void emitAssert(const Assert& stmt);
  void emitAugAssignment(const AugAssign& stmt);
  void emitAugAssignmentToSelectVar(const AugAssign& stmt);
  void emitAugAssignmentToVar(const AugAssign& stmt);
  Value* emitAugAssignmentHelper(const AugAssign& stmt, Value* lhs);
  void emitAugAssignmentGeneric(
      const AugAssign& stmt,
      const Subscript& lhs,
      Value* sliceable);
  void emitAugAssignmentToSubscript(const AugAssign& stmt);
  NamedValue emitValueToTensor(
      const NamedValue& value,
      const NamedValue& matchTypeOf);
  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const Expr& rhs);
  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const NamedValue& rhs);
  void emitTupleAssign(const TupleLiteral& tl, const Expr& rhs);
  void emitTupleAssign(
      const TupleLiteral& tl,
      const SugaredValuePtr& rhs_output,
      const SourceRange& rhs_loc,
      size_t n_binders,
      bool starred_unpack);
  void emitExprsAssign(
      const List<Expr>& lhs_exprs,
      const at::ArrayRef<SugaredValuePtr> outputs,
      const SourceRange& rhs_loc,
      size_t n_binders);
  void emitAssignment(const Assign& stmt);
  void emitSingleAssignment(const Assign& stmt);
  void emitSelectAssign(const Assign& stmt);
  void emitSelectAssign(
      const Expr& lhs,
      SugaredValuePtr rhs,
      const SourceRange& loc);
  std::shared_ptr<SugaredValue> emitApplyExpr(
      Apply& apply,
      size_t n_binders,
      const TypePtr& type_hint = nullptr);
  std::shared_ptr<SugaredValue> emitApplySpecialForm(
      Symbol form,
      Apply& apply,
      const TypePtr& type_hint = nullptr);
  std::shared_ptr<SugaredValue> emitApplySpecialFormForList(
      Apply& apply,
      const TypePtr& type_hint = nullptr);
  std::shared_ptr<SugaredValue> emitApplySpecialFormForDict(
      Apply& apply,
      const TypePtr& type_hint = nullptr);
  Value* emitExpr(const Expr& tree, const TypePtr& type_hint = nullptr);
  std::shared_ptr<SugaredValue> emitSugaredExpr(
      const Expr& tree,
      size_t n_binders,
      const TypePtr& type_hint = nullptr);
  Value* emitUnaryOp(
      const TreeRef& tree,
      const std::string& magicMethod,
      const c10::Symbol& opSymbol);
  std::shared_ptr<SugaredValue> emitForkExpr(
      SourceRange loc,
      const std::shared_ptr<SugaredValue>& forked,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs);
  std::shared_ptr<SugaredValue> emitRpcExpr(const Apply& apply, Symbol rpc_op);
  Value* emitBinaryOp(const TreeRef& tree);
  Value* emitSimpleExpr(
      const TreeRef& tree,
      const TypePtr& type_hint = nullptr);
  Value* emitConst(const Const& c);
  Value* emitStringLiteral(const StringLiteral& c);
  Value* emitSelect(
      const SourceRange& loc,
      Value* input,
      Value* dim,
      Value* index);
  Value* emitSliceOp(
      const SourceRange& loc,
      Value* sliceable,
      Value* dim,
      Value* start,
      Value* end,
      Value* step);
  Value* emitSlice(
      const SourceRange& loc,
      Value* input,
      Value* dim, // Only used for tensor slicing
      const SliceExpr& slice);
  Value* emitUnsqueeze(const SourceRange& loc, Value* input, Value* dim_val);
  Value* emitIndex(
      const SourceRange& loc,
      Value* input,
      at::ArrayRef<Value*> indices);
  std::pair<Value*, std::vector<Value*>> emitIntAndSliceIndexing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs);
  Value* emitMultidimSlicing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs);
  Value* emitBasicSlice(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs);
  Value* emitTupleIndex(
      const SourceRange& loc,
      Value* tuple_val,
      Value* idx_val);
  Value* emitTupleSlice(
      const SourceRange& loc,
      const NamedValue& tuple_val,
      const std::vector<at::optional<NamedValue>>& tuple_args);
  std::shared_ptr<SugaredValue> emitSubscript(
      const Subscript& subscript,
      TypePtr type_hint = nullptr);
};

} // namespace jit
} // namespace torch
