#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/generated/aten_dispatch.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/parser.h"
#include "torch/csrc/utils/object_ptr.h"

#include "ATen/optional.h"

#include <climits>

namespace torch {
namespace jit {
namespace script {

using SugaredValuePtr = std::shared_ptr<SugaredValue>;
using FunctionTable = std::unordered_map<std::string, Method&>;
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using AttributeMap = std::unordered_map<std::string, Const>;
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

// Tuple of values. Used to implement tuple return values and unpacking
struct TupleValue : public SugaredValue {
  TupleValue() {}
  TupleValue(std::vector<std::shared_ptr<SugaredValue>> values) : values(std::move(values)) {}

  virtual std::string kind() const override {
    return "tuple";
  }

  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m)
      override {
    return values;
  }

 private:
  std::vector<std::shared_ptr<SugaredValue>> values;
};

// Auxiliary data structure for desugaring variable binding into our always
// explicitly scoped language as we descend down
// nested control structures in the frontend (which themselves don't introduce
// scopes)
//
// The algorithm is roughly as follows:
// 1) While emitting a block within a control operator, add inputs and outputs
//      from the block for each value referenced (both "reads" and "writes").
//      This sets the value up as a candidate loop carried dependency.
// 2) When we reach the end of the block, examine all the values in the current
//      scope's value map. If the name also resides in an outer scope with a
//      different Value*, this is a true loop-carried dependency. If not, this
//      value was not assigned to. Replace all references to the block input
//      with the Value* pointed to in the tightest enclosing scope. Then delete
//      that block input and output.
// 3) When we emit the actual control operator, take all of the loop-carried
//      dependency values as inputs and return them as outputs from the control
//      op
//
//  Note that an alternative implementation could only add the loop-carried dep
//      inputs and outputs when we see a value that is mutated. This, however
//      requires replacing all references to that value *within the current
//      block* with a new input. That is to say: we need to traverse the pre-
//      decessor nodes and replace inputs that reference that value with the
//      newly-created input. This could be made less expensive with a change to
//      the IR API, but for now we choose to pessimisitically create inputs and
//      delete unnecessary ones later with replaceAllusesWith().
struct Environment {
  Environment(Method & method, const Resolver& resolver, Block* b, std::shared_ptr<Environment> next = nullptr)
      : method(method), resolver(resolver), b(b), next(next) {}

  Method & method;
  const Resolver& resolver;
  std::vector<std::string> captured_inputs;
  Block* b;

  std::shared_ptr<Environment> next;

  SugaredValuePtr findInThisFrame(const std::string& name) {
    if (value_table.count(name)) {
      return value_table.at(name);
    }
    return nullptr;
  }

  SugaredValuePtr findInParentFrame(const std::string& name) {
    for (auto runner = next; runner; runner = runner->next) {
      if (runner->value_table.count(name)) {
        return runner->value_table.at(name);
      }
    }
    return nullptr;
  }

  Value* getValueInThisFrame(const SourceRange& loc, const std::string& name) {
    return value_table.at(name)->asValue(loc, method);
  }

  SugaredValuePtr createCapturedInput(const std::string& name) {
    // Create the input
    Value* new_input = b->addInput();

    // Associate this name with this value
    auto sv = std::make_shared<SimpleValue>(new_input);
    value_table[name] = sv;

    // List as a positional input
    captured_inputs.push_back(name);

    return sv;
  }
  Block* block() {
    return b;
  }
  Symbol getBlockOwningKind() {
    Symbol owning_kind = Symbol();
    if (b->owningNode()) {
      owning_kind = b->owningNode()->kind();
    }
    return owning_kind;
  }

  void setVar(const SourceRange& loc, const std::string& name, Value* value) {
    setSugaredVar(loc, name, std::make_shared<SimpleValue>(value));
  }
  static bool isSimple(SugaredValuePtr value) {
    return dynamic_cast<SimpleValue*>(value.get()) != nullptr;
  }

  void setSugaredVar(const SourceRange& loc, const std::string& name, SugaredValuePtr value) {
    bool is_simple_value = isSimple(value);
    // prevent re-assignment involving any sugared values
    // any reassignment like:
    // a = ...
    // while ...
    //   a = ..
    // requires 'a' to be first-class in the graph since its value depends on
    // control flow
    if(auto parent = findInParentFrame(name)) {
      if(!is_simple_value) {
        throw ErrorReport(loc) << "cannot re-assign '" << name << "' to a value of type " << value->kind()
        << ". Only reassignments to first-class values are allowed";
      }
      if(!isSimple(parent)) {
        throw ErrorReport(loc) << "cannot re-assign '" << name << "' because it has type " << value->kind()
        << ". Only reassignments to first-class values are allowed";
      }
    }
    if (is_simple_value &&
        !findInThisFrame(name) &&
        findInParentFrame(name) &&
        getBlockOwningKind() == prim::Loop) {
      createCapturedInput(name);
    }
    value_table[name] = std::move(value);
  }

  SugaredValuePtr getSugaredVar(const Ident& ident, bool required=true) {
    return getSugaredVar(ident.name(), ident.range());
  }
  Value* getVar(const Ident& ident) {
    return getSugaredVar(ident)->asValue(ident.range(), method);
  }

  SugaredValuePtr getSugaredVar(const std::string& ident, SourceRange range, bool required=true) {
    auto retval = findInThisFrame(ident);

    if (!retval && (retval = findInParentFrame(ident)) &&
        getBlockOwningKind() == prim::Loop) {
      retval = createCapturedInput(ident);
    }

    if(!retval) {
      retval = resolver(ident);
    }

    if (!retval && required) {
      throw ErrorReport(range) << "undefined value " << ident;
    }
    return retval;
  }

  Value* getVar(const std::string& ident, SourceRange range) {
    return getSugaredVar(ident, range)->asValue(range, method);
  }

  // Given that after emitting statements in a block, we've added block inputs
  // for all value references and assignments, delete inputs for which there was
  // no assignment, only references.
  void deleteExtraInputs(const SourceRange& loc, size_t skip_num = 0) {
    std::vector<size_t> inputs_to_delete;
    int i = skip_num;
    for (const auto& x : captured_inputs) {
      if (b->inputs()[i] == getValueInThisFrame(loc, x)) {
        inputs_to_delete.push_back(i);
      }
      i++;
    }

    for (auto ritr = inputs_to_delete.rbegin(); ritr != inputs_to_delete.rend();
         ++ritr) {
      auto name = captured_inputs[*ritr - skip_num];
      Value* v = getValueInThisFrame(loc, name);
      Value* orig = findInParentFrame(name)->asValue(loc, method);
      // Replace all matching node inputs with original value
      // from an enclosing scope
      v->replaceAllUsesWith(orig);

      // Actually remove the input
      b->eraseInput(*ritr);
      captured_inputs.erase(captured_inputs.begin() + *ritr - skip_num);
    }
  }
  std::vector<std::string> definedVariables() {
    std::vector<std::string> result;
    for(auto & kv : value_table) {
      result.push_back(kv.first);
    }
    return result;
  }
private:
  ValueTable value_table;
};

Const getAttributeValue(Expr value_expr) {
  switch (value_expr.kind()) {
    case TK_CONST: {
      return Const(value_expr);
    } break;
    case TK_TRUE: {
      return Const::create(value_expr.range(), "1");
    } break;
    case TK_FALSE: {
      return Const::create(value_expr.range(), "0");
    } break;
    default:
      throw ErrorReport(value_expr) << "attributes must be constants, or a list of constants";
      break;
  }
}

std::shared_ptr<SugaredValue> packOutputs(at::ArrayRef<Value*> values) {
  if(values.size() == 1) {
    return std::make_shared<SimpleValue>(values[0]);
  }
  auto svalues = fmap(values, [](Value* v) -> std::shared_ptr<SugaredValue> {
    return std::make_shared<SimpleValue>(v);
  });
  return std::make_shared<TupleValue>(std::move(svalues));
}

std::shared_ptr<SugaredValue> emitBuiltinCall(
  const SourceRange& loc,
  Method& method,
  const std::string & name,
  at::ArrayRef<Value*> inputs,
  List<Attribute> attributes,
  // if true, emitBuiltinCall will throw an exception if this builtin does not exist,
  // otherwise it will return nullptr if the builtin is not found.
  bool required) {

  NodeKind kind(Symbol::aten(name)); // TODO: this is a guess; could it be jit?
  auto graph = method.graph();
  auto n = graph->insertNode(graph->create(kind, inputs, 0))
                ->setSourceLocation(std::make_shared<SourceRange>(loc));

  for (const auto& attr : attributes) {
    const auto& name = Symbol::attr(attr.name().name());
    const Expr& value_expr = attr.value();
    if(value_expr.kind() == TK_LIST_LITERAL) {
      auto value_list = ListLiteral(value_expr).inputs();
      std::vector<Const> values = fmap(value_list, getAttributeValue);
      bool is_float = std::any_of(values.begin(), values.end(),
                                  [](const Const& c) { return c.isFloatingPoint(); });
      if (is_float) {
        n->fs_(name, fmap(values, [](const Const& c) { return c.asFloatingPoint(); }));
      } else {
        n->is_(name, fmap(values, [](const Const& c) { return c.asIntegral(); }));
      }
    } else {
      auto value = getAttributeValue(value_expr);
      if (value.isFloatingPoint()) {
        n->f_(name, value.asFloatingPoint());
      } else {
        n->i_(name, value.asIntegral());
      }
    }
  }
  auto op = findTensorOp(n);
  if(!op) {
    n->destroy();
    if(!required)
      return nullptr;
    throw ErrorReport(loc) << "unknown builtin op";
  }
  if(op->num_outputs == UNKNOWN_OUTPUTS) {
    throw ErrorReport(loc) << "produces an unknown number of outputs, so it cannot be used directly from script methods";
  }
  for(size_t i = 0; i < op->num_outputs; ++i)
    n->addOutput();

  return packOutputs(n->outputs());
}


std::shared_ptr<SugaredValue> BuiltinFunction::call(
    SourceRange loc,
    Method & m,
    at::ArrayRef<Value*> inputs_,
    List<Attribute> attributes,
    size_t n_binders) {
  std::vector<Value*> inputs;
  if (value)
    inputs.push_back(value);
  inputs.insert(inputs.end(), inputs_.begin(), inputs_.end());
  return emitBuiltinCall(loc, m, name, inputs, attributes, true);
}

struct to_ir {
  to_ir(
      Def def,
      FunctionTable& function_table,
      const Resolver& resolver,
      SugaredValuePtr self,
      Method& method) // method being constructed
      : method(method)
      , graph(method.graph())
      , def(def)
      , function_table(function_table)
      , resolver(resolver)
      , environment_stack(nullptr) {
    pushFrame(graph->block());
    // inputs
    auto it = def.params().begin();
    auto end = def.params().end();
    if(self) {
      if(it == end)
        throw ErrorReport(def.params().range()) << "methods must have a self argument";
      environment_stack->setSugaredVar(def.range(), (*it).ident().name(), self);
      ++it;
    }
    for(;it != end; ++it) {
      auto& name = (*it).ident().name();
      environment_stack->setVar((*it).ident().range(), name, graph->addInput(name));
    }
    // body
    auto stmts = def.statements();
    auto stmts_begin = stmts.begin();
    auto stmts_end = stmts.end();
    bool has_return = false;
    if (stmts_begin != stmts_end && (*std::prev(stmts_end)).kind() == TK_RETURN) {
      --stmts_end;
      has_return = true;
    }

    emitStatements(stmts_begin, stmts_end);

    // outputs
    if (has_return) {
      auto results = getValues(Return(*stmts_end).values(), true);
      for(auto r : results) {
        graph->registerOutput(r);
      }
    }
  }

private:
  Method& method;
  std::shared_ptr<Graph> graph;
  Def def;
  FunctionTable& function_table;
  const Resolver& resolver;

  // Singly-linked list of environments. This top element contains a member
  // `next` that points to the most immediate enclosing scope's value.
  std::shared_ptr<Environment> environment_stack;

  void pushFrame(Block * b) {
    environment_stack = std::make_shared<Environment>(method, resolver, b, environment_stack);
  }
  std::shared_ptr<Environment> popFrame() {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    return old_frame;
  }
  void emitStatements(const List<Stmt>& statements) {
    return emitStatements(statements.begin(), statements.end());
  }
  void emitStatements(List<Stmt>::const_iterator begin, List<Stmt>::const_iterator end) {
    for (; begin != end; ++begin) {
      auto stmt = *begin;
      switch (stmt.kind()) {
        case TK_IF:
          emitIf(If(stmt));
          break;
        case TK_WHILE:
          emitWhile(While(stmt));
          break;
        case TK_FOR:
          emitFor(For(stmt));
          break;
        case TK_ASSIGN:
          emitAssignment(Assign(stmt));
          break;
        case TK_GLOBAL:
          for (auto ident : Global(stmt).names()) {
            const auto& name = Ident(ident).name();
            environment_stack->setVar(ident.range(), name, graph->addInput(name));
          }
          break;
        case TK_EXPR_STMT: {
          auto exprs = ExprStmt(stmt).exprs();
          for (const auto& expr : exprs) {
            emitSugaredExpr(expr, 0);
          }
        }
        break;
        case TK_RETURN:
          throw ErrorReport(stmt) << "return statements can appear only at the end "
                                  << "of the function body";
          break;
      }
    }
  }

  std::shared_ptr<Environment> emitSingleIfBranch(
      Block* b,
      const List<Stmt> branch,
      std::unordered_set<std::string>* mutated_parent_values) {
    pushFrame(b);
    WithInsertPoint guard(b);
    emitStatements(branch);

    for (const auto & n : environment_stack->definedVariables()) {
      if (environment_stack->findInParentFrame(n)) {
        mutated_parent_values->insert(n);
      }
    }
    return popFrame();
  }

  Node* create(Symbol kind, const SourceRange& loc,  size_t n_outputs) {
    return graph
             ->create(kind, n_outputs)
             ->setSourceLocation(std::make_shared<SourceRange>(loc));
  }

  Value* emitTernaryIf(const TernaryIf& expr) {
    Value* cond_value = emitExpr(expr.cond());

    Node* n = graph->insertNode(create(prim::If, expr.range(), 0));
    n->addInput(cond_value);
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    auto emit_if_expr = [this](Block* b, const Expr& expr) {
      pushFrame(b);
      WithInsertPoint guard(b);
      Value* out_val = emitExpr(expr);
      b->registerOutput(out_val);
      popFrame();
    };

    emit_if_expr(true_block, expr.true_expr());
    emit_if_expr(false_block, expr.false_expr());

    // Add op outputs
    auto expr_value = n->addOutput(); // Resulting value

    return expr_value;
  }

  void emitIf(const If& stmt) {
    Value* cond_value = emitExpr(stmt.cond());

    Node* n = graph->insertNode(create(prim::If, stmt.range(), 0));
    n->addInput(cond_value);
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    // Emit both blocks once to get the union of all mutated values
    std::unordered_set<std::string> mutated_parent_values;
    auto save_true = emitSingleIfBranch(
        true_block, stmt.trueBranch(), &mutated_parent_values);
    auto save_false = emitSingleIfBranch(
        false_block, stmt.falseBranch(), &mutated_parent_values);

    std::vector<std::string> sorted_mutations(
        mutated_parent_values.begin(), mutated_parent_values.end());
    std::sort(sorted_mutations.begin(), sorted_mutations.end());

    // Register outputs in each block
    for (const auto& x : sorted_mutations) {
      true_block->registerOutput(save_true->getVar(x, stmt.range()));
    }
    for (const auto& x : sorted_mutations) {
      false_block->registerOutput(save_false->getVar(x, stmt.range()));
    }

    // Add op outputs
    for (const auto& x : sorted_mutations) {
      environment_stack->setVar(stmt.range(), x, n->addOutput());
    }
  }

  // *********************** Loop Operators ************************************
  // Emits a loop operators conforming to the semantics specified at
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#experimental-loop
  // TODO: implement scan_outputs

  // the format of the Loop instruction is:
  // loop_carried_outputs* = Loop(max_trip_count, start_condition,
  // loop_carried_inputs*)
  //                          block0(loop_counter, loop_carried_block*) {
  //                             <body>
  //                             -> (continue_condition,
  //                             loop_carried_block_outputs*)
  //                          }
  // all loop_carried_... lists are the same length and represent the value of
  // loop-carried variables whose definitions are updated as the loop executes
  // in a way that ensure single static assignment.

  void emitLoopCommon(
      SourceRange range,
      at::optional<Expr> max_trip_count,
      at::optional<Expr> cond,
      const List<Stmt>& body,
      at::optional<Ident> itr_ident) {
    Node* n = graph->insertNode(create(prim::Loop, range, 0));
    Value *max_trip_count_val, *cond_val;
    {
      WithInsertPoint guard(n);
      if (max_trip_count) {
        max_trip_count_val = emitExpr(max_trip_count.value());
      } else {
        max_trip_count_val =
            emitConst(Const::create(range, std::to_string(INT_MAX)));
      }
      if (cond) {
        cond_val = emitExpr(cond.value());
      } else {
        cond_val = emitBooleanConst(range, true);
      }
    }
    n->addInput(max_trip_count_val);
    n->addInput(cond_val);
    auto* body_block = n->addBlock();
    Value* trip_count = body_block->addInput(); // Iteration num
    size_t skip_inputs_num = 1;

    {
      pushFrame(body_block);
      if (itr_ident) {
        environment_stack->setVar(itr_ident->range(), itr_ident->name(), trip_count);
      }
      WithInsertPoint guard(body_block);
      emitStatements(body);

      // Also emit the conditional
      if (cond) {
        Value* body_cond_value = emitExpr(cond.value());
        body_block->registerOutput(body_cond_value);
      } else {
        Value* cond_value_dummy = emitBooleanConst(range, true);
        body_block->registerOutput(cond_value_dummy);
      }

      auto body_frame = popFrame();
      auto outer_frame = environment_stack;
      // Remove inputs for values that did not mutate within the
      // block
      body_frame->deleteExtraInputs(range, skip_inputs_num);

      // Add block outputs
      for (const auto& x : body_frame->captured_inputs) {
        body_block->registerOutput(body_frame->getValueInThisFrame(range, x));
        n->addInput(outer_frame->getVar(x, range));
        outer_frame->setVar(range, x, n->addOutput());
      }

    }
  }

  void emitForRange(SourceRange range, const Ident& target, const List<Expr>& args, const List<Stmt>& body) {
    // TODO: start, stop, step loop
    if (args.size() != 1) {
      throw ErrorReport(range)
          << "range() expects one argument but got" << args.size();
    }
    emitLoopCommon(range, {args[0]}, {}, body, target);
  }

  void emitFor(const For& stmt) {
    // For now, we only support range loops. e.g. for i in range(3): ...
    auto targets = stmt.targets();
    auto itrs = stmt.itrs();
    auto body = stmt.body();

    if (stmt.itrs().size() != 1) {
      throw ErrorReport(stmt)
          << "List of iterables is not supported currently.";
    }
    if (targets.size() != 1) {
      throw ErrorReport(stmt) << "Iteration variable unpacking is not supported";
    }

    if (targets[0].kind() != TK_VAR) {
      throw ErrorReport(targets[0]) << "Starred unpacking is currently not"
          << " supported for for loops.";
    }
    auto target = Var(targets[0]).name();

    // match range(<expr>) style loops
    // itrs must consist of a single Apply node
    if (itrs[0].kind() == TK_APPLY) {
      Apply range_iterator = Apply(itrs[0]);
      if (range_iterator.callee().kind() == TK_VAR) {
        Var var = Var(range_iterator.callee());
        if (var.name().name() == "range") {
          return emitForRange(stmt.range(), target, range_iterator.inputs(), body);
        }
      }
    }

    // it isn't a range(<expr>) loop, treat it as a sugared value that maybe can be
    // unrolled
    auto sv = emitSugaredExpr(itrs[0], 1);
    auto instances = sv->asTuple(stmt.range(), method);
    const std::string& target_name = target.name();
    pushFrame(environment_stack->block());
    for(auto inst : instances) {
      environment_stack->setSugaredVar(itrs[0].range(), target_name, inst);
      emitStatements(body);
    }

    for (const auto & n : environment_stack->definedVariables()) {
      if (environment_stack->findInParentFrame(n)) {
        environment_stack->next->setVar(stmt.range(), n, environment_stack->getVar(n, stmt.range()));
      }
    }
    popFrame();
  }

  void emitWhile(const While& stmt) {
    auto cond = stmt.cond();
    emitLoopCommon(stmt.range(), {}, {cond}, stmt.body(), {});
  }

  // Validate that the `lhs` Expr's in an assignment statement are valid. That
  // is:
  //
  // 1) All lhs Expr's are either Var or Starred nodes
  // 2) There is at most one Starred node in the lhs Expr
  // 3) A Starred node can only appear when there is another non-Starred lhs Expr
  //    Concretely this means that `*abc = func()` is illegal. Unpacking all
  //    outputs into a tuple is covered by `abc = func()`.
  bool calcNumStarredUnpack(const List<Expr>& lhs, const SourceRange& r) {
    size_t num_normal_assign = 0;
    size_t num_starred = 0;
    for (const auto& assignee : lhs) {
      if (assignee.kind() == TK_VAR) {
        num_normal_assign++;
      } else if (assignee.kind() == TK_STARRED) {
        num_starred++;
      } else {
        throw ErrorReport(assignee)
            << "lhs of assignment must be a variable or starred expression.";
      }
    }

    if (num_starred > 1) {
      throw ErrorReport(r)
          << "Only one starred expression is allowed on the lhs.";
    }

    if (num_starred > 0 && num_normal_assign == 0) {
      throw ErrorReport(r) << "A Starred expression may only appear on the "
                              << "lhs within the presence of another non-starred"
                              << " expression.";
    }

    return num_starred;
  }

  void emitAssignment(const Assign& stmt) {
    bool starred_unpack = calcNumStarredUnpack(stmt.lhs(), stmt.range());
    if (stmt.reduction() != '=') {
      if (stmt.lhs().size() != 1) {
        throw ErrorReport(stmt)
            << "reductions are only allowed when there is a single variable "
            << "on the left-hand side.";
      }
      Ident lhs = Var(stmt.lhs()[0]).name();
      Expr expr = BinOp::create(stmt.range(), stmt.reduction(),
                                Var::create(lhs.range(), lhs), stmt.rhs());
      environment_stack->setVar(lhs.range(), lhs.name(), emitExpr(expr));
      return;
    }
    // how many non-starred things are on the lhs?
    // a = ... (1)
    // b, *c = ... (1)
    // a, b = ... (2)
    size_t n_binders = stmt.lhs().size();
    if(starred_unpack)
      n_binders--;

    auto output = emitSugaredExpr(stmt.rhs(), n_binders);

    if(stmt.lhs().size() == 1) {
      JIT_ASSERT(!starred_unpack);
      auto v = Var(stmt.lhs()[0]);
      environment_stack->setSugaredVar(v.range(), v.name().name(), output);
      return;
    }

    auto outputs = output->asTuple(stmt.rhs().range(), method);
    if(outputs.size() < n_binders) {
      throw ErrorReport(stmt)
        << "need " << (starred_unpack ? "at least " : "")
        << n_binders << " values to unpack but found only "
        << outputs.size();
    }
    if(outputs.size() > n_binders && !starred_unpack) {
      throw ErrorReport(stmt)
      << "too many values to unpack, need " << n_binders << " but found "
      << outputs.size();
    }
    int i = 0;
    for (auto assignee : stmt.lhs()) {
      if (assignee.kind() == TK_VAR) {
        Value * value = outputs.at(i)->asValue(assignee.range(), method);
        environment_stack->setVar(assignee.range(), Var(assignee).name().name(), value);
        i++;
      } else if (assignee.kind() == TK_STARRED) {
        auto var = Starred(assignee).expr();
        if (var.kind() != TK_VAR) {
          throw ErrorReport(var) << "Cannot pack a tuple into a non-variable.";
        }
        size_t n_matched = outputs.size() - n_binders;
        ArrayRef<std::shared_ptr<SugaredValue>> outputs_ref = outputs;
        SugaredValuePtr tup = std::make_shared<TupleValue>(outputs_ref.slice(i, n_matched));
        environment_stack->setSugaredVar(
          var.range(), Var(var).name().name(), tup);
        i += n_matched;
      }
    }
  }

  NodeKind getNodeKind(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return aten::add;
      case '-':
        return aten::sub;
      case TK_UNARY_MINUS:
        return aten::neg;
      case '*':
        return aten::mul;
      case TK_STARRED:
        return prim::Starred;
      case '/':
        return aten::div;
      case TK_NE:
        return aten::ne;
      case TK_EQ:
        return aten::eq;
      case '<':
        return aten::lt;
      case '>':
        return aten::gt;
      case TK_LE:
        return aten::le;
      case TK_GE:
        return aten::ge;
      case TK_AND:
        return aten::__and__;
      case TK_OR:
        return aten::__or__;
      case TK_NOT:
        return aten::__not__;
      default:
        throw std::runtime_error("unknown kind " + std::to_string(kind));
    }
  }

  std::vector<Value*> getValues(TreeList trees, bool maybe_unpack=false) {
    std::vector<Value*> values;
    for (const auto& tree : trees) {
      if(maybe_unpack && tree->kind() == TK_STARRED) {
        auto starred = Starred(tree);
        auto entries = emitSugaredExpr(starred.expr(), 1)->asTuple(starred.range(), method);
        for(auto entry : entries) {
          values.push_back(entry->asValue(starred.range(), method));
        }
      } else {
        values.push_back(emitExpr(Expr(tree)));
      }
    }
    return values;
  }
  std::vector<Value*> getValues(List<Expr> trees, bool maybe_unpack=false) {
    return getValues(trees.tree()->trees(), maybe_unpack);
  }

  // special rules apply when we directly call foo(a,b) when foo is an ident
  std::shared_ptr<SugaredValue> emitApplyIdent(Ident ident, std::vector<Value*> inputs, List<Attribute> attributes, size_t n_binders) {
    auto it = function_table.find(ident.name());
    if (it != function_table.end()) {
      if(inputs.size() != it->second.num_inputs())
        throw ErrorReport(ident) << "expected " << it->second.num_inputs() << " but found " << inputs.size();
      return packOutputs(method.emit_call_to(it->second, inputs));
    } else if (ident.name() == "print") {
      if (!attributes.empty())
        throw ErrorReport(ident) << "print doesn't accept any keyword arguments";
      emitNode(prim::Print, ident.range(), inputs, 0);
      return std::make_shared<TupleValue>();
    }
    if(auto result = emitBuiltinCall(ident.range(), method, ident.name(), inputs, attributes, false)) {
      return result;
    }
    // it wasn't known built in, so treat it like standard apply
    return emitApplyExpr(Var::create(ident.range(), ident), inputs, attributes, n_binders);
  }

  std::shared_ptr<SugaredValue> emitApplyExpr(Expr callee, const std::vector<Value*>& inputs, List<Attribute> attributes, size_t n_binders) {
    // otherwise we evaluate the callee and then desugar it
    auto sv = emitSugaredExpr(callee, 1);
    return sv->call(callee.range(), method, inputs, attributes, n_binders);
  }

  Value* emitExpr(Expr tree) {
    return emitSugaredExpr(tree, 1)->asValue(tree.range(), method);
  }

  // any expression that can produce a SugaredValue is handled here
  // expressions that only return a single Value* are handled in emitSimpleExpr
  std::shared_ptr<SugaredValue> emitSugaredExpr(Expr tree, size_t n_binders) {
    switch(tree.kind()) {
      case TK_VAR:
        return environment_stack->getSugaredVar(Var(tree).name());
      case '.': {
        auto select = Select(tree);
        auto sv = emitSugaredExpr(select.value(), 1);
        return sv->attr(select.range(), method, select.selector().name());
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        auto inputs = getValues(apply.inputs(), true);
        // the apply is directly an identifier 'foo'
        if(apply.callee().kind() == TK_VAR) {
          return emitApplyIdent(Var(apply.callee()).name(), inputs, apply.attributes(), n_binders);
        }
        return emitApplyExpr(apply.callee(), inputs, apply.attributes(), n_binders);
      } break;
      default:
        return std::make_shared<SimpleValue>(emitSimpleExpr(tree));
    }
  }

  Value* emitSimpleExpr(
      const TreeRef& tree) {
    switch (tree->kind()) {
      case TK_NE:
      case TK_EQ:
      case '<':
      case '>':
      case TK_LE:
      case TK_GE:
      case '*':
      case '/':
      case TK_AND:
      case TK_OR:
      case TK_NOT:
      case TK_UNARY_MINUS: {
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        return emitNode(kind, tree->range(), getValues(inputs), 1)->output();
      } break;
      case '+':
      case '-': {
        const auto& inputs =tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        auto* node = emitNode(kind, tree->range(), getValues(inputs), 1);
        node->t_(Symbol::attr("alpha"), at::CPU(at::kFloat).scalarTensor(1.0));
        return node->output();
      }
      case TK_STARRED: {
        throw ErrorReport(tree) << "Unexpected starred expansion. File a bug report.";
      }
      case TK_CAST: {
        const auto cast = Cast(tree);
        return emitCast(cast.input(), cast.type());
      } break;
      case TK_CONST: {
        return emitConst(Const(tree));
      } break;
      case TK_TRUE: {
        return emitBooleanConst(tree->range(), true);
      } break;
      case TK_FALSE: {
        return emitBooleanConst(tree->range(), false);
      } break;
      case TK_SLICE: {
        const auto slice = Slice(tree);
        return emitSlice(
            slice.range(),
            {slice.value(), slice.startOr(0), slice.endOr(-1)});
      } break;
      case TK_GATHER: {
        const auto gather = Gather(tree);
        return emitGather(
            gather.range(), {gather.value(), gather.indices()});
      } break;
      case TK_IF_EXPR: {
        return emitTernaryIf(TernaryIf(tree));
      } break;
      default:
        throw ErrorReport(tree) << "NYI: " << tree;
        break;
    }
  }

  Value* emitCast(Expr input, const ScalarType& type) {
    at::ScalarType t;
    switch (type.kind()) {
      case TK_INT:
        t = at::kInt;
        break;
      case TK_FLOAT:
        t = at::kFloat;
        break;
      case TK_LONG:
        t = at::kLong;
        break;
      case TK_BOOL:
        t = at::kByte;
        break;
      default:
        throw ErrorReport(input) << "Unrecognized type: " << type;
    }
    return emitNode(
               Symbol::aten("type_as"),
               input.range(),
               {emitExpr(input), createConstant(input.range(), at::ones(at::CPU(t), {1}))},
               1)
        ->output();
  }

  Value* emitBooleanConst(SourceRange range, bool val) {
    return createConstant(range, at::CPU(at::kByte).scalarTensor(val));
  }

  Value* emitConst(const Const& c) {
    if (c.isFloatingPoint()) {
      return createConstant(c.range(), at::CPU(at::kFloat).scalarTensor(c.asFloatingPoint()));
    } else {
      return createConstant(c.range(), at::CPU(at::kLong).scalarTensor(c.asIntegral()));
    }
  }

  Node* emitNode(
      NodeKind kind,
      const SourceRange& loc,
      const std::vector<Value*> inputs,
      size_t n_outputs) {
    Node* n = graph->insertNode(create(kind, loc, n_outputs));
    for (auto* input_value : inputs) {
      n->addInput(input_value);
    }
    return n;
  }

  // Desugars slice syntactic sugar tensor[begin:end] -> tensor.slice(begin,
  // end).
  Value* emitSlice(
      const SourceRange& loc,
      TreeList&& inputs) {
    const auto applyInputs =
        Compound::create(TK_LIST, loc, std::move(inputs));
    const auto input_values = getValues(applyInputs->trees());
    Value* tensor = input_values[0];
    const auto& begin = at::Scalar(input_values[1]->node()->t(attr::value)).toInt();
    const auto& end = at::Scalar(input_values[2]->node()->t(attr::value)).toInt();
    return emitNode(
               Symbol::aten("slice"),
               loc,
               {tensor},
               1)
               ->i_(attr::dim, 0)
               ->i_(attr::step, 1)
               ->i_(attr::start, begin)
               ->i_(attr::end, end)->output();
  }

  // Desugars gather syntactic sugar tensor[idx] -> tensor.select(idx).
  Value* emitGather(
      const SourceRange& loc,
      TreeList&& inputs) {
    const auto applyInputs =
        Compound::create(TK_LIST, loc, std::move(inputs));
    const auto input_values = getValues(applyInputs->trees());
    Value* tensor = input_values[0];
    const auto& idx = at::Scalar(input_values[1]->node()->t(attr::value)).toInt();
    return emitNode(
               Symbol::aten("select"),
               loc,
               {tensor},
               1)
               ->i_(attr::dim, 0)
               ->i_(attr::index, idx)
               ->output();
  }

  Value* createConstant(const SourceRange& loc, const at::Tensor& val) {
    auto n = graph->createConstant(val);
    n->setSourceLocation(std::make_shared<SourceRange>(loc));
    return graph->insertNode(n)->output();
  }
};

// support syntax sugar for x.foo(y, z) by allowing x.foo to return a
// callable value that will resolve to foo(x, y, z) when called.
std::shared_ptr<SugaredValue> SimpleValue::attr(SourceRange loc, Method & m, const std::string& field) {
  return std::make_shared<BuiltinFunction>(field, value);
}

std::vector<Value*> inlineCallTo(Graph& g, Graph& callee, ArrayRef<Value*> inputs) {
  std::unordered_map<Value*, Value*> value_map;
  auto value_map_func = [&](Value* v) { return value_map.at(v); };
  JIT_ASSERT(callee.inputs().size() == inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    value_map[callee.inputs()[i]] = inputs[i];
  }
  for (auto* node : callee.nodes()) {
    auto* new_node =
        g.insertNode(g.createClone(node, value_map_func));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      value_map[node->outputs()[i]] = new_node->outputs()[i];
    }
  }

  std::vector<Value*> outputs;
  for (auto* output : callee.outputs()) {
    outputs.push_back(value_map_func(output));
  }
  return outputs;
}

void defineMethodsInModule(Module & m, const std::vector<Def>& definitions, const Resolver& resolver, SugaredValuePtr self) {
  FunctionTable table;
  for(auto def : definitions) {
    const std::string& name = def.name().name();
    Method& method = m.create_method(name);
    to_ir(def, table, resolver, self,  method);
    auto result = table.emplace(name, method);
    if(!result.second) {
      throw ErrorReport(def) << "duplicate definition of function '" << name << "'";
    }
  }
}

void defineMethodsInModule(Module & m, const std::string& source, const Resolver& resolver, SugaredValuePtr self) {
  Parser p(source);
  std::vector<Def> definitions;
  while (p.lexer().cur().kind != TK_EOF) {
    definitions.push_back(Def(p.parseFunction()));
  }
  defineMethodsInModule(m, definitions, resolver, self);
}

std::shared_ptr<Graph> compileFunction(Def def, const Resolver& resolver) {
  Module m; //note: we don't use 'm' to execute so this setting is unused
  defineMethodsInModule(m, {def}, resolver, nullptr);
  return m.get_method(def.name().name()).graph();
}

} // namespace script
} // namespace jit
} // namespace torch
