#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/generated/aten_dispatch.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/parser.h"
#include "torch/csrc/utils/object_ptr.h"

#include <climits>

namespace torch {
namespace jit {
namespace script {

using SugaredValuePtr = std::shared_ptr<SugaredValue>;
using FunctionTable = std::unordered_map<std::string, Method&>;
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using AttributeMap = std::unordered_map<std::string, Const>;
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

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

  Symbol getBlockOwningKind() {
    Symbol owning_kind = Symbol();
    if (b->owningNode()) {
      owning_kind = b->owningNode()->kind();
    }
    return owning_kind;
  }

  void setVar(const std::string& name, Value* value) {
    setSugaredVar(name, std::make_shared<SimpleValue>(value));
  }
  void setSugaredVar(const std::string& name, SugaredValuePtr value) {
    if (!findInThisFrame(name) && findInParentFrame(name) &&
        getBlockOwningKind() == kLoop)
      createCapturedInput(name);
    value_table[name] = std::move(value);
  }

  SugaredValuePtr getSugaredVar(const Ident& ident, bool required=true) {
    return getSugaredVar(ident.name(), ident);
  }
  Value* getVar(const Ident& ident) {
    return getSugaredVar(ident)->asValue(ident.range(), method);
  }

  SugaredValuePtr getSugaredVar(const std::string& ident, const TreeView& tv, bool required=true) {
    auto retval = findInThisFrame(ident);

    if (!retval && (retval = findInParentFrame(ident)) &&
        getBlockOwningKind() == kLoop) {
      retval = createCapturedInput(ident);
    }

    if(!retval) {
      retval = resolver(ident);
    }

    if (!retval && required) {
      throw ErrorReport(tv) << "undefined value " << ident;
    }
    return retval;
  }

  Value* getVar(const std::string& ident, const TreeView& tv) {
    return getSugaredVar(ident, tv)->asValue(tv.range(), method);
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

Node* emitBuiltinCall(
  const SourceRange& loc,
  Method& method,
  const std::string & name,
  at::ArrayRef<Value*> inputs,
  List<Attribute> attributes,
  size_t n_outputs) {

  NodeKind kind(name);
  auto graph = method.graph();
  auto n = graph->insertNode(graph->create(kind, inputs, n_outputs))
                ->setSourceLocation(std::make_shared<SourceRange>(loc));

  for (const auto& attr : attributes) {
    const auto& name = Symbol(attr.name().name());
    const Expr& value_expr = attr.value();
    switch (value_expr.kind()) {
      case TK_CONST: {
        Const value {value_expr};
        if (value.isFloatingPoint()) {
          n->f_(name, value.asFloatingPoint());
        } else {
          n->i_(name, value.asIntegral());
        }
      } break;
      case TK_LIST_LITERAL: {
        List<Const> value_list {ListLiteral(value_expr).inputs()};
        std::vector<Const> values;
        for (Const number : value_list)
          values.push_back(std::move(number));
        bool is_float = std::any_of(values.begin(), values.end(),
                                    [](const Const& c) { return c.isFloatingPoint(); });
        if (is_float) {
          n->fs_(name, fmap(values, [](const Const& c) { return c.asFloatingPoint(); }));
        } else {
          n->is_(name, fmap(values, [](const Const& c) { return c.asIntegral(); }));
        }
      } break;
    default:
        throw ErrorReport(attr) << "Unexpected kind of attribute value: " << value_expr.kind();
        break;
    }
  }

  return n;
}

std::vector<Value*> BuiltinFunction::call(
    SourceRange loc,
    Method & m,
    at::ArrayRef<Value*> inputs_,
    List<Attribute> attributes,
    size_t n_outputs) {
  std::vector<Value*> inputs;
  if (value) inputs.push_back(value);
  inputs.insert(inputs.end(), inputs_.begin(), inputs_.end());
  Node * n = emitBuiltinCall(loc, m, name, inputs, attributes, n_outputs);
  if (!hasTensorOp(n)) {
    throw ErrorReport(loc) << "unknown builtin op";
  }
  return n->outputs();
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
      environment_stack->setSugaredVar((*it).ident().name(), self);
      ++it;
    }
    for(;it != end; ++it) {
      auto& name = (*it).ident().name();
      environment_stack->setVar(name, graph->addInput(name));
    }
    // body
    auto stmts = def.statements();
    auto stmts_begin = stmts.begin();
    auto stmts_end = stmts.end();
    if (stmts_begin == stmts_end)
      throw ErrorReport(def) << "functions need to have a non-empty body";
    --stmts_end;
    if ((*stmts_end).kind() != TK_RETURN)
      throw ErrorReport(*stmts_end) << "functions need to end with a return statement";

    emitStatements(stmts_begin, stmts_end);

    // outputs
    for (auto output : Return(*stmts_end).values()) {
      graph->registerOutput(emitExpr(output, 1)[0]);
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
        case TK_ASSIGN:
          emitAssignment(Assign(stmt));
          break;
        case TK_GLOBAL:
          for (auto ident : Global(stmt).names()) {
            const auto& name = Ident(ident).name();
            environment_stack->setVar(name, graph->addInput(name));
          }
          break;
        case TK_EXPR_STMT:
          emitExpr(ExprStmt(stmt).expr(), 0);
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

  Node* create(Symbol kind, const SourceRange& loc,  size_t num_outputs) {
    return graph
             ->create(kind, num_outputs)
             ->setSourceLocation(std::make_shared<SourceRange>(loc));
  }

  std::vector<Value*> emitTernaryIf(const TernaryIf& expr) {
    Value* cond_value = emitExpr(expr.cond(), 1)[0];

    Node* n = graph->insertNode(create(kIf, expr.range(), 0));
    n->addInput(cond_value);
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    auto emit_if_expr = [this](Block* b, const Expr& expr) {
      pushFrame(b);
      WithInsertPoint guard(b);
      Value* out_val = emitExpr(expr, 1)[0];
      b->registerOutput(out_val);
      popFrame();
    };

    emit_if_expr(true_block, expr.true_expr());
    emit_if_expr(false_block, expr.false_expr());

    // Add op outputs
    auto expr_value = n->addOutput(); // Resulting value

    return {expr_value};
  }

  void emitIf(const If& stmt) {
    Value* cond_value = emitExpr(stmt.cond(), 1)[0];

    Node* n = graph->insertNode(create(kIf, stmt.range(), 0));
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
      true_block->registerOutput(save_true->getVar(x, stmt));
    }
    for (const auto& x : sorted_mutations) {
      false_block->registerOutput(save_false->getVar(x, stmt));
    }

    // Add op outputs
    for (const auto& x : sorted_mutations) {
      environment_stack->setVar(x, n->addOutput());
    }
  }

  void emitWhile(const While& stmt) {
    // Emits a loop operators conforming to the semantics specified at
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#experimental-loop
    // TODO: implement scan_outputs

    // the format of the Loop instruction is:
    // loop_carried_outputs* = Loop(max_trip_count, start_condition, loop_carried_inputs*)
    //                          block0(loop_counter, loop_carried_block*) {
    //                             <body>
    //                             -> (continue_condition, loop_carried_block_outputs*)
    //                          }
    // all loop_carried_... lists are the same length and represent the value of
    // loop-carried variables whose definitions are updated as the loop executes
    // in a way that ensure single static assignment.

    // TODO: clarify that this is an optional input that isn't needed here
    Value* max_trip_count_dummy = emitConst(Const::create(stmt.range(), std::to_string(INT_MAX)))[0];
    Value* cond_value = emitExpr(stmt.cond(), 1)[0];

    Node* n = graph->insertNode(create(kLoop, stmt.range(), 0));
    n->addInput(max_trip_count_dummy);
    n->addInput(cond_value);
    auto* body_block = n->addBlock();
    // Trip count required by spec. Since this is a while loop, we do not
    // provide access to this from user code
    // TODO: it seems like we should implement a `for` loop as well, otherwise
    // we'll probably have to pattern match iteration number machinery in user
    // code to conform to the spec
    body_block->addInput(); // Iteration num
    size_t skip_inputs_num = 1;

    {
      pushFrame(body_block);
      WithInsertPoint guard(body_block);
      emitStatements(stmt.body());

      // Also emit the conditional
      Value *body_cond_value = emitExpr(stmt.cond(), 1)[0];
      body_block->registerOutput(body_cond_value);

      auto body_frame = popFrame();
      auto outer_frame = environment_stack;
      // Remove inputs for values that did not mutate within the
      // block
      body_frame->deleteExtraInputs(stmt.range(), skip_inputs_num);

      // Add block outputs
      for (const auto& x : body_frame->captured_inputs) {
        body_block->registerOutput(body_frame->getValueInThisFrame(stmt.range(), x));
        n->addInput(outer_frame->getVar(x, stmt));
        outer_frame->setVar(x, n->addOutput());
      }

    }
  }

  std::vector<Value*> emitAssignment(const Assign& stmt) {
    std::vector<Value*> outputs{stmt.lhs().size()};
    if (stmt.reduction() != '=') {
      if (stmt.lhs().size() != 1) {
        throw ErrorReport(stmt)
            << "reductions are only allow when there is a single variable "
            << "on the left-hand side.";
      }
      Ident lhs = stmt.lhs()[0];
      Expr expr = BinOp::create(stmt.range(), stmt.reduction(),
                                Var::create(lhs.range(), lhs), stmt.rhs());
      outputs = emitExpr(expr, 1);
    } else {
      outputs = emitExpr(stmt.rhs(), stmt.lhs().size());
    }
    int i = 0;
    for (auto ident : stmt.lhs()) {
      environment_stack->setVar(Ident(ident).name(), outputs.at(i));
      i++;
    }
    return outputs;
  }

  NodeKind getNodeKind(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return kadd;
      case '-':
        if (ninputs == 1)
          return kneg;
        else
          return ksub;
      case '*':
        return kmul;
      case '/':
        return kdiv;
      case TK_NE:
        return kne;
      case TK_EQ:
        return keq;
      case '<':
        return klt;
      case '>':
        return kgt;
      case TK_LE:
        return kle;
      case TK_GE:
        return kge;
      case TK_AND:
        return k__and__;
      case TK_OR:
        return k__or__;
      case TK_NOT:
        return k__not__;
      default:
        throw std::runtime_error("unknown kind " + std::to_string(kind));
    }
  }

  template <typename Trees>
  std::vector<Value*> getValues(const Trees& trees) {
    std::vector<Value*> values;
    for (const auto& tree : trees) {
      values.push_back(emitExpr(tree, 1)[0]);
    }
    return values;
  }

  void expectOutputs(
      const TreeRef& tree,
      const size_t expected_size,
      const size_t size) {
    if (expected_size != 0 && expected_size != size) {
      throw ErrorReport(tree)
          << "expected operator to produce " << expected_size
          << " outputs but it produced " << size;
    }
  }

  // special rules apply when we directly call foo(a,b) when foo is an ident
  std::vector<Value*> emitApplyIdent(Ident ident, std::vector<Value*> inputs, List<Attribute> attributes, size_t output_size) {
    auto it = function_table.find(ident.name());
    if (it != function_table.end()) {
      if(inputs.size() != it->second.num_inputs())
        throw ErrorReport(ident) << "expected " << it->second.num_inputs() << " but found " << inputs.size();
      auto outputs = method.emit_call_to(it->second, inputs);
      expectOutputs(ident, output_size, outputs.size());
      return outputs;
    } else if (ident.name() == "print") {
      expectOutputs(ident, output_size, 0);
      if (!attributes.empty())
        throw ErrorReport(ident) << "print doesn't accept any keyword arguments";
      return emitNode(kPrint, ident.range(), inputs, 0 )->outputs();
    }
    Node* builtin = emitBuiltinCall(ident.range(), method, ident.name(), inputs, attributes, output_size);
    if (hasTensorOp(builtin)) {
      return builtin->outputs();
    }
    builtin->destroy();
    // it wasn't known built in, so treat it like standard apply
    return emitApplyExpr(Var::create(ident.range(), ident), inputs, attributes, output_size);
  }

  std::vector<Value*> emitApplyExpr(Expr callee, const std::vector<Value*>& inputs, List<Attribute> attributes, size_t output_size) {
    // otherwise we evaluate the callee and then desugar it
    auto sv = emitSugaredExpr(callee);
    return sv->call(callee.range(), method, inputs, attributes, output_size);
  }

  // any expression that can produce a SugaredValue are handled here
  // with emitExpr falling back to this function to handle them
  // the kinds handled here should be kept in sync with [SUGARED VALUES]
  // in emitExpr
  std::shared_ptr<SugaredValue> emitSugaredExpr(Expr tree) {
    switch(tree.kind()) {
      case TK_VAR:
        return environment_stack->getSugaredVar(Var(tree).name());
      case '.': {
        auto select = Select(tree);
        auto sv = emitSugaredExpr(select.value());
        return sv->attr(select.range(), method, select.selector().name());
      }
      default:
        return std::make_shared<SimpleValue>(emitExpr(tree, 1)[0]);
    }
  }

  std::vector<Value*> emitExpr(
      const TreeRef& tree,
      const size_t output_size = 0) {
    switch (tree->kind()) {
      // the expressions have special handling because they may operate
      // on sugared values
      // [SUGARED VALUES]
      case TK_VAR: case '.': {
        return { emitSugaredExpr(Expr(tree))->asValue(tree->range(), method) };
      } break;
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
      case TK_NOT: {
        expectOutputs(tree, output_size, 1);
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        return emitNode(kind, tree->range(), getValues(inputs), output_size)->outputs();
      } break;
      case '+':
      case '-': {
        expectOutputs(tree, output_size, 1);
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        auto* node = emitNode(kind, tree->range(), getValues(inputs), output_size);
        if (kind != kneg)
          node->t_(Symbol("alpha"), at::CPU(at::kFloat).scalarTensor(1.0));
        return node->outputs();
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        auto inputs = getValues(apply.inputs());
        // the apply is directly an identifier 'foo'
        if(apply.callee().kind() == TK_VAR) {
          return emitApplyIdent(Var(apply.callee()).name(), inputs, apply.attributes(), output_size);
        }
        return emitApplyExpr(apply.callee(), inputs, apply.attributes(), output_size);
      } break;
      case TK_CAST: {
        expectOutputs(tree, output_size, 1);
        const auto cast = Cast(tree);
        return emitCast(cast.input(), cast.type());
      } break;
      case TK_CONST: {
        expectOutputs(tree, output_size, 1);
        return emitConst(Const(tree));
      } break;
      case TK_SLICE: {
        expectOutputs(tree, output_size, 1);
        const auto slice = Slice(tree);
        return emitSlice(
            slice.range(),
            {slice.value(), slice.startOr(0), slice.endOr(-1)},
            output_size);
      } break;
      case TK_GATHER: {
        expectOutputs(tree, output_size, 1);
        const auto gather = Gather(tree);
        return emitGather(
            gather.range(), {gather.value(), gather.indices()}, output_size);
      } break;
      case TK_IF_EXPR: {
        expectOutputs(tree, output_size, 1);
        return emitTernaryIf(TernaryIf(tree));
      } break;
      default:
        throw ErrorReport(tree) << "NYI: " << tree;
        break;
    }
  }

  std::vector<Value*> emitCast(const TreeRef& input, const ScalarType& type) {
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
               Symbol("type_as"),
               input->range(),
               {emitExpr(input, 1)[0], createConstant(input->range(), at::ones(at::CPU(t), {1}))},
               1)
        ->outputs();
  }

  std::vector<Value*> emitConst(const Const& c) {
    if (c.isFloatingPoint()) {
      return {createConstant(c.range(), at::CPU(at::kFloat).scalarTensor(c.asFloatingPoint()))};
    } else {
      return {createConstant(c.range(), at::CPU(at::kLong).scalarTensor(c.asIntegral()))};
    }
  }

  Node* emitNode(
      NodeKind kind,
      const SourceRange& loc,
      const std::vector<Value*> inputs,
      const size_t output_size) {
    Node* n = graph->insertNode(create(kind, loc, output_size));
    for (auto* input_value : inputs) {
      n->addInput(input_value);
    }
    return n;
  }

  // Desugars slice syntactic sugar tensor[begin:end] -> tensor.slice(begin,
  // end).
  std::vector<Value*> emitSlice(
      const SourceRange& loc,
      TreeList&& inputs,
      const size_t output_size) {
    const auto applyInputs =
        Compound::create(TK_LIST, loc, std::move(inputs));
    const auto input_values = getValues(applyInputs->trees());
    Value* tensor = input_values[0];
    const auto& begin = at::Scalar(input_values[1]->node()->t(kvalue)).toInt();
    const auto& end = at::Scalar(input_values[2]->node()->t(kvalue)).toInt();
    return emitNode(
               Symbol("slice"),
               loc,
               {tensor},
               output_size)
               ->i_(kdim, 0)
               ->i_(kstep, 1)
               ->i_(kstart, begin)
               ->i_(kend, end)->outputs();
  }

  // Desugars gather syntactic sugar tensor[idx] -> tensor.select(idx).
  std::vector<Value*> emitGather(
      const SourceRange& loc,
      TreeList&& inputs,
      const size_t output_size) {
    const auto applyInputs =
        Compound::create(TK_LIST, loc, std::move(inputs));
    const auto input_values = getValues(applyInputs->trees());
    Value* tensor = input_values[0];
    const auto& idx = at::Scalar(input_values[1]->node()->t(kvalue)).toInt();
    return emitNode(
               Symbol("select"),
               loc,
               {tensor},
               output_size)
               ->i_(kdim, 0)
               ->i_(kindex, idx)
               ->outputs();
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
  Module m(/*optimize=*/false); //note: we don't use 'm' to execute so this setting is unused
  defineMethodsInModule(m, {def}, resolver, nullptr);
  return m.get_method(def.name().name()).graph();
}

} // namespace script
} // namespace jit
} // namespace torch
