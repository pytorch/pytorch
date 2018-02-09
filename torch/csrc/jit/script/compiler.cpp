#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/parser.h"

namespace torch {
namespace jit {
namespace script {

namespace {

// record of defined function
// Graph + metadata
struct FunctionDefinition {
  explicit FunctionDefinition(Def tree_)
      : tree(new Def(tree_)), graph(new Graph()) {}

  explicit FunctionDefinition(std::unique_ptr<Graph> graph_)
      : tree(nullptr), graph(std::move(graph_)) {}

  bool isExtern() const {
    return tree == nullptr;
  }
  std::unique_ptr<Def> tree;
  std::shared_ptr<Graph> graph;
};

} // namespace

using FunctionTable = std::unordered_map<std::string, FunctionDefinition>;
using ValueTable = std::unordered_map<std::string, Value*>;
using AttributeMap =
    std::unordered_map<std::string, std::pair<double, std::string>>;
using ListAttributeMap = std::unordered_map<
    std::string,
    std::pair<const std::vector<double>, std::string>>;

// Keep track of environment as we descend down nested control
// structures.
struct Environment {
  enum class Type {
    kNormal,
    kWhile,
    kIf,
  };

  Environment(
      Type t,
      Block* b = nullptr,
      std::shared_ptr<Environment> next = nullptr)
      : t(t), b(b), lazy(false), next(next) {}

  Type t;

  std::vector<std::string> positional_inputs;
  ValueTable value_table;
  Block* b;
  // When referring to or setting a value, do not create a new block input if
  // true
  bool lazy;

  std::shared_ptr<Environment> next;
};

struct to_ir {
  to_ir(FunctionDefinition& def, FunctionTable& function_table)
      : def(def), function_table(function_table) {
    environment_stack =
        std::make_shared<Environment>(Environment::Type::kNormal);
    // populate def->graph
    auto& tree = *def.tree;
    for (auto input : tree.params()) {
      auto& name = input.ident().name();
      setVar(name, def.graph->addInput(name));
    }
    emitStatements(tree.statements());
    for (auto output : tree.returns()) {
      def.graph->registerOutput(getVar(output.ident()));
    }
  }
  void emitStatements(const List<TreeRef>& statements) {
    for (auto stmt : statements) {
      switch (stmt->kind()) {
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
          for (auto ident : stmt->trees()) {
            const auto& name = Ident(ident).name();
            setVar(name, def.graph->addInput(name));
          }
          break;
        case TK_EXPR_STMT:
          emitExpr(ExprStmt(stmt).expr(), 0);
          break;
      }
    }
  }

  // Given that after emitting statements in a block, we've added block inputs
  // for all value references and assignments, delete inputs for which there was
  // no assignment, only references.
  void deleteExtraInputs(Block* b, std::shared_ptr<Environment> e) {
    auto& curr_frame = *e;
    std::vector<size_t> inputs_to_delete;
    int i = 0;
    for (const auto& x : curr_frame.positional_inputs) {
      if (b->inputs()[i] == curr_frame.value_table[x]) {
        inputs_to_delete.push_back(i);
      }
      i++;
    }

    for (auto ritr = inputs_to_delete.rbegin(); ritr != inputs_to_delete.rend();
         ++ritr) {
      auto name = curr_frame.positional_inputs[*ritr];
      Value* v = curr_frame.value_table[name];
      Value* orig = findInParentFrame(name);
      // Replace all matching node inputs with original value
      // from an enclosing scope
      for (auto node : b->nodes()) {
        for (size_t i = 0; i < node->inputs().size(); ++i) {
          if (node->input(i) == v) {
            node->replaceInput(i, orig);
          }
        }
      }

      // Actually remove the input
      b->eraseInput(*ritr);
      curr_frame.positional_inputs.erase(
          curr_frame.positional_inputs.begin() + *ritr);
    }
  }

  void emitIf(const If& stmt) {
    Value* cond_value = emitExpr(stmt.cond(), 1)[0];

    Node* n = def.graph->insertNode(def.graph->create(Symbol("If"), 0));
    n->addInput(cond_value);
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    // Emit both blocks once to get the union of all mutated values
    std::shared_ptr<Environment> save_true_environment, save_false_environment;
    {
      environment_stack = std::make_shared<Environment>(
          Environment::Type::kIf, true_block, environment_stack);
      WithInsertPoint guard(*def.graph, true_block);
      emitStatements(stmt.trueBranch());
      deleteExtraInputs(true_block, environment_stack);

      save_true_environment = environment_stack;
      environment_stack = environment_stack->next;
    }

    {
      environment_stack = std::make_shared<Environment>(
          Environment::Type::kIf, false_block, environment_stack);
      WithInsertPoint guard(*def.graph, false_block);
      emitStatements(stmt.falseBranch());
      deleteExtraInputs(false_block, environment_stack);

      save_false_environment = environment_stack;
      environment_stack = environment_stack->next;
    }

    std::unordered_set<std::string> all_mutated_values;
    for (auto& input : save_true_environment->positional_inputs) {
      all_mutated_values.insert(input);
    }
    for (auto& input : save_false_environment->positional_inputs) {
      all_mutated_values.insert(input);
    }

    // Delete both blocks
    n->eraseBlock(1);
    n->eraseBlock(0);

    true_block = n->addBlock();
    false_block = n->addBlock();

    {
      environment_stack = std::make_shared<Environment>(
          Environment::Type::kIf, true_block, environment_stack);
      environment_stack->lazy = true;

      // Pre-populate block inputs with the union of mutated values from both
      // branches
      for (const auto& input : all_mutated_values)
        createBlockInput(input);

      WithInsertPoint guard(*def.graph, true_block);
      emitStatements(stmt.trueBranch());

      auto& curr_frame = *environment_stack;
      for (const auto& x : curr_frame.positional_inputs) {
        true_block->registerOutput(curr_frame.value_table[x]);
      }

      environment_stack = environment_stack->next;
    }

    {
      environment_stack = std::make_shared<Environment>(
          Environment::Type::kIf, false_block, environment_stack);
      environment_stack->lazy = true;

      // Pre-populate block inputs with the union of mutated values from both
      // branches
      for (const auto& input : all_mutated_values)
        createBlockInput(input);

      WithInsertPoint guard(*def.graph, false_block);
      emitStatements(stmt.falseBranch());

      auto& curr_frame = *environment_stack;
      for (const auto& x : curr_frame.positional_inputs) {
        false_block->registerOutput(curr_frame.value_table[x]);
      }

      environment_stack = environment_stack->next;
    }

    // Add op inputs and outputs
    for (const auto& x : all_mutated_values) {
      n->addInput(getVar(x));
      setVar(x, n->addOutput());
    }
  }

  void emitWhile(const While& stmt) {
    // TODO: scan outputs

    Value* trip_count_dummy = emitConst(0, "i")[0];
    Value* cond_value = emitExpr(stmt.cond(), 1)[0];

    Node* n = def.graph->insertNode(def.graph->create(Symbol("Loop"), 0));
    n->addInput(trip_count_dummy);
    n->addInput(cond_value);
    auto* body_block = n->addBlock();

    // TODO iteration num and condition

    {
      environment_stack = std::make_shared<Environment>(
          Environment::Type::kWhile, body_block, environment_stack);
      WithInsertPoint guard(*def.graph, body_block);
      emitStatements(stmt.body());

      // Also emit the conditional
      Value *body_cond_value = emitExpr(stmt.cond(), 1)[0];
      body_block->registerOutput(body_cond_value);

      // Remove inputs for values that did not mutate within the
      // block
      deleteExtraInputs(body_block, environment_stack);

      // Add block outputs
      auto& curr_frame = *environment_stack;
      for (const auto& x : curr_frame.positional_inputs) {
        body_block->registerOutput(curr_frame.value_table[x]);
      }

      auto preserve_positional_inputs = curr_frame.positional_inputs;

      // Drop out of block environment
      environment_stack = environment_stack->next;

      // Add op inputs
      for (const auto& x : preserve_positional_inputs) {
        n->addInput(getVar(x));
        setVar(x, n->addOutput());
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
    for (const Ident& ident : stmt.lhs()) {
      setVar(ident.name(), outputs.at(i));
      i++;
    }
    return outputs;
  }

  Value* findInThisFrame(const std::string& name) {
    const auto& e = *environment_stack;
    if (e.value_table.count(name)) {
      return e.value_table.at(name);
    }
    return nullptr;
  }

  Value* findInParentFrame(const std::string& name) {
    for (auto runner = environment_stack->next; runner; runner = runner->next) {
      if (runner->value_table.count(name)) {
        return runner->value_table.at(name);
      }
    }
    return nullptr;
  }

  Value* createBlockInput(const std::string& name) {
    auto& curr_frame = *environment_stack;
    Block* b = curr_frame.b;

    // Create the input
    Value* new_input = b->addInput();

    // Associate this name with this value
    curr_frame.value_table[name] = new_input;

    // List as a positional input
    curr_frame.positional_inputs.push_back(name);

    return new_input;
  }

  void setVar(const std::string& name, Value* value) {
    auto& curr_frame = *environment_stack;

    switch (curr_frame.t) {
      case Environment::Type::kNormal: {
        curr_frame.value_table[name] = value;
      } break;

      case Environment::Type::kWhile:
      case Environment::Type::kIf: {
        // Overwriting an existing value means it's already been
        // accounted for.
        if (findInThisFrame(name)) {
          curr_frame.value_table[name] = value;
          return;
        }

        // Value not here. search in parent scopes
        if (findInParentFrame(name)) {
          // Writing to a value in a parent frame. Make this a
          // loop-carried dependency.
          if (!curr_frame.lazy)
            createBlockInput(name);

          // Overwrite in value map
          curr_frame.value_table[name] = value;
        } else {
          // Not accessing a value in enclosing scope. Make a new
          // value as usual
          curr_frame.value_table[name] = value;
        }
      } break;
    }
  }

  Value* getVar(const Ident& ident) {
    try {
      return getVar(ident.name());
    } catch (ErrorReport e) {
      throw ErrorReport(ident) << e.what();
    }
  }

  Value* getVar(const std::string& ident) {
    Value* retval = findInThisFrame(ident);
    if (retval) {
      return retval;
    }

    retval = findInParentFrame(ident);

    auto& curr_frame = *environment_stack;
    switch (curr_frame.t) {
      case Environment::Type::kNormal: {
        return retval;
      } break;

      case Environment::Type::kWhile:
      case Environment::Type::kIf: {
        if (retval) {
          // Reading from a value in a parent frame. Make this a
          // loop-carried dependency or explicit input
          if (!curr_frame.lazy)
            retval = createBlockInput(ident);

          return retval;
        } else {
          throw ErrorReport() << "undefined value " << ident;
        }
      } break;
    }
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

  // emit a function call by inlining the function's Graph into our
  // Graph
  std::vector<Value*> emitFunctionCall(Apply& apply, const size_t output_size) {
    auto& fn = function_table.at(apply.name().name());
    std::vector<Value*> inputs = getValues(apply.inputs());

    std::unordered_map<Value*, Value*> value_map;
    auto value_map_func = [&](Value* v) { return value_map.at(v); };
    for (size_t i = 0; i < inputs.size(); ++i) {
      value_map[fn.graph->inputs()[i]] = inputs[i];
    }
    for (auto* node : fn.graph->nodes()) {
      auto* new_node =
          def.graph->insertNode(def.graph->createClone(node, value_map_func));
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        value_map[node->outputs()[i]] = new_node->outputs()[i];
        new_node->outputs()[i]->copyMetadata(node->outputs()[i]);
      }
    }

    std::vector<Value*> outputs{};
    for (auto* output : fn.graph->outputs()) {
      outputs.push_back(value_map_func(output));
    }
    return outputs;
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

  // This will _always_ compute something, unlike 'getValue' which simply
  // returns an already computed reference if possible.
  std::vector<Value*> emitExpr(
      const TreeRef& tree,
      const size_t output_size = 0) {
    switch (tree->kind()) {
      case TK_VAR: {
        expectOutputs(tree, output_size, 1);
        return {getVar(Var(tree).name())};
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
        return emitNode(kind, getValues(inputs), output_size)->outputs();
      } break;
      case '+':
      case '-': {
        expectOutputs(tree, output_size, 1);
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        auto* node = emitNode(kind, getValues(inputs), output_size);
        node->t_(Symbol("alpha"), at::CPU(at::kFloat).scalarTensor(1.0));
        return node->outputs();
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        if (function_table.count(apply.name().name()) > 0) {
          return emitFunctionCall(apply, output_size);
        } else {
          const auto& inputs = getValues(apply.inputs());
          NodeKind kind{apply.name().name()};

          AttributeMap attributes{};
          ListAttributeMap list_attributes{};
          for (const auto& attr : apply.attributes()) {
            const auto& name = attr.name().name();
            const Expr& value = attr.value();
            // TODO: handle non-float attributes
            switch (value.kind()) {
              case TK_CONST: {
                auto v = value.get()->tree(0)->doubleValue();
                const auto& type = value.get()->tree(1)->stringValue();
                attributes.insert({name, {v, type}});
              } break;
              case TK_LIST:
                std::vector<double> vs{};
                for (const auto& tree : value.get()->trees()) {
                  vs.push_back(tree->tree(0)->doubleValue());
                }
                const auto& type = value.get()->trees()[0]->tree(1)->stringValue();
                list_attributes.insert({name, {std::move(vs), type}});
            }
            break;
            break;
          }
          return emitNode(
                     kind, inputs, output_size, attributes, list_attributes)
              ->outputs();
        }
      } break;
      case TK_CAST: {
        expectOutputs(tree, output_size, 1);
        const auto cast = Cast(tree);
        return emitCast(cast.input(), cast.type());
      } break;
      case TK_CONST: {
        expectOutputs(tree, output_size, 1);
        return emitConst(
            tree->tree(0)->doubleValue(), tree->tree(1)->stringValue());
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
      case '.':
        // TODO: add support for "."
      case TK_IF_EXPR:
        // TODO: add support for conditional
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
               {emitExpr(input, 1)[0], createConstant(at::CPU(t).ones({1}))},
               1)
        ->outputs();
  }

  std::vector<Value*> emitConst(const double val, const std::string& type) {
    if (type == "f") {
      return {createConstant(at::CPU(at::kFloat).scalarTensor(val))};
    } else if (type == "LL") {
      return {createConstant(at::CPU(at::kLong).scalarTensor(val))};
    } else if (type == "b") {
      return {createConstant(at::CPU(at::kByte).scalarTensor(val))};
    } else if (type == "i") {
      return {createConstant(at::CPU(at::kInt).scalarTensor(val))};
    } else {
      throw std::runtime_error("unknown const type " + type);
    }
  }

  Node* emitNode(
      NodeKind kind,
      const std::vector<Value*> inputs,
      const size_t output_size,
      const AttributeMap& attributes = AttributeMap{},
      const ListAttributeMap& list_attributes = ListAttributeMap{}) {
    Node* n = def.graph->insertNode(def.graph->create(kind, output_size));
    for (auto* input_value : inputs) {
      n->addInput(input_value);
    }
    for (const auto& attr : attributes) {
      const auto name = Symbol(attr.first);
      auto value = attr.second.first;
      const auto& type = attr.second.second;
      if (type == "f") {
        n->f_(name, value);
      } else {
        n->i_(name, value);
      }
    }
    for (const auto& attr : list_attributes) {
      const auto name = Symbol(attr.first);
      const auto& values = attr.second.first;
      const auto& type = attr.second.second;
      if (type == "f") {
        n->fs_(name, std::vector<double>{values.begin(), values.end()});
      } else {
        n->is_(name, std::vector<int64_t>{values.begin(), values.end()});
      }
    }
    return n;
  }

  // Desugars slice syntactic sugar tensor[begin:end] -> tensor.slice(begin,
  // end).
  std::vector<Value*> emitSlice(
      const SourceRange& range,
      TreeList&& inputs,
      const size_t output_size) {
    const auto applyInputs =
        Compound::create(TK_LIST, range, std::move(inputs));
    const auto input_values = getValues(applyInputs->trees());
    Value* tensor = input_values[0];
    const auto& begin = at::Scalar(input_values[1]->node()->t(kvalue)).toInt();
    const auto& end = at::Scalar(input_values[2]->node()->t(kvalue)).toInt();
    return emitNode(
               Symbol("slice"),
               {tensor},
               output_size,
               {{"dim", {0, "LL"}},
                {"step", {1, "LL"}},
                {"start", {begin, "LL"}},
                {"end", {end, "LL"}}})
        ->outputs();
  }

  // Desugars gather syntactic sugar tensor[idx] -> tensor.select(idx).
  std::vector<Value*> emitGather(
      const SourceRange& range,
      TreeList&& inputs,
      const size_t output_size) {
    const auto applyInputs =
        Compound::create(TK_LIST, range, std::move(inputs));
    const auto input_values = getValues(applyInputs->trees());
    Value* tensor = input_values[0];
    const auto& idx = at::Scalar(input_values[1]->node()->t(kvalue)).toInt();
    return emitNode(
               Symbol("select"),
               {tensor},
               output_size,
               {{"dim", {0, "LL"}}, {"index", {idx, "LL"}}})
        ->outputs();
  }

  FunctionDefinition& def; // the def being constructed
  FunctionTable& function_table;

  // Singly-linked list of environments. This top element contains a member
  // `next` that points to the most immediate enclosing scope's value.
  std::shared_ptr<Environment> environment_stack;

 private:
  Value* createConstant(const at::Tensor& val) {
    return def.graph->insertNode(def.graph->createConstant(val))->output();
  }
};

struct CompilationUnitImpl {
  void defineFunction(const Def& def) {
    const auto& name = def.name().name();

    if (functions.count(name) > 0) {
      throw ErrorReport(def) << name << " already defined.";
    }

    auto it = functions.emplace(name, FunctionDefinition{def}).first;
    to_ir(it->second, functions);
  }

  void define(const std::string& script) {
    Parser p(script);
    while (p.lexer().cur().kind != TK_EOF) {
      defineFunction(Def(p.parseFunction()));
    }
  }

  std::shared_ptr<Graph> getGraph(const std::string& func_name) {
    if (functions.count(func_name) == 0)
      throw ErrorReport() << "undefined function: " << func_name << "\n";
    auto& def = functions.at(func_name);
    return def.graph;
  }

 private:
  friend struct to_ir;
  FunctionTable functions;
};

CompilationUnit::CompilationUnit() : pImpl(new CompilationUnitImpl()) {}

void CompilationUnit::define(const std::string& script) {
  return pImpl->define(script);
}

std::shared_ptr<Graph> CompilationUnit::getGraph(const std::string& func_name) {
  return pImpl->getGraph(func_name);
}

CompilationUnit::~CompilationUnit() {}

std::unique_ptr<CompilationUnit> jitScriptCompile(const std::string& script) {
  std::unique_ptr<CompilationUnit> cu{new CompilationUnit};
  cu->define(script);
  return cu;
}

} // namespace script
} // namespace jit
} // namespace torch
