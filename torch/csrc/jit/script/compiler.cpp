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
  std::unique_ptr<Graph> graph;
};

} // namespace

using FunctionTable = std::unordered_map<std::string, FunctionDefinition>;
using ValueTable = std::unordered_map<std::string, Value*>;

struct to_ir {
  to_ir(FunctionDefinition& def, FunctionTable& function_table)
      : def(def), function_table(function_table) {
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
  void emitStatements(const ListView<TreeRef>& statements) {
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
        default:
          emitExpr(stmt, 0);
          break;
      }
    }
  }
  void emitIf(const If& stmt) {
    // TODO: add support for control flow ops
    throw ErrorReport(stmt) << "Control flow is not supported yet.";
  }

  void emitWhile(const While& stmt) {
    // TODO: add support for control flow ops
    throw ErrorReport(stmt) << "Control flow is not supported yet.";
  }

  std::vector<Value*> emitAssignment(const Assign& stmt) {
    std::vector<Value*> outputs{stmt.lhs().size()};
    if (stmt.reduction() != '=') {
      if (stmt.lhs().size() != 1) {
        throw ErrorReport(stmt)
            << "reductions are only allow when there is a single variable "
            << "on the left-hand side.";
      }
      auto lhs = stmt.lhs()[0];
      auto expr =
          Compound::create(stmt.reduction(), stmt.range(), {lhs, stmt.rhs()});
      outputs = emitExpr(expr, 1);
    } else {
      outputs = emitExpr(stmt.rhs(), stmt.lhs().size());
    }
    int i = 0;
    for (auto ident : stmt.lhs()) {
      if (ident->kind() == TK_IDENT)
        setVar(Ident(ident).name(), outputs.at(i));
      i++;
    }
    return outputs;
  }

  void setVar(const std::string& name, Value* value) {
    value_table[name] = value;
  }

  Value* getVar(const Ident& ident) {
    if (value_table.count(ident.name()) == 0)
      throw ErrorReport(ident) << "undefined value " << ident.name();
    return value_table[ident.name()];
  }

  NodeKind getNodeKind(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return kAdd;
      case '-':
        if (ninputs == 1)
          return kneg;
        else
          return kSub;
      case '*':
        return kMul;
      case '/':
        return kDiv;
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

    std::unordered_map<Value*, Value*> value_table;
    auto value_map = [&](Value* v) { return value_table.at(v); };
    for (size_t i = 0; i < inputs.size(); ++i) {
      value_table[fn.graph->inputs()[i]] = inputs[i];
    }
    for (auto* node : fn.graph->nodes()) {
      auto* new_node =
          def.graph->appendNode(def.graph->createClone(node, value_map));
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        value_table[node->outputs()[i]] = new_node->outputs()[i];
        new_node->outputs()[i]->copyMetadata(node->outputs()[i]);
      }
    }

    std::vector<Value*> outputs{};
    for (auto* output : fn.graph->outputs()) {
      outputs.push_back(value_map(output));
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
      case TK_IDENT: {
        expectOutputs(tree, output_size, 1);
        return {getVar(Ident(tree))};
      } break;
      case TK_NE:
      case TK_EQ:
      case '<':
      case '>':
      case TK_LE:
      case TK_GE:
      case '-':
      case '*':
      case '/':
      case '+':
      case TK_AND:
      case TK_OR:
      case TK_NOT: {
        expectOutputs(tree, output_size, 1);
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        return emitNode(kind, getValues(inputs), {}, output_size);
      } break;
      case TK_APPLY: {
        auto apply = Apply(tree);
        if (function_table.count(apply.name().name()) > 0) {
          return emitFunctionCall(apply, output_size);
        } else {
          const auto& inputs = getValues(apply.inputs());
          NodeKind kind{apply.name().name()};

          std::unordered_map<std::string, TreeRef> attributes{};
          for (const auto& attr : apply.attributes()) {
            attributes[attr.name().name()] = attr.value();
          }
          return emitNode(kind, inputs, attributes, output_size);
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

  std::vector<Value*> emitCast(const TreeRef& input, const int type) {
    return emitNode(
        Symbol("type_as"),
        {emitExpr(input, 1)[0],
         createConstant(at::CPU(at::kInt).scalarTensor(0))},
        {},
        1);
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

  std::vector<Value*> emitNode(
      NodeKind kind,
      const std::vector<Value*> inputs,
      const std::unordered_map<std::string, TreeRef>& attributes,
      const size_t output_size) {
    Node* n = def.graph->appendNode(def.graph->create(kind, output_size));
    for (auto* input_value : inputs) {
      n->addInput(input_value);
    }
    for (const auto& iter : attributes) {
      const auto& name = Symbol(iter.first);
      const auto& value = iter.second;
      // TODO: handle non-float attributes
      switch (value->kind()) {
        case TK_CONST: {
          auto v = value->tree(0)->doubleValue();
          auto type = value->tree(1)->stringValue();
          if (type == "f") {
            n->f_(name, v);
          } else {
            n->i_(name, v);
          }
        } break;
        case TK_LIST:
          if (value->trees().size()) {
            std::vector<double> values{};
            for (const auto& tree : value->trees()) {
              values.push_back(tree->tree(0)->doubleValue());
            }
            if (value->trees()[0]->tree(1)->stringValue() == "f") {
              n->fs_(name, std::move(values));
            } else {
              n->is_(name, std::vector<int64_t>(values.begin(), values.end()));
            }
          }
          break;
      }
    }
    return n->outputs();
  }

  // Desugars slice syntactic sugar tensor[begin:end] -> tensor.slice(begin,
  // end).
  std::vector<Value*> emitSlice(
      const SourceRange& range,
      TreeList&& inputs,
      const size_t output_size) {
    const auto applyInputs =
        Compound::create(TK_LIST, range, std::move(inputs));
    std::vector<Value*> input_values = getValues(applyInputs->trees());
    auto* dim_0 = createConstant(at::CPU(at::kInt).scalarTensor(0));
    input_values.insert(input_values.begin() + 1, dim_0);
    return emitNode(
        Symbol("slice"), input_values, {}, output_size);
  }

  // Desugars gather syntactic sugar tensor[idx] -> tensor.select(idx).
  std::vector<Value*> emitGather(
      const SourceRange& range,
      TreeList&& inputs,
      const size_t output_size) {
    const auto applyInputs =
        Compound::create(TK_LIST, range, std::move(inputs));
    std::vector<Value*> input_values = getValues(applyInputs->trees());
    auto* dim_0 = createConstant(at::CPU(at::kInt).scalarTensor(0));
    input_values.insert(input_values.begin() + 1, dim_0);
    return emitNode(
        Symbol("select"), input_values, {}, output_size);
  }

  FunctionDefinition& def; // the def being constructed
  FunctionTable& function_table;
  ValueTable value_table;

 private:
  Value* createConstant(const at::Tensor& val) {
    return def.graph->appendNode(def.graph->createConstant(val))->output();
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

  const Graph& getGraph(const std::string& func_name) {
    if (functions.count(func_name) == 0)
      throw ErrorReport() << "undefined function: " << func_name << "\n";
    auto& def = functions.at(func_name);
    return *def.graph;
  }

 private:
  friend struct to_ir;
  FunctionTable functions;
};

CompilationUnit::CompilationUnit() : pImpl(new CompilationUnitImpl()) {}

void CompilationUnit::define(const std::string& script) {
  return pImpl->define(script);
}

const Graph& CompilationUnit::getGraph(const std::string& func_name) {
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
