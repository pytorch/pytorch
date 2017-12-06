#include "caffe2/core/net.h"
#include "caffe2/utils/proto_utils.h"

#include "compiler.h"
#include "parser.h"

namespace caffe2 {
namespace script {

struct DefCompiler {
  DefCompiler(const Def& def, NetDef& net_def)
      : def(def), net_def_stack({&net_def}) {}
  void run() {
    cur().set_name(def.name().name());
    for (auto input : def.params()) {
      auto& name = input.ident().name();
      map(name, name);
      // cur().add_external_input(name);
    }
    for (auto output : def.returns()) {
      auto& name = output.ident().name();
      map(name, name);
      // cur().add_external_output(name);
    }
    emitStatements(def.statements());
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
        default:
          throw ErrorReport(stmt) << "NYI: " << stmt;
      }
    }
  }
  void map(const std::string& name, const std::string& value) {
    env[name] = value;
  }
  const std::string& lookup(const Ident& ident) {
    if (env.count(ident.name()) == 0)
      throw ErrorReport(ident) << "undefined value " << ident.name();
    return env[ident.name()];
  }
  void emitAssignment(const Assign& stmt) {
    auto op = emit(stmt.rhs());
    while (op->output_size() < stmt.idents().size())
      op->add_output();
    int i = 0;
    for (auto ident : stmt.idents()) {
      op->set_output(i++, ident.name());
      map(ident.name(), ident.name());
    }
  }
  void emitIf(const If& stmt) {
    auto cond = getValue(stmt.cond());
    auto op = cur().add_op();
    op->set_type("If");
    op->add_input(cond);
    auto true_branch = op->add_arg();
    true_branch->set_name("then_net");
    auto nd = true_branch->mutable_n();
    net_def_stack.push_back(nd);
    emitStatements(stmt.trueBranch());
    net_def_stack.pop_back();
    if (stmt.falseBranch().size() > 0) {
      auto false_branch = op->add_arg();
      false_branch->set_name("else_net");
      auto nd = false_branch->mutable_n();
      net_def_stack.push_back(nd);
      emitStatements(stmt.falseBranch());
      net_def_stack.pop_back();
    }
  }
  void emitWhile(const While& stmt) {
    std::string loop_var = fresh();
    emitConst(0, loop_var, ""); // it needs a definition before loop
    auto op = cur().add_op();
    op->set_type("While");
    auto cond = op->add_arg();
    cond->set_name("cond_net");
    auto cond_net = cond->mutable_n();

    net_def_stack.push_back(cond_net);
    auto cond_op = emit(stmt.cond());
    cond_op->set_output(0, loop_var);
    net_def_stack.pop_back();

    op->add_input(loop_var);
    auto body = op->add_arg();
    body->set_name("loop_net");
    auto body_net = body->mutable_n();

    net_def_stack.push_back(body_net);
    emitStatements(stmt.body());
    net_def_stack.pop_back();
  }
  const std::string& getValue(const TreeRef& tree) {
    switch (tree->kind()) {
      case TK_IDENT:
        return lookup(Ident(tree));
      default:
        auto op = emit(tree);
        return op->output(0);
    }
  }
  std::string fresh() {
    return std::string("$t") + caffe2::to_string(next_fresh++);
  }
  const char* operatorName(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return "Add";
      case '-':
        if (ninputs == 1)
          return "Negative";
        else
          return "Sub";
      case '*':
        return "Mul";
      case '/':
        return "Div";
      case TK_NE:
        return "NE";
      case TK_EQ:
        return "EQ";
      case '<':
        return "LT";
      case '>':
        return "GT";
      case TK_LE:
        return "LE";
      case TK_GE:
        return "GE";
      default:
        throw std::runtime_error("unknown kind " + caffe2::to_string(kind));
    }
  }
  void fillArg(Argument* arg, const Attribute& attr) {
    std::string name = attr.name().name();
    arg->set_name(name);
    auto value = attr.value();
    // TODO: handle non-float attributes
    switch (value->kind()) {
      case TK_CONST: {
        auto v = value->tree(0)->doubleValue();
        auto f = value->tree(1)->stringValue();
        if (f == "f")
          arg->set_f(v);
        else
          arg->set_i(v);
      } break;
      case TK_LIST:
        for (auto t : value->trees()) {
          auto v = t->tree(0)->doubleValue();
          auto f = t->tree(1)->stringValue();
          if (f == "f")
            arg->add_floats(v);
          else
            arg->add_ints(v);
        }
        break;
    }
  }
  template <typename Trees>
  std::vector<std::string> getValues(const Trees& trees) {
    std::vector<std::string> result;
    for (const auto& tree : trees) {
      result.push_back(getValue(tree));
    }
    return result;
  }
  OperatorDef* emit(const TreeRef& tree) {
    switch (tree->kind()) {
      case TK_IDENT: {
        auto op = cur().add_op();
        op->set_type("Copy");
        op->add_input(lookup(Ident(tree)));
        op->add_output(fresh());
        return op;
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
      case '+': {
        // must be before add_op
        auto values = getValues(tree->trees());
        auto op = cur().add_op();
        op->set_type(operatorName(tree->kind(), tree->trees().size()));
        for (auto& v : values) {
          op->add_input(v);
        }
        op->add_output(fresh());
        return op;
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        // must be before add_op
        auto values = getValues(apply.inputs());
        auto op = cur().add_op();
        op->set_type(apply.name().name());
        for (auto& v : values) {
          op->add_input(v);
        }
        // assume 1 output unless matched to more
        op->add_output(fresh());
        for (auto attribute : apply.attributes()) {
          fillArg(op->add_arg(), attribute);
        }
        return op;
      } break;
      case TK_CONST: {
        return emitConst(
            tree->tree(0)->doubleValue(),
            fresh(),
            tree->tree(1)->stringValue());
      } break;
      default:
        throw ErrorReport(tree) << "NYI: " << tree;
    }
  }
  OperatorDef* emitConst(
      double v,
      const std::string& output,
      const std::string& type_ident) {
    auto op = cur().add_op();
    op->set_type("ConstantFill");
    auto dtype = op->add_arg();
    dtype->set_name("dtype");
    auto value = op->add_arg();
    value->set_name("value");
    if (type_ident == "f") {
      dtype->set_i(TensorProto_DataType_FLOAT);
      value->set_f(v);
    } else if (type_ident == "LL") {
      dtype->set_i(TensorProto_DataType_INT64);
      value->set_i(v);
    } else {
      dtype->set_i(TensorProto_DataType_INT32);
      value->set_i(v);
    }
    auto shape = op->add_arg();
    shape->set_name("shape");
    shape->add_ints(1);
    op->add_output(output);
    return op;
  }
  NetDef& cur() {
    return *net_def_stack.back();
  }
  const Def& def;
  std::unordered_map<std::string, std::string>
      env; // map from name in Def to name in NetDef
  std::vector<NetDef*> net_def_stack;
  int next_fresh = 0;
};

struct CompilationUnitImpl {
  CompilationUnitImpl() {}
  void defineFunction(const Def& def) {
    if (functions.count(def.name().name()) > 0) {
      throw ErrorReport(def) << def.name().name() << " already defined.";
    }
    DefCompiler c(
        def, functions.emplace(def.name().name(), NetDef()).first->second);
    c.run();
  }
  void define(const std::string& str) {
    Parser p(str);
    while (p.lexer().cur().kind != TK_EOF) {
      defineFunction(Def(p.parseFunction()));
    }
  }
  std::unique_ptr<NetBase> createNet(Workspace* ws, const std::string& str) {
    if (functions.count(str) == 0)
      throw ErrorReport() << "undefined function: " << str << "\n";
    auto& net_def = functions[str];
    return caffe2::CreateNet(net_def, ws);
  }

 private:
  std::unordered_map<std::string, NetDef> functions;
};

CompilationUnit::CompilationUnit() : pImpl(new CompilationUnitImpl()) {}

void CompilationUnit::define(const std::string& str) {
  return pImpl->define(str);
}

std::unique_ptr<NetBase> CompilationUnit::createNet(
    Workspace* ws,
    const std::string& str) {
  return pImpl->createNet(ws, str);
}

CompilationUnit::~CompilationUnit() {}

} // namespace script
} // namespace caffe2
