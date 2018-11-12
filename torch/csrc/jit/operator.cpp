#include "ATen/ATen.h"
#include "torch/csrc/jit/alias_info.h"
#include "torch/csrc/jit/script/lexer.h"
#include "torch/csrc/jit/script/tree.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/passes/python_print.h"
#include "torch/csrc/jit/script/error_report.h"

namespace torch { namespace jit {

namespace script {
struct SchemaParser {
  SchemaParser(const std::string& str)
  : L(str) {}

  FunctionSchema parseDeclaration() {
    auto name = L.expect(TK_IDENT).text();
    if(L.nextIf(':')) {
      L.expect(':');
      name = name + "::" + L.expect(TK_IDENT).text();
    }
    std::vector<Argument> arguments;
    std::vector<Argument> returns;
    std::vector<Symbol> writes;
    bool kwarg_only = false;
    bool is_vararg = false;
    size_t idx = 0;
    parseList('(', ',', ')', [&] {
      if(is_vararg)
        throw ErrorReport(L.cur()) << "... must be the last element of the argument list";
      if (L.nextIf('*')) {
        kwarg_only = true;
      } else if(L.nextIf(TK_DOTS)) {
        is_vararg = true;
      } else {
        arguments.push_back(parseArgument(
            idx++, /*is_return=*/false, /*kwarg_only=*/kwarg_only, writes));
      }
    });
    idx = 0;
    L.expect(TK_ARROW);
    if (L.cur().kind == '(') {
      parseList('(', ',', ')', [&] {
        returns.push_back(
            parseArgument(idx++, /*is_return=*/true, /*kwarg_only=*/false, writes));
      });
    } else {
      returns.push_back(
          parseArgument(0, /*is_return=*/true, /*kwarg_only=*/false, writes));
    }
    return FunctionSchema { name, std::move(arguments), std::move(returns),
                            is_vararg, false, std::move(writes) };
  }

  std::vector<FunctionSchema> parseDeclarations() {
    std::vector<FunctionSchema> results;
    do {
      results.push_back(parseDeclaration());
    } while(L.nextIf(TK_NEWLINE));
    L.expect(TK_EOF);
    return results;
  }

  TreeRef parseIdent() {
    return String::create(L.expect(TK_IDENT).text());
  }
  TypePtr parseBaseType() {
    static std::unordered_map<std::string, TypePtr> type_map = {
      {"Generator", GeneratorType::get() },
      {"ScalarType", IntType::get() },
      {"Layout", IntType::get() },
      {"Device", ListType::ofInts() },
      {"Scalar", NumberType::get() },
      {"str", StringType::get() },
      {"float", FloatType::get() },
      {"int", IntType::get() },
      {"bool", BoolType::get() },
    };
    auto tok = L.expect(TK_IDENT);
    auto text = tok.text();
    auto it = type_map.find(text);
    if(it == type_map.end()) {
      if(text.size() > 0 && islower(text[0])) {
        // lower case identifiers that are not otherwise valid types
        // are treated as type variables
        return VarType::create(text);
      }
      throw ErrorReport(tok.range) << "unknown type specifier";
    }
    return it->second;
  }
  static void addToWrites(std::vector<Symbol>& writes, const Symbol& alias_set) {
    auto it = std::find(writes.begin(), writes.end(), alias_set);
    if(it == writes.end())
      writes.push_back(alias_set);
  }
  // Examples:
  // Tensor(a) // Tensor is in set a
  // Tensor(a!) // it is also written to
  // Tensor!  // shorthand for Tensor(fresh_identifier!)
  std::vector<Symbol> parseAliasAnnotation(std::vector<Symbol>& writes) {
    std::vector<Symbol> sets;
    if(L.nextIf('(')) {
      // optional 'alias set annotation'
      sets.push_back(Symbol::fromQualString("alias::"+L.expect(TK_IDENT).text()));
      if(L.nextIf('!')) {
        addToWrites(writes, sets.back());
      }
      L.expect(')');
    } else if(L.nextIf('!')) {
      sets.push_back(Symbol::fromQualString("alias::$"+std::to_string(next_id++)));
      addToWrites(writes, sets.back());
    }
    return sets;
  }
  std::pair<TypePtr, AliasInfo> parseType(std::vector<Symbol>& writes) {
    TypePtr value;
    AliasInfo alias_info;
    if (L.cur().kind == '(') {
      std::vector<TypePtr> types;
      std::vector<AliasInfo> alias_infos;
      parseList('(', ',', ')', [&]{
        auto r = parseType(writes);
        types.push_back(std::move(r.first));
        alias_infos.push_back(std::move(r.second));
      });
      value = TupleType::create(std::move(types));
      alias_info = AliasInfo({}, std::move(alias_infos));
    } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Future") {
      L.next(); // Future
      L.expect('(');
      auto p = parseType(writes);
      auto subtype = std::move(p.first);
      auto subalias = std::move(p.second);
      L.expect(')');
      value = FutureType::create(subtype);
    } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Tensor") {
      L.next();
      value = DynamicType::get();
      alias_info = AliasInfo(parseAliasAnnotation(writes));
    } else {
      value = parseBaseType();
    }
    while(true) {
      if(L.cur().kind == '[' && L.lookahead().kind == ']') {
        L.next(); // [
        L.next(); // ]
        value = ListType::create(value);
        alias_info = AliasInfo(parseAliasAnnotation(writes), {std::move(alias_info)});
      } else if(L.nextIf('?')) {
        value = OptionalType::create(value);
      } else {
        break;
      }
    }
    return std::make_pair(std::move(value), std::move(alias_info));
  }

  Argument parseArgument(size_t idx, bool is_return, bool kwarg_only, std::vector<Symbol>& writes) {
    Argument result;
    auto p = parseType(writes);
    auto type = std::move(p.first);
    auto alias_info = std::move(p.second);
    c10::optional<int32_t> N;
    c10::optional<IValue> default_value;
    c10::optional<std::string> alias_set;
    std::string name;
    if(L.nextIf('[')) {
      // note: an array with a size hint can only occur at the Argument level
      type = ListType::create(type);
      N = std::stoll(L.expect(TK_NUMBER).text());
      L.expect(']');
      alias_info = AliasInfo(parseAliasAnnotation(writes), {std::move(alias_info)});
    }
    if(is_return) {
      // optionally named return values
      if(L.cur().kind == TK_IDENT) {
        name = L.next().text();
      } else {
        name = "ret" + std::to_string(idx);
      }
    } else {
      name = L.expect(TK_IDENT).text();
      if(L.nextIf('=')) {
        default_value = parseDefaultValue(type, N);
      }
    }
    return Argument(std::move(name), std::move(type), N, std::move(default_value), !is_return && kwarg_only, std::move(alias_info));
  }
  IValue parseSingleConstant(TypeKind kind) {
    switch(L.cur().kind) {
      case TK_TRUE:
        L.next();
        return true;
      case TK_FALSE:
        L.next();
        return false;
      case TK_NONE:
        L.next();
        return IValue();
      case TK_IDENT: {
        auto tok = L.next();
        auto text = tok.text();
        if("float" == text) {
          return static_cast<int64_t>(at::kFloat);
        } else if("cpu" == text) {
          return static_cast<int64_t>(at::Device::Type::CPU);
        } else if("strided" == text) {
          return static_cast<int64_t>(at::kStrided);
        } else if("Mean" == text) {
          return static_cast<int64_t>(Reduction::Mean);
        } else {
          throw ErrorReport(L.cur().range) << "invalid numeric default value";
        }
      }
      default:
        std::string n;
        if(L.nextIf('-'))
          n = "-" + L.expect(TK_NUMBER).text();
        else
          n = L.expect(TK_NUMBER).text();
        if(kind == TypeKind::FloatType || n.find(".") != std::string::npos || n.find("e") != std::string::npos) {
          return std::stod(n);
        } else {
          int64_t v = std::stoll(n);
          return v;
        }
    }
  }
  IValue convertToList(TypeKind kind, const SourceRange& range, std::vector<IValue> vs) {
    switch(kind) {
        case TypeKind::FloatType:
          return fmap(vs, [](IValue v) {
            return v.toDouble();
          });
        case TypeKind::IntType:
          return fmap(vs, [](IValue v) {
            return v.toInt();
          });
        case TypeKind::BoolType:
          return fmap(vs, [](IValue v) {
            return v.toBool();
          });
        default:
          throw ErrorReport(range) << "lists are only supported for float or int types.";
      }
  }
  IValue parseConstantList(TypeKind kind) {
    auto tok = L.expect('[');
    std::vector<IValue> vs;
    if(L.cur().kind != ']') {
      do {
        vs.push_back(parseSingleConstant(kind));
      } while(L.nextIf(','));
    }
    L.expect(']');
    return convertToList(kind, tok.range, std::move(vs));
  }

  IValue parseTensorDefault(const SourceRange& range) {
    L.expect(TK_NONE);
    return IValue();
  }
  IValue parseDefaultValue(TypePtr arg_type, c10::optional<int32_t> arg_N) {
    auto range = L.cur().range;
    switch(arg_type->kind()) {
      case TypeKind::DynamicType:
      case TypeKind::GeneratorType: {
        return parseTensorDefault(range);
      }  break;
      case TypeKind::OptionalType:
      case TypeKind::NumberType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
        return parseSingleConstant(arg_type->kind());
        break;
      case TypeKind::ListType: {
        auto elem_kind = arg_type->cast<ListType>()->getElementType();
        if(L.cur().kind == TK_IDENT) {
          return parseTensorDefault(range);
        } else if(arg_N && L.cur().kind != '[') {
          IValue v = parseSingleConstant(elem_kind->kind());
          std::vector<IValue> repeated(*arg_N, v);
          return convertToList(elem_kind->kind(), range, repeated);
        } else {
          return parseConstantList(elem_kind->kind());
        }
      } break;
      default:
        throw ErrorReport(range) << "unexpected type, file a bug report";
    }
    return IValue(); // silence warnings
  }

  void parseList(int begin, int sep, int end, std::function<void()> callback) {
    auto r = L.cur().range;
    if (begin != TK_NOTHING)
      L.expect(begin);
    if (L.cur().kind != end) {
      do {
        callback();
      } while (L.nextIf(sep));
    }
    if (end != TK_NOTHING)
      L.expect(end);
  }
  Lexer L;
  size_t next_id = 0;
};

} // namespace script

namespace {

std::string canonicalSchemaString(const FunctionSchema& schema) {
  std::ostringstream out;

  out << schema.name();
  out << "(";

  bool seen_kwarg_only = false;
  for(size_t i = 0; i < schema.arguments().size(); ++i) {
    if (i > 0) out << ", ";
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    const auto & arg = schema.arguments()[i];
    out << arg.type()->str() << " " << arg.name();
  }

  out << ") -> ";
  if (schema.returns().size() == 1) {
    out << schema.returns().at(0).type()->str();
  } else if (schema.returns().size() > 1) {
    out << "(";
    for (size_t i = 0; i < schema.returns().size(); ++i) {
      if (i > 0) out << ", ";
      out << schema.returns()[i].type()->str();
    }
    out << ")";
  }
  return out.str();
}

using OperatorMap = std::unordered_map<Symbol, std::vector<std::shared_ptr<Operator>>>;
struct OperatorRegistry  {
private:
  std::mutex lock;
  OperatorMap operators;
  // list of operators whose schema have not yet been parsed, and must
  // be registered before any call to lookup an opeator
  std::vector<std::shared_ptr<Operator>> to_register;
  // Those two maps are used to implement lookupByLiteral, which is needed for the n->match(...) calls.
  // Basically, every function schema is assigned a unique string you can use to match it. However,
  // parsing those strings or comparing and hashing them character by character would be very slow, so
  // we use a trick here! Every string literal in your program is guaranteed to have static storage
  // duration and so its address won't change at runtime. This allows us to memoize answers for every
  // pointer, which is done by the operators_by_sig_literal map. Still, this map is initially
  // empty, and so we still need to do the complete string matching at the first time, which is implemented
  // by performing a lookup in the operators_by_sig map.
  std::unordered_map<std::string, std::shared_ptr<Operator>> operators_by_sig;
  std::unordered_map<const char *, std::shared_ptr<Operator>> operators_by_sig_literal;

  // XXX - caller must be holding lock
  void registerPendingOperators() {
    for(auto op : to_register) {
      Symbol sym = Symbol::fromQualString(op->schema().name());
      operators[sym].push_back(op);
      operators_by_sig[canonicalSchemaString(op->schema())] = op;
    }
    to_register.clear();
  }

public:
  void registerOperator(Operator&& op) {
    std::lock_guard<std::mutex> guard(lock);
    to_register.push_back(std::make_shared<Operator>(std::move(op)));
  }

  const std::shared_ptr<Operator>& lookupByLiteral(const char * name) {
    std::lock_guard<std::mutex> guard(lock);
    registerPendingOperators();
    auto it = operators_by_sig_literal.find(name);
    if (it == operators_by_sig_literal.end()) {
      auto op_ptr_it = operators_by_sig.find(name);
      // Handy debugging code that dumps all operators we know about on mismatch
#if 0
      if (op_ptr_it == operators_by_sig.end()) {
        for (auto & entry : operators_by_sig) {
          std::cout << entry.first << std::endl;
        }
      }
#endif
      JIT_ASSERTM(op_ptr_it != operators_by_sig.end(), "Couldn't find an operator for ", name);
      it = operators_by_sig_literal.emplace_hint(it, name, op_ptr_it->second);
    }
    return it->second;
  }


  const std::vector<std::shared_ptr<Operator>>& getOperators(Symbol name) {
    std::lock_guard<std::mutex> guard(lock);
    registerPendingOperators();
    static std::vector<std::shared_ptr<Operator>> empty;
    auto it = operators.find(name);
    if(it != operators.end())
      return it->second;
    return empty;
  }
};

OperatorRegistry& getRegistry() {
  static OperatorRegistry r;
  return r;
}

} // anonymous namespace

void registerOperator(Operator&& op) {
  if(op.schema().is_varret()) {
    Symbol s = Symbol::fromQualString(op.schema().name());
    if (!printerHasSpecialCaseFor(s)) {
      std::cout << c10::str(
          "missing special case in python printer for non-schematized operator ",
          op.schema().name(),
          ". File a bug to add a case for this operator.\n");
    }
  }

  getRegistry().registerOperator(std::move(op));
}

const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol name) {
  return getRegistry().getOperators(name);
}

Operator& sig(const char *signature) {
  return *getRegistry().lookupByLiteral(signature);
}

FunctionSchema parseSchema(const std::string& schema) {
  return script::SchemaParser(schema).parseDeclarations().at(0);
}

bool Operator::matches(const Node* node) const {
  // wrong name
  if (node->kind().toQualString() != schema().name()) {
    return false;
  }
  at::ArrayRef<const Value*> actuals = node->inputs();
  const auto& formals = schema().arguments();

  // not enough inputs
  if(actuals.size() < formals.size())
    return false;


  TypeEnv type_env;
  for(size_t i = 0; i < formals.size(); ++i) {
    try {
      TypePtr formal = matchTypeVariables(formals[i].type(), actuals[i]->type(), type_env);
      // mismatched input type
      if (!actuals[i]->type()->isSubtypeOf(formal)) {
        return false;
      }
    } catch(TypeMatchError& err) {
      return false;
    }
  }

  // too many inputs
  if(!schema().is_vararg() && actuals.size() != formals.size()) {
    // std::cout << "not all inputs used\n" << input_i << " " << inputs_size << "\n";
    return false;
  }

  return true;
}

std::shared_ptr<Operator> findOperatorFor(const Node* node) {
  const auto& candidates = getAllOperatorsFor(node->kind());
  for(const auto& candidate : candidates) {
    if(candidate->matches(node)) {
      return candidate;
    }
  }
  return nullptr;
}

const Operator& getOperatorFor(const Node* node) {
  auto op = findOperatorFor(node);
  if(op)
    return *op;

  auto er = script::ErrorReport(node->getSourceLocation());
  er << "Schema not found for node. File a bug report.\n";
  er << "Node: " << *node << "\n";
  er << "Input types:";
  for(size_t i = 0; i < node->inputs().size(); ++i) {
    if(i > 0)
      er << ", ";
    er << *node->inputs()[i]->type();
  }
  er << "\ncandidates were:\n";
  const auto& candidates = getAllOperatorsFor(node->kind());
  for(auto & candidate : candidates) {
    er << "  " << candidate->schema() << "\n";
  }
  throw er;
}


OperatorSet::OperatorSet(std::initializer_list<const char *> sig_literals) {
  auto & registry = getRegistry();
  for (const char * sig : sig_literals) {
    auto op = registry.lookupByLiteral(sig);
    ops[Symbol::fromQualString(op->schema().name())].push_back(op);
  }
}

Operator* OperatorSet::find(const Node *n) const  {
  auto it = ops.find(n->kind());
  if (it == ops.end()) {
    return nullptr;
  }
  for (auto & op : it->second) {
    if (op->matches(n)) {
      return op.get();
    }
  }
  return nullptr;
}

}}
